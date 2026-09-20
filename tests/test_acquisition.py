import numpy as np
import pytest
import torch

from planner_atlas.acquisition import (
    Selection,
    acquire,
    load_acquisitions,
    planner_selector,
    plans_for_budget,
    random_selector,
    save_acquisitions,
    uncertainty_selector,
)
from planner_atlas.models.reference_lewm import ACTION_DIM, LATENT_DIM
from planner_atlas.planning import random_shooting
from planner_atlas.uncertainty import DynamicsEnsemble, cost_uncertainty

HORIZON = 5
BOUNDS = (torch.full((ACTION_DIM,), -2.0), torch.full((ACTION_DIM,), 2.0))


class FakeModel:
    """Latents are the brightness of a frame; a block advances a latent by the sum of its actions."""

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        brightness = observations.mean(dim=(2, 3, 4))
        return brightness[..., None].expand(*observations.shape[:2], LATENT_DIM).clone()

    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        steps = action_blocks.sum(-1).cumsum(-1)  # [B, S, T]
        start = torch.zeros(*steps.shape[:2], 1)
        return latent[:, None, None] + torch.cat([start, steps], dim=-1)[..., None]

    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        return (self.rollout(latent, action_blocks)[:, :, -1] - goal[:, None]).square().sum(-1)


class SpreadMember:
    """A member that scales the plan's action sum, so members disagree most on the largest plans."""

    def __init__(self, scale: float) -> None:
        self.scale = scale

    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        raise AssertionError("uncertainty must come from costs")

    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        return self.scale * action_blocks.sum(dim=(2, 3))


def frames_of(plan: torch.Tensor) -> np.ndarray:
    """Frames whose brightness follows the plan, so executions differ visibly."""
    shades = (plan.sum(-1).abs() * 10).clamp(0, 255).to(torch.uint8).numpy()
    return np.stack([np.full((224, 224, 3), shade, dtype=np.uint8) for shade in shades])


def executor(task_cost: float = 1.0, log: list | None = None):
    """An execution that reports task_cost; nothing about it may reach a selector."""

    def execute(plan: torch.Tensor):
        if log is not None:
            log.append(plan)
        return frames_of(plan), {"task_cost": task_cost, "success": False}

    return execute


def acquire_once(select, execute, *, strategy: str = "test"):
    return acquire(
        strategy,
        select,
        execute,
        model=FakeModel(),
        latent=torch.zeros(1, LATENT_DIM),
        goal=torch.full((1, LATENT_DIM), 3.0),
        case=0,
        episode=7,
        start_step=13,
    )


def test_budget_is_counted_in_true_transitions() -> None:
    assert plans_for_budget(500, horizon=HORIZON) == 20  # 25 transitions per plan
    for budget in (0, -25, 30, 24):
        with pytest.raises(ValueError):
            plans_for_budget(budget, horizon=HORIZON)

    generator = torch.Generator().manual_seed(0)
    select = random_selector(horizon=HORIZON, bounds=BOUNDS, generator=generator)
    acquisitions = [acquire_once(select, executor()) for _ in range(4)]
    assert all(one.transitions == HORIZON * 5 for one in acquisitions)
    assert sum(one.transitions for one in acquisitions) == 100


def test_uncertainty_acquisition_takes_the_most_uncertain_candidate() -> None:
    ensemble = DynamicsEnsemble([SpreadMember(scale) for scale in (0.5, 1.0, 2.0)])
    generator = torch.Generator().manual_seed(0)
    select = uncertainty_selector(
        horizon=HORIZON,
        bounds=BOUNDS,
        generator=generator,
        ensemble=ensemble,
        num_candidates=16,
    )
    latent, goal = torch.zeros(1, LATENT_DIM), torch.zeros(1, LATENT_DIM)

    # the candidates the selector saw, redrawn from the same generator state
    replay = torch.Generator().manual_seed(0)
    from planner_atlas.planning import initial_proposal, sample_proposal

    plans = sample_proposal(
        initial_proposal(HORIZON), num_samples=16, bounds=BOUNDS, generator=replay
    )
    selection = select(latent, goal)

    canonical = cost_uncertainty(ensemble.cost(latent, goal, plans[None]))[0]
    assert selection.candidate == int(canonical.argmax())
    assert selection.diagnostics["uncertainty"] == pytest.approx(float(canonical.max()))
    torch.testing.assert_close(selection.plan, plans[selection.candidate])
    assert selection.model_evaluations == 16 * len(ensemble)


def test_audit_and_acquisition_share_one_uncertainty_score() -> None:
    """The number the selector records is the canonical score, not a second implementation."""
    ensemble = DynamicsEnsemble([SpreadMember(scale) for scale in (0.5, 1.0, 2.0)])
    select = uncertainty_selector(
        horizon=HORIZON,
        bounds=BOUNDS,
        generator=torch.Generator().manual_seed(3),
        ensemble=ensemble,
        num_candidates=8,
    )
    latent, goal = torch.zeros(1, LATENT_DIM), torch.zeros(1, LATENT_DIM)
    selection = select(latent, goal)

    audited = ensemble.uncertainty(latent, goal, selection.plan[None, None])[0, 0]
    assert selection.diagnostics["uncertainty"] == pytest.approx(float(audited))
    costs = ensemble.cost(latent, goal, selection.plan[None, None])
    assert float(audited) == pytest.approx(float(costs.var(dim=0, unbiased=False)[0, 0]))


def test_planner_acquisition_takes_the_lowest_predicted_cost() -> None:
    model = FakeModel()
    latent, goal = torch.zeros(1, LATENT_DIM), torch.full((1, LATENT_DIM), 3.0)

    def plan(latent, goal):
        return random_shooting(
            model,
            latent,
            goal,
            horizon=HORIZON,
            num_samples=32,
            bounds=BOUNDS,
            generator=torch.Generator().manual_seed(5),
        )

    selection = planner_selector(plan)(latent, goal)

    result = plan(latent, goal)
    torch.testing.assert_close(selection.plan, result.actions)
    assert selection.diagnostics["predicted_cost"] == pytest.approx(result.predicted_cost)
    # and that is the argmin of what the planner scored
    candidates = torch.stack([result.actions, torch.zeros_like(result.actions)])
    costs = model.cost(latent, goal, candidates[None])[0]
    assert selection.diagnostics["predicted_cost"] <= float(costs[1]) + 1e-6


def test_selection_cannot_see_what_execution_reveals() -> None:
    executed: list[torch.Tensor] = []

    def select(latent, goal) -> Selection:
        assert not executed  # nothing has been paid for when the choice is made
        return Selection(torch.full((HORIZON, ACTION_DIM), 0.25), 0, {"score": 1.0}, 0)

    cheap = acquire_once(select, executor(task_cost=0.01, log=executed))
    executed.clear()
    expensive = acquire_once(select, executor(task_cost=99.0, log=executed))

    np.testing.assert_array_equal(cheap.blocks, expensive.blocks)  # the same choice either way
    assert cheap.selection == expensive.selection
    assert cheap.outcome["task_cost"] == 0.01 and expensive.outcome["task_cost"] == 99.0
    assert not set(cheap.selection) & set(cheap.outcome)  # selection and hindsight stay apart


def test_acquisition_records_the_trajectory_and_its_hindsight(tmp_path) -> None:
    generator = torch.Generator().manual_seed(1)
    select = random_selector(horizon=HORIZON, bounds=BOUNDS, generator=generator)
    acquisitions = [acquire_once(select, executor()) for _ in range(3)]
    one = acquisitions[0]

    assert one.latents.shape == (HORIZON + 1, LATENT_DIM)  # the start, then one per block
    assert one.blocks.shape == (HORIZON, ACTION_DIM)
    assert {"predicted_cost", "realized_cost", "optimism", "rollout_error"} <= set(one.outcome)
    assert one.outcome["optimism"] == pytest.approx(
        one.outcome["realized_cost"] - one.outcome["predicted_cost"]
    )
    assert one.record["episode"] == 7 and one.record["outcome_task_cost"] == 1.0

    path = tmp_path / "acquired.h5"
    save_acquisitions(path, acquisitions, {"strategy": "test", "acquisition_round": 0})
    latents, blocks, transitions = load_acquisitions(path)
    assert latents.shape == (3, HORIZON + 1, LATENT_DIM) and transitions == 3 * HORIZON * 5
    np.testing.assert_array_equal(blocks[0], one.blocks)
    records = [line for line in path.with_suffix(".jsonl").read_text().splitlines()]
    assert len(records) == 3 and '"acquisition_round": 0' in records[0]


def test_acquisition_is_deterministic() -> None:
    def run():
        generator = torch.Generator().manual_seed(4)
        select = random_selector(horizon=HORIZON, bounds=BOUNDS, generator=generator)
        return [acquire_once(select, executor()) for _ in range(3)]

    for first, second in zip(run(), run(), strict=True):
        np.testing.assert_array_equal(first.blocks, second.blocks)
        np.testing.assert_array_equal(first.latents, second.latents)
        assert first.outcome == second.outcome
