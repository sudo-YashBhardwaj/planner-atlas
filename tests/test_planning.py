import pytest
import torch

from planner_atlas.data import ActionStats, denormalize_actions, unpack_action_blocks
from planner_atlas.planning import cem, model_action_bounds, random_shooting

# Asymmetric bounds, and a y-range narrower than the proposal's unit std.
STATS = ActionStats(mean=torch.tensor([0.3, -0.2]), std=torch.tensor([0.5, 2.0]))
BOUNDS = model_action_bounds(STATS, device="cpu")
LATENT = GOAL = torch.zeros(1, 192)


class FakeModel:
    """Stands in for ReferenceLeWM.cost: an objective over plans that records every candidate."""

    def __init__(self, objective):
        self.objective = objective  # plans [S, H, 10] -> costs [S]
        self.candidates: list[torch.Tensor] = []

    def cost(self, latent, goal, action_blocks):
        assert latent.shape == goal.shape == (1, 192) and action_blocks.shape[0] == 1
        self.candidates.append(action_blocks[0])
        return self.objective(action_blocks[0])[None]

    def evaluated(self) -> torch.Tensor:
        return torch.cat(self.candidates)


def quadratic(target):
    return lambda plans: (plans - target).square().sum(dim=(1, 2))


def run_random_shooting(model, **overrides):
    kwargs = {
        "horizon": 2,
        "num_samples": 128,
        "bounds": BOUNDS,
        "generator": torch.Generator().manual_seed(0),
    }
    return random_shooting(model, LATENT, GOAL, **(kwargs | overrides))


def run_cem(model, **overrides):
    kwargs = {
        "horizon": 2,
        "num_samples": 64,
        "iterations": 4,
        "elite_fraction": 0.1,
        "bounds": BOUNDS,
        "generator": torch.Generator().manual_seed(0),
    }
    return cem(model, LATENT, GOAL, **(kwargs | overrides))


PLANNERS = [run_random_shooting, run_cem]


def test_model_action_bounds_are_the_raw_bounds_in_model_space() -> None:
    low, high = BOUNDS
    assert low.shape == high.shape == (10,)
    raw_low, raw_high = (denormalize_actions(unpack_action_blocks(b), STATS) for b in BOUNDS)
    torch.testing.assert_close(raw_low, torch.full((5, 2), -1.0))
    torch.testing.assert_close(raw_high, torch.full((5, 2), 1.0))


@pytest.mark.parametrize("plan", PLANNERS)
def test_every_evaluated_plan_is_executable(plan) -> None:
    model = FakeModel(quadratic(torch.full((2, 10), 5.0)))  # pulls candidates onto the bounds
    plan(model)
    plans = model.evaluated()
    low, high = BOUNDS
    assert ((plans >= low) & (plans <= high)).all()
    assert denormalize_actions(unpack_action_blocks(plans), STATS).abs().max() <= 1 + 1e-6


@pytest.mark.parametrize("plan", PLANNERS)
def test_returns_the_lowest_cost_evaluated_plan(plan) -> None:
    model = FakeModel(quadratic(torch.zeros(2, 10)))
    result = plan(model)
    plans = model.evaluated()
    costs = model.objective(plans)
    assert result.actions.shape == (2, 10)
    assert torch.equal(result.actions, plans[costs.argmin()])
    assert result.predicted_cost == costs.min().item()


def test_random_shooting_evaluates_exactly_its_budget_in_chunks() -> None:
    model = FakeModel(quadratic(torch.zeros(2, 10)))
    result = run_random_shooting(model, num_samples=100, score_batch_size=32)
    assert [len(chunk) for chunk in model.candidates] == [32, 32, 32, 4]
    assert result.model_evaluations == 100
    assert result.iterations == ()
    assert torch.equal(result.final.mean, result.initial.mean)


def test_cem_evaluates_samples_times_iterations() -> None:
    model = FakeModel(quadratic(torch.zeros(2, 10)))
    result = run_cem(model, num_samples=50, iterations=3)
    assert [len(chunk) for chunk in model.candidates] == [50] * 3
    assert result.model_evaluations == 150
    assert len(result.iterations) == 3
    assert result.final.mean.shape == result.final.std.shape == (2, 10)
    assert torch.equal(result.initial.mean, torch.zeros(2, 10))
    assert torch.equal(result.initial.std, torch.ones(2, 10))


def test_random_shooting_finds_a_low_cost_plan() -> None:
    model = FakeModel(quadratic(torch.zeros(1, 10)))
    result = run_random_shooting(model, horizon=1, num_samples=4096)
    assert result.predicted_cost < 0.25 * model.objective(model.evaluated()).median()


def test_cem_converges_to_the_bounded_optimum() -> None:
    target = torch.linspace(-3.0, 3.0, 20).view(2, 10)  # partly outside the bounds
    result = run_cem(FakeModel(quadratic(target)), num_samples=256, iterations=30)
    low, high = BOUNDS
    torch.testing.assert_close(result.actions, target.clamp(low, high), atol=0.15, rtol=0)


def test_cem_returns_best_evaluated_plan_not_its_final_mean() -> None:
    # Minima at x0 = -1 and x0 = +1: the elites straddle both, so their mean lands in between.
    def objective(plans):
        return torch.minimum((plans[:, 0, 0] - 1).square(), (plans[:, 0, 0] + 1).square())

    result = run_cem(FakeModel(objective), horizon=1, num_samples=1024, iterations=1)
    assert result.predicted_cost < 0.01
    assert objective(result.final.mean[None]) > 0.5


@pytest.mark.parametrize("plan", PLANNERS)
def test_same_seed_gives_the_same_plan(plan) -> None:
    first, second = (plan(FakeModel(quadratic(torch.ones(2, 10)))) for _ in range(2))
    assert torch.equal(first.actions, second.actions)
    assert first.predicted_cost == second.predicted_cost
    assert torch.equal(first.final.std, second.final.std)


def test_chunked_scoring_does_not_change_the_plan() -> None:
    small, large = (
        run_cem(FakeModel(quadratic(torch.ones(2, 10))), score_batch_size=size)
        for size in (7, 1024)
    )
    assert torch.equal(small.actions, large.actions)
    assert small.predicted_cost == large.predicted_cost
    assert torch.equal(small.final.std, large.final.std)


@pytest.mark.parametrize(
    "overrides",
    [{"elite_fraction": 0.0}, {"elite_fraction": 1.5}, {"iterations": 0}, {"num_samples": 0}],
)
def test_cem_rejects_invalid_parameters(overrides) -> None:
    with pytest.raises(ValueError):
        run_cem(FakeModel(quadratic(torch.zeros(2, 10))), **overrides)
