import numpy as np
import pytest
import torch

from planner_atlas.atlas import atlas_metrics, audit_samples, pairwise_rank_agreement
from planner_atlas.evaluation import CandidateEvaluation
from planner_atlas.planning import Proposal, cem, random_shooting

BOUNDS = (torch.full((10,), -2.0), torch.full((10,), 2.0))
INITIAL = Proposal(torch.zeros(3, 10), torch.ones(3, 10))
FINAL = Proposal(torch.full((3, 10), 0.5), torch.full((3, 10), 0.1))


def evaluation(predicted, realized, rollout_error, distance) -> CandidateEvaluation:
    n = len(predicted)
    return CandidateEvaluation(
        predicted_cost=np.array(predicted, dtype=float),
        realized_cost=np.array(realized, dtype=float),
        rollout_error=np.array(rollout_error, dtype=float),
        terminal_error=np.zeros(n),
        distance=np.array(distance, dtype=float),
        reached=np.zeros(n, dtype=bool),
    )


def test_pairwise_rank_agreement() -> None:
    order = np.array([1.0, 2.0, 3.0, 4.0])
    assert pairwise_rank_agreement(order, order) == 1.0
    assert pairwise_rank_agreement(order, -order) == -1.0
    # pairs tied in either ranking are not compared
    assert pairwise_rank_agreement(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) == 1.0
    assert pairwise_rank_agreement(np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 1.0])) == -1.0
    assert np.isnan(pairwise_rank_agreement(np.ones(3), np.arange(3.0)))


def test_atlas_metrics_on_hand_computed_values() -> None:
    selected = evaluation([1.0], [5.0], [0.3], [10.0])  # optimism 4, within the success radius
    initial = evaluation(
        predicted=[1.0, 2.0, 3.0, 4.0],
        realized=[2.0, 1.0, 3.0, 4.0],  # optimism [1, -1, 0, 0]
        rollout_error=[0.1, 0.2, 0.3, 0.4],
        distance=[4.0, 3.0, 1.0, 2.0],
    )
    final = evaluation([1.0] * 4, [3.0] * 4, [0.5] * 4, [5.0] * 4)  # optimism 2

    metrics = atlas_metrics(selected, initial, final)
    assert metrics["selected_optimism"] == 4.0
    assert metrics["selected_task_cost"] == 10.0 / 224.0 and metrics["selected_success"] is True
    assert metrics["q0_mean_optimism"] == 0.0
    assert metrics["q0_mean_rollout_error"] == pytest.approx(0.25)
    assert metrics["selection_amplification"] == 4.0
    assert metrics["qfinal_mean_optimism"] == 2.0 and metrics["adaptive_optimism_shift"] == 2.0
    assert metrics["adaptive_rollout_error_shift"] == pytest.approx(0.25)
    # 6 pairs each: dynamics 5 concordant / 1 discordant, objective 2 / 4, end to end 1 / 5
    assert metrics["q0_rank_dynamics"] == pytest.approx(2 / 3)
    assert metrics["q0_rank_objective"] == pytest.approx(-1 / 3)
    assert metrics["q0_rank_end_to_end"] == pytest.approx(-2 / 3)

    shooting = atlas_metrics(selected, initial)
    assert shooting["selection_amplification"] == 4.0
    assert not any(key.startswith(("qfinal", "adaptive")) for key in shooting)


class QuadraticModel:
    def cost(self, latent, goal, action_blocks):
        return action_blocks.square().sum(dim=(2, 3))


def test_paired_planners_share_one_initial_proposal_audit() -> None:
    kwargs = {"horizon": 3, "bounds": BOUNDS}
    latent = goal = torch.zeros(1, 192)
    generators = [torch.Generator().manual_seed(0) for _ in range(2)]
    shooting = random_shooting(
        QuadraticModel(), latent, goal, **kwargs, num_samples=128, generator=generators[0]
    )
    adaptive = cem(
        QuadraticModel(),
        latent,
        goal,
        **kwargs,
        num_samples=64,
        iterations=2,
        elite_fraction=0.1,
        generator=generators[1],
    )
    audits = [
        audit_samples(result.initial, num_samples=32, bounds=BOUNDS, seed=7)
        for result in (shooting, adaptive)
    ]
    assert torch.equal(audits[0], audits[1])


def test_initial_and_final_audits_share_their_noise() -> None:
    wide = (torch.full((10,), -100.0), torch.full((10,), 100.0))  # nothing is clamped
    initial = audit_samples(INITIAL, num_samples=8, bounds=wide, seed=3)
    final = audit_samples(FINAL, num_samples=8, bounds=wide, seed=3)
    torch.testing.assert_close(
        (initial - INITIAL.mean) / INITIAL.std, (final - FINAL.mean) / FINAL.std
    )
