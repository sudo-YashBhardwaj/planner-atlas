import pytest
import torch

from planner_atlas.models.reference_lewm import LATENT_DIM
from planner_atlas.uncertainty import DynamicsEnsemble, check_provenance, cost_uncertainty


class FakeMember:
    """A member whose predicted cost is its own constant, added to a per-candidate pattern."""

    def __init__(self, offset: float) -> None:
        self.offset = offset

    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        b, s, t, _ = action_blocks.shape
        return torch.full((b, s, t + 1, LATENT_DIM), self.offset)

    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        pattern = torch.arange(action_blocks.shape[1], dtype=torch.float32)
        return self.offset + pattern.expand(action_blocks.shape[0], -1)


def test_ensemble_shapes() -> None:
    ensemble = DynamicsEnsemble([FakeMember(float(e)) for e in range(5)])
    latent, goal = torch.zeros(2, LATENT_DIM), torch.zeros(2, LATENT_DIM)
    blocks = torch.zeros(2, 7, 3, 10)

    assert len(ensemble) == 5
    assert ensemble.rollout(latent, blocks).shape == (5, 2, 7, 4, LATENT_DIM)
    assert ensemble.cost(latent, goal, blocks).shape == (5, 2, 7)
    assert ensemble.uncertainty(latent, goal, blocks).shape == (2, 7)


def test_identical_members_leave_no_uncertainty() -> None:
    ensemble = DynamicsEnsemble([FakeMember(1.5) for _ in range(5)])
    latent, goal = torch.zeros(3, LATENT_DIM), torch.zeros(3, LATENT_DIM)
    uncertainty = ensemble.uncertainty(latent, goal, torch.zeros(3, 4, 2, 10))
    assert torch.equal(uncertainty, torch.zeros(3, 4))


def test_uncertainty_is_the_population_variance_of_member_costs() -> None:
    costs = torch.tensor([[[1.0, 2.0]], [[3.0, 2.0]], [[5.0, 8.0]]])  # [E=3, B=1, S=2]
    # means 3 and 4; population variances (4 + 0 + 4) / 3 and (4 + 4 + 16) / 3
    torch.testing.assert_close(cost_uncertainty(costs), torch.tensor([[8 / 3, 8.0]]))

    ensemble = DynamicsEnsemble([FakeMember(0.0), FakeMember(4.0)])
    latent = goal = torch.zeros(1, LATENT_DIM)
    uncertainty = ensemble.uncertainty(latent, goal, torch.zeros(1, 3, 2, 10))
    # members differ by a constant 4, so every candidate carries variance 4
    torch.testing.assert_close(uncertainty, torch.full((1, 3), 4.0))


def test_members_must_agree_on_their_provenance() -> None:
    """A member trained on another episode split would leak evaluation episodes into selection."""
    shared = {"split_seed": 0, "validation_fraction": 0.1, "cache_identity": {"a": 1}}
    check_provenance([shared, dict(shared)], tuple(shared))  # agreeing members pass
    with pytest.raises(ValueError, match="split_seed"):
        check_provenance([shared, {**shared, "split_seed": 1}], tuple(shared))
    with pytest.raises(ValueError, match="cache_identity"):
        check_provenance([shared, {**shared, "cache_identity": {"a": 2}}], tuple(shared))
    with pytest.raises(ValueError):
        check_provenance([], ("split_seed",))


def test_an_ensemble_needs_a_member() -> None:
    with pytest.raises(ValueError):
        DynamicsEnsemble([])
