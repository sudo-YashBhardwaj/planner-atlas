"""Bootstrap ensembles of repaired dynamics, and the uncertainty they hand a planner.

Every member is the released model carrying its own trained dynamics, so all members predict in
the frozen latent coordinate system and their costs are directly comparable with each other and
with the Atlas. Members share the frozen encoder and projector in memory: only their dynamics
differ, and nothing here encodes pixels anyway.

Uncertainty of a candidate plan is the population variance across members of the predicted
terminal goal cost, that is, disagreement about the quantity the planner actually optimizes rather
than about the latent vector.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch

from planner_atlas.training import load_dynamics


class LatentModel(Protocol):
    """What planning and acquisition consume from a latent world model."""

    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor: ...

    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor: ...


class DynamicsEnsemble:
    """Members of one latent coordinate system that differ only in their trained dynamics."""

    def __init__(self, members: Sequence[LatentModel]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = tuple(members)

    def __len__(self) -> int:
        return len(self.members)

    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        """Predicted latents [E, B, S, T + 1, 192], one rollout per member."""
        return torch.stack([member.rollout(latent, action_blocks) for member in self.members])

    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        """Terminal goal cost [E, B, S] of every candidate, per member."""
        return torch.stack([member.cost(latent, goal, action_blocks) for member in self.members])

    def uncertainty(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        """Task-relevant epistemic uncertainty [B, S] of every candidate plan."""
        return cost_uncertainty(self.cost(latent, goal, action_blocks))


def cost_uncertainty(costs: torch.Tensor) -> torch.Tensor:
    """Population variance across ensemble members of terminal costs [E, B, S] -> [B, S]."""
    return costs.var(dim=0, unbiased=False)


def load_ensemble(
    base_checkpoint: Path, dynamics: Sequence[Path], *, device: torch.device | str
) -> DynamicsEnsemble:
    """Load trained members over one base checkpoint, sharing its frozen representation."""
    members = [load_dynamics(base_checkpoint, path, device=device) for path in dynamics]
    for member in members[1:]:
        member.encoder, member.projector = members[0].encoder, members[0].projector
    return DynamicsEnsemble(members)
