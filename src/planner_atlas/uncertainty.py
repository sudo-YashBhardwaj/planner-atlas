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

from planner_atlas.training import load_dynamics, load_metadata

PROVENANCE = ("split_seed", "validation_fraction", "cache_identity", "base_checkpoint")


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


def check_provenance(metadata: Sequence[dict], fields: Sequence[str]) -> None:
    """Refuse a set of artifacts that disagree on how they were made."""
    if not metadata:
        raise ValueError("nothing to check")
    for field in fields:
        values = [str(one.get(field)) for one in metadata]
        if len(set(values)) > 1:
            raise ValueError(f"artifacts disagree on {field}: {sorted(set(values))[:2]}")


def cost_uncertainty(costs: torch.Tensor) -> torch.Tensor:
    """Population variance across ensemble members of terminal costs [E, B, S] -> [B, S]."""
    return costs.var(dim=0, unbiased=False)


def load_ensemble(
    base_checkpoint: Path,
    dynamics: Sequence[Path],
    *,
    device: torch.device | str,
    expect: dict | None = None,
) -> DynamicsEnsemble:
    """Load trained members over one base checkpoint, sharing its frozen representation.

    Members must agree on the episode split, the dataset and representation they were built from
    and the base checkpoint: a member trained on another split would leak evaluation episodes into
    whatever the ensemble is used to select.
    """
    metadata = [load_metadata(path) for path in dynamics]
    check_provenance(metadata, PROVENANCE)
    for field, value in (expect or {}).items():
        if metadata[0].get(field) != value:
            raise ValueError(f"the ensemble was built with a different {field}")
    members = [load_dynamics(base_checkpoint, path, device=device) for path in dynamics]
    for member in members[1:]:
        member.encoder, member.projector = members[0].encoder, members[0].projector
    return DynamicsEnsemble(members)
