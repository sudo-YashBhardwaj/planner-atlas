"""Where latents sit relative to the base training distribution: the project's one reference.

The reference is 100,000 training-split latents of the live-render latent cache, drawn with rng
seed 0 from the rows of the episodes the 10% validation split (seed 0) leaves out. Distances are
measured to it in the frozen latent coordinate system every model here shares.
"""

from dataclasses import dataclass

import numpy as np
import torch

from planner_atlas.training import LatentCache, split_episodes

REFERENCE_SIZE = 100_000
REFERENCE_SEED = 0
VALIDATION_FRACTION = 0.1
SPLIT_SEED = 0


@dataclass(frozen=True)
class BaseReference:
    """Base training latents [R, D], with the mean and precision their Mahalanobis distance uses."""

    latents: torch.Tensor
    centre: torch.Tensor
    precision: torch.Tensor


def base_reference(
    cache: LatentCache,
    *,
    device: torch.device | str,
    size: int = REFERENCE_SIZE,
    seed: int = REFERENCE_SEED,
) -> BaseReference:
    """The fixed random subset of training-split latents every coverage measurement compares to."""
    validation = split_episodes(
        len(cache.lengths), validation_fraction=VALIDATION_FRACTION, seed=SPLIT_SEED
    )
    episode_of_row = np.repeat(np.arange(len(cache.lengths)), cache.lengths)
    train_rows = np.flatnonzero(~validation[episode_of_row])
    chosen = np.sort(np.random.default_rng(seed).choice(train_rows, size, replace=False))
    reference = torch.from_numpy(cache.latents[chosen]).to(device)
    precision = torch.linalg.inv(
        torch.cov(reference.T) + 1e-6 * torch.eye(reference.shape[1], device=device)
    )
    return BaseReference(reference, reference.mean(0), precision)


def nearest_base_distance(latents: torch.Tensor, reference: BaseReference) -> torch.Tensor:
    """Euclidean distance [N] from each latent [N, D] to its nearest reference latent."""
    return torch.cat(
        [torch.cdist(part, reference.latents).min(1).values for part in latents.split(2048)]
    )


def mahalanobis_to_base(latents: torch.Tensor, reference: BaseReference) -> torch.Tensor:
    """Mahalanobis distance [N] of each latent [N, D] under the reference's mean and covariance."""
    offset = latents - reference.centre
    return torch.sqrt(((offset @ reference.precision) * offset).sum(1))


def participation_ratio(latents: torch.Tensor) -> float:
    """Effective dimensionality (sum of covariance eigenvalues)² / sum of their squares."""
    eigen = torch.linalg.eigvalsh(torch.cov(latents.T))
    return float(eigen.sum() ** 2 / (eigen**2).sum())
