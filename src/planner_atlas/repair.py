"""Repairing the dynamics on newly acquired data, beside the data they were trained on.

Every repair branch starts from the same base checkpoint, whatever acquired its data, and differs
only in which transitions it saw. Each batch mixes base windows with acquired windows in a fixed
ratio: a few thousand new transitions would otherwise be statistically invisible next to two
million base windows. Both sides are sampled with replacement, so the ratio holds for however many
steps a run takes, and the ratio is part of the experiment's controls rather than a free parameter
per strategy.

Training itself is the same objective, optimizer and clipping as planner_atlas.training, with the
same held normalization statistics; only the sampler differs.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from planner_atlas.training import (
    NUM_STEPS,
    LatentWindows,
    TrainableDynamics,
    dynamics_optimizer,
    optimizer_step,
    validation_loss,
)

WindowSource = LatentWindows  # anything with len() and batch(indices, device=...)


@dataclass(frozen=True)
class RepairWindows:
    """Windows over acquired trajectories, in the shape base windows come in.

    A trajectory of T executed blocks yields T - 3 windows: the loss needs four observations five
    raw actions apart together with the action blocks applied at them, and only blocks that were
    actually executed may appear.
    """

    latents: np.ndarray  # [K, T + 1, 192]
    blocks: np.ndarray  # [K, T, 10]
    index: np.ndarray  # [W, 2]: trajectory, first block of the window

    def __len__(self) -> int:
        return len(self.index)

    def batch(
        self, indices: np.ndarray, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latents [B, 4, 192] and normalized action blocks [B, 4, 10] of the chosen windows."""
        trajectory = self.index[indices, 0][:, None]
        first = self.index[indices, 1][:, None] + np.arange(NUM_STEPS)
        return (
            torch.from_numpy(self.latents[trajectory, first]).to(device),
            torch.from_numpy(self.blocks[trajectory, first]).to(device),
        )


def repair_windows(latents: np.ndarray, blocks: np.ndarray) -> RepairWindows:
    """Every window of every acquired trajectory, in acquisition order."""
    trajectories, horizon = blocks.shape[:2]
    if horizon < NUM_STEPS:
        raise ValueError(f"a trajectory of {horizon} blocks is shorter than a {NUM_STEPS} window")
    index = np.array(
        [(k, first) for k in range(trajectories) for first in range(horizon - NUM_STEPS + 1)]
    )
    return RepairWindows(latents, blocks, index)


def mixture_sizes(batch_size: int, repair_fraction: float) -> tuple[int, int]:
    """How many base and acquired windows one batch holds."""
    if not 0.0 <= repair_fraction <= 1.0:
        raise ValueError(f"repair_fraction {repair_fraction} is not a fraction")
    repair = round(batch_size * repair_fraction)
    return batch_size - repair, repair


def mixed_batch(
    base: WindowSource,
    repair: WindowSource,
    *,
    batch_size: int,
    repair_fraction: float,
    generator: np.random.Generator,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One batch drawn from both sources in the configured ratio, each with replacement."""
    base_size, repair_size = mixture_sizes(batch_size, repair_fraction)
    parts = []
    if base_size:
        parts.append(base.batch(generator.integers(len(base), size=base_size), device=device))
    if repair_size:
        parts.append(repair.batch(generator.integers(len(repair), size=repair_size), device=device))
    return tuple(torch.cat(tensors) for tensors in zip(*parts, strict=True))


def repair_dynamics(
    model: TrainableDynamics,
    base: WindowSource,
    repair: WindowSource,
    *,
    steps: int,
    batch_size: int,
    repair_fraction: float,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    validation: WindowSource | None = None,
    log_every: int = 100,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """Fixed-step repair training on the mixture; one record per logged step."""
    device = next(model.parameters()).device
    optimizer = dynamics_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    generator = np.random.default_rng(seed)
    history, losses = [], []
    for step in range(steps):
        model.train()
        latents, blocks = mixed_batch(
            base,
            repair,
            batch_size=batch_size,
            repair_fraction=repair_fraction,
            generator=generator,
            device=device,
        )
        losses.append(optimizer_step(model, optimizer, latents, blocks))
        if (step + 1) % log_every == 0 or step + 1 == steps:
            record = {"step": step + 1, "train_loss": float(np.mean(losses[-log_every:]))}
            if validation is not None:
                record["validation_loss"] = validation_loss(
                    model, validation, batch_size=batch_size
                )
            history.append(record)
            log(str(record))
    return history
