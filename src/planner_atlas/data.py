"""Raw environment actions and the action blocks consumed by the released LeWM.

A recorded episode alternates observations and 2-d actions, o_t, a_t, o_{t+1}, a_{t+1}, ...,
where a_t is executed after observing o_t. One LeWM transition, o_t -> o_{t+5}, consumes the
block [a_t, ..., a_{t+4}], flattened in time order to [ax_t, ay_t, ..., ax_{t+4}, ay_{t+4}].
A block never spans two episodes.

Normalization acts on each 2-d action separately, with the mean and population standard
deviation of all recorded actions in the official dataset, exactly as the reference evaluation
fits them. It is a pure affine map: nothing is clipped to the environment's action bounds.
"""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

BLOCK_STEPS = 5  # environment actions per LeWM action block
ENV_ACTION_DIM = 2


@dataclass(frozen=True)
class ActionStats:
    """Per-coordinate mean and standard deviation [2] of raw environment actions."""

    mean: torch.Tensor
    std: torch.Tensor


def action_stats(path: Path) -> ActionStats:
    """Action statistics of an official LeWM HDF5 dataset (TwoRoom or PushT).

    Matches the StandardScaler of the reference evaluation: every action row without NaN
    (TwoRoom marks episode ends with NaN actions) and the population standard deviation.
    """
    with h5py.File(path, "r") as file:
        actions = file["action"][:].astype(np.float64)
    if actions.ndim != 2 or actions.shape[1] != ENV_ACTION_DIM:
        raise ValueError(f"expected actions [N, 2] in {path}, got {list(actions.shape)}")
    actions = actions[~np.isnan(actions).any(axis=1)]
    mean, std = actions.mean(axis=0), actions.std(axis=0)
    if (std == 0).any():
        raise ValueError(f"zero action standard deviation in {path}")
    return ActionStats(
        torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
    )


def pack_action_blocks(actions: torch.Tensor) -> torch.Tensor:
    """Flatten actions [..., 5, 2] into blocks [..., 10], in time order."""
    _check_trailing_shape(actions, (BLOCK_STEPS, ENV_ACTION_DIM), "actions")
    return actions.flatten(-2)


def unpack_action_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Split blocks [..., 10] into actions [..., 5, 2]; the inverse of pack_action_blocks."""
    _check_trailing_shape(blocks, (BLOCK_STEPS * ENV_ACTION_DIM,), "action blocks")
    return blocks.unflatten(-1, (BLOCK_STEPS, ENV_ACTION_DIM))


def normalize_actions(actions: torch.Tensor, stats: ActionStats) -> torch.Tensor:
    """Map raw actions [..., 2] to the model's normalized coordinates."""
    _check_trailing_shape(actions, (ENV_ACTION_DIM,), "actions")
    mean, std = stats.mean.to(actions.device), stats.std.to(actions.device)
    return (actions - mean) / std


def denormalize_actions(actions: torch.Tensor, stats: ActionStats) -> torch.Tensor:
    """Map normalized actions [..., 2] back to raw environment actions."""
    _check_trailing_shape(actions, (ENV_ACTION_DIM,), "actions")
    mean, std = stats.mean.to(actions.device), stats.std.to(actions.device)
    return actions * std + mean


def _check_trailing_shape(tensor: torch.Tensor, trailing: tuple[int, ...], name: str) -> None:
    if tuple(tensor.shape[-len(trailing) :]) != trailing:
        expected = ", ".join(str(size) for size in trailing)
        raise ValueError(f"expected {name} [..., {expected}], got {list(tensor.shape)}")
