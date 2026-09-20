from pathlib import Path

import numpy as np
import pytest
import torch

from planner_atlas.data import ENV_ACTION_DIM
from planner_atlas.models.reference_lewm import ACTION_DIM, LATENT_DIM, ReferenceLeWM
from planner_atlas.repair import (
    RepairWindows,
    mixed_batch,
    mixture_sizes,
    repair_dynamics,
    repair_windows,
)
from planner_atlas.training import (
    NUM_STEPS,
    WINDOW_ROWS,
    LatentWindows,
    TrainableDynamics,
)

HORIZON = 5


@pytest.fixture(scope="module")
def base_checkpoint(tmp_path_factory) -> Path:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("lewm") / "weights.pt"
    torch.save(ReferenceLeWM().state_dict(), path)
    return path


def acquired(trajectories: int = 4, value: float = 1.0) -> RepairWindows:
    """Acquired trajectories whose latents all carry one value, so batches are countable."""
    latents = np.full((trajectories, HORIZON + 1, LATENT_DIM), value, dtype=np.float32)
    blocks = np.full((trajectories, HORIZON, ACTION_DIM), value, dtype=np.float32)
    return repair_windows(latents, blocks)


def base(rows: int = 96, value: float = 0.0) -> LatentWindows:
    return LatentWindows(
        latents=np.full((rows, LATENT_DIM), value, dtype=np.float32),
        actions=np.full((rows, ENV_ACTION_DIM), value, dtype=np.float32),
        starts=np.arange(rows - WINDOW_ROWS + 1),
    )


def test_acquired_trajectories_yield_only_executed_windows() -> None:
    windows = acquired(trajectories=3)
    assert len(windows) == 3 * (HORIZON - NUM_STEPS + 1)  # 2 windows per 5 block trajectory
    latents, blocks = windows.batch(np.arange(len(windows)), device="cpu")
    assert latents.shape == (len(windows), NUM_STEPS, LATENT_DIM)
    assert blocks.shape == (len(windows), NUM_STEPS, ACTION_DIM)

    # every window is four consecutive observations of one trajectory with the blocks between them
    steps = np.arange(NUM_STEPS)
    for row, (trajectory, first) in enumerate(windows.index):
        np.testing.assert_array_equal(
            latents[row].numpy(), windows.latents[trajectory, first + steps]
        )
        np.testing.assert_array_equal(
            blocks[row].numpy(), windows.blocks[trajectory, first + steps]
        )
    with pytest.raises(ValueError):
        repair_windows(np.zeros((1, 3, LATENT_DIM)), np.zeros((1, 2, ACTION_DIM)))


def test_mixture_holds_the_configured_ratio() -> None:
    assert mixture_sizes(128, 0.5) == (64, 64)
    assert mixture_sizes(128, 0.25) == (96, 32)
    assert mixture_sizes(128, 0.0) == (128, 0) and mixture_sizes(128, 1.0) == (0, 128)
    with pytest.raises(ValueError):
        mixture_sizes(128, 1.5)

    generator = np.random.default_rng(0)
    latents, blocks = mixed_batch(
        base(value=0.0),
        acquired(value=1.0),
        batch_size=32,
        repair_fraction=0.25,
        generator=generator,
        device="cpu",
    )
    assert latents.shape == (32, NUM_STEPS, LATENT_DIM)
    from_repair = latents.reshape(32, -1).mean(dim=1) == 1.0
    assert int(from_repair.sum()) == 8 and int((~from_repair).sum()) == 24
    assert blocks.shape == (32, NUM_STEPS, ACTION_DIM)


def test_repair_branches_start_from_the_same_base(base_checkpoint) -> None:
    released = ReferenceLeWM.from_checkpoint(base_checkpoint, device="cpu")
    first = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    repair_dynamics(
        first,
        base(),
        acquired(value=0.7),
        steps=5,
        batch_size=8,
        repair_fraction=0.5,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=0,
        log=lambda record: None,
    )
    trained = [
        name
        for name, tensor in first.model.state_dict().items()
        if not torch.equal(tensor, released.state_dict()[name])
    ]
    assert trained  # the first branch really moved

    # a second branch, created afterwards from the same path, is untouched by the first
    second = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    assert all(
        torch.equal(tensor, released.state_dict()[name])
        for name, tensor in second.model.state_dict().items()
    )


def test_repair_holds_the_normalization_statistics(base_checkpoint) -> None:
    model = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    norm = next(m for m in model.model.pred_proj.modules() if isinstance(m, torch.nn.BatchNorm1d))
    released = (
        norm.running_mean.clone(),
        norm.running_var.clone(),
        norm.num_batches_tracked.clone(),
    )
    weight = norm.weight.clone()

    history = repair_dynamics(
        model,
        base(),
        acquired(value=0.7),
        steps=6,
        batch_size=8,
        repair_fraction=0.5,
        learning_rate=1e-2,
        weight_decay=0.0,
        seed=0,
        log_every=3,
        log=lambda record: None,
    )
    assert [record["step"] for record in history] == [3, 6]
    assert all(np.isfinite(record["train_loss"]) for record in history)
    assert torch.equal(norm.running_mean, released[0])
    assert torch.equal(norm.running_var, released[1])
    assert torch.equal(norm.num_batches_tracked, released[2])
    assert not norm.training and not torch.equal(norm.weight, weight)
