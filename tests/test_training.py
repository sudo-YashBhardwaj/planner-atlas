from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from planner_atlas.data import ENV_ACTION_DIM
from planner_atlas.models.reference_lewm import LATENT_DIM, ReferenceLeWM
from planner_atlas.training import (
    FROZEN,
    NUM_STEPS,
    WINDOW_ROWS,
    LatentWindows,
    TrainableDynamics,
    bootstrap_indices,
    cache_identity,
    load_dynamics,
    load_latent_cache,
    predictor_loss,
    protocol_metadata,
    save_dynamics,
    split_episodes,
    train_dynamics,
    window_starts,
)


@pytest.fixture(scope="module")
def base_checkpoint(tmp_path_factory) -> Path:
    """A stand-in released checkpoint: random weights in the real architecture."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("lewm") / "weights.pt"
    torch.save(ReferenceLeWM().state_dict(), path)
    return path


def synthetic_windows(rows: int = 96, seed: int = 0) -> LatentWindows:
    generator = np.random.default_rng(seed)
    return LatentWindows(
        latents=generator.normal(size=(rows, LATENT_DIM)).astype(np.float32),
        actions=generator.normal(size=(rows, ENV_ACTION_DIM)).astype(np.float32),
        starts=np.arange(rows - WINDOW_ROWS + 1),
    )


def test_initialization_matches_the_reference_model(base_checkpoint) -> None:
    reference = ReferenceLeWM.from_checkpoint(base_checkpoint, device="cpu")
    trainable = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    torch.manual_seed(1)
    latent, goal = torch.randn(2, LATENT_DIM), torch.randn(2, LATENT_DIM)
    blocks = torch.randn(2, 3, 4, 10)

    torch.testing.assert_close(
        trainable.rollout(latent, blocks), reference.rollout(latent, blocks), rtol=0, atol=0
    )
    torch.testing.assert_close(
        trainable.cost(latent, goal, blocks), reference.cost(latent, goal, blocks), rtol=0, atol=0
    )


def test_a_training_step_moves_the_dynamics_and_leaves_the_representation(base_checkpoint) -> None:
    model = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    latents, blocks = synthetic_windows().batch(np.arange(8), device="cpu")

    model.train()
    assert all(not getattr(model.model, name).training for name in FROZEN)  # never leaves eval
    loss = predictor_loss(model, latents, blocks)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.model.projector.parameters())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.trainable_parameters()
    )

    torch.optim.AdamW(model.trainable_parameters(), lr=1e-3).step()
    after = model.state_dict()
    frozen = [name for name in before if name.startswith(("model.encoder.", "model.projector."))]
    assert frozen and all(torch.equal(before[name], after[name]) for name in frozen)
    trained = [name for name in before if name.startswith("model.predictor.")]
    assert any(not torch.equal(before[name], after[name]) for name in trained)


def test_normalization_statistics_hold_while_their_affine_parameters_train(base_checkpoint) -> None:
    model = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    norm = next(m for m in model.model.pred_proj.modules() if isinstance(m, torch.nn.BatchNorm1d))
    released = (
        norm.running_mean.clone(),
        norm.running_var.clone(),
        norm.num_batches_tracked.clone(),
    )
    weight, bias = norm.weight.clone(), norm.bias.clone()
    predictor = {name: p.clone() for name, p in model.model.predictor.named_parameters()}

    model.train()
    assert not norm.training  # training mode must not put it back on batch statistics
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    windows = synthetic_windows()
    for step in range(3):
        latents, blocks = windows.batch(np.arange(step * 8, step * 8 + 8), device="cpu")
        loss = predictor_loss(model, latents, blocks)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert torch.equal(norm.running_mean, released[0])  # byte for byte the released statistics
    assert torch.equal(norm.running_var, released[1])
    assert torch.equal(norm.num_batches_tracked, released[2])
    assert norm.weight.grad is not None and torch.isfinite(norm.weight.grad).all()
    assert not torch.equal(norm.weight, weight) and not torch.equal(norm.bias, bias)
    changed = [
        n for n, p in model.model.predictor.named_parameters() if not torch.equal(p, predictor[n])
    ]
    assert changed  # and the dynamics they feed keep training


def test_training_overfits_a_tiny_set_of_windows(base_checkpoint) -> None:
    model = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    windows = synthetic_windows(rows=WINDOW_ROWS + 8)
    torch.manual_seed(0)
    history = train_dynamics(
        model,
        windows,
        indices=np.arange(len(windows)),
        batch_size=4,
        epochs=30,
        learning_rate=1e-3,
        weight_decay=0.0,
        seed=0,
        log=lambda record: None,
    )
    assert history[-1]["train_loss"] < 0.5 * history[0]["train_loss"]


def test_bootstrap_is_deterministic_and_specific_to_the_member() -> None:
    first = bootstrap_indices(500, member=0, seed=7)
    assert np.array_equal(first, bootstrap_indices(500, member=0, seed=7))
    assert len(first) == 500 and first.max() < 500
    assert len(np.unique(first)) < 500  # drawn with replacement
    assert not np.array_equal(first, bootstrap_indices(500, member=1, seed=7))
    assert not np.array_equal(first, bootstrap_indices(500, member=0, seed=8))


def test_saved_dynamics_reload_into_the_same_rollout(base_checkpoint, tmp_path) -> None:
    model = TrainableDynamics.from_checkpoint(base_checkpoint, device="cpu")
    latents, blocks = synthetic_windows().batch(np.arange(8), device="cpu")
    loss = predictor_loss(model.train(), latents, blocks)
    loss.backward()
    torch.optim.AdamW(model.trainable_parameters(), lr=1e-2).step()
    model.eval()

    path = tmp_path / "member.pt"
    save_dynamics(path, model, {"member": 3, "seed": 1})
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert saved["metadata"] == protocol_metadata() | {"member": 3, "seed": 1}
    assert saved["metadata"]["normalization"] == {
        "running_stats": "frozen",
        "affine_parameters": "trainable",
    }
    assert not any(name.startswith(("encoder.", "projector.")) for name in saved["dynamics"])

    reloaded = load_dynamics(base_checkpoint, path, device="cpu")
    torch.manual_seed(2)
    latent, blocks = torch.randn(2, LATENT_DIM), torch.randn(2, 3, 4, 10)
    torch.testing.assert_close(reloaded.rollout(latent, blocks), model.rollout(latent, blocks))
    # and it is no longer the released model it was loaded over
    released = ReferenceLeWM.from_checkpoint(base_checkpoint, device="cpu")
    assert not torch.equal(reloaded.rollout(latent, blocks), released.rollout(latent, blocks))

    torch.save({"dynamics": model.dynamics_state_dict(), "metadata": {"member": 3}}, path)
    with pytest.raises(ValueError, match="protocol"):  # no protocol tag: an older run
        load_dynamics(base_checkpoint, path, device="cpu")

    save_dynamics(path, model, {})
    torch.save(
        {"dynamics": {"predictor.nonsense": torch.zeros(1)}, "metadata": protocol_metadata()}, path
    )
    with pytest.raises(ValueError, match="dynamics"):
        load_dynamics(base_checkpoint, path, device="cpu")


def test_windows_stay_inside_episodes_and_split_by_episode() -> None:
    lengths = np.array([WINDOW_ROWS - 1, WINDOW_ROWS, 3 * WINDOW_ROWS])
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    starts = window_starts(lengths, offsets)

    assert len(starts) == 1 + (3 * WINDOW_ROWS - WINDOW_ROWS + 1)  # the short episode gives none
    episodes = np.searchsorted(offsets, starts, side="right") - 1
    assert set(episodes) == {1, 2}
    assert np.all(starts + WINDOW_ROWS <= (offsets + lengths)[episodes])

    validation = split_episodes(200, validation_fraction=0.1, seed=3)
    assert validation.sum() == 20
    assert np.array_equal(validation, split_episodes(200, validation_fraction=0.1, seed=3))
    assert not np.array_equal(validation, split_episodes(200, validation_fraction=0.1, seed=4))


def test_batches_keep_the_upstream_window_layout() -> None:
    rows = WINDOW_ROWS + 3
    latents = np.arange(rows * LATENT_DIM, dtype=np.float32).reshape(rows, LATENT_DIM)
    actions = np.arange(rows * ENV_ACTION_DIM, dtype=np.float32).reshape(rows, ENV_ACTION_DIM)
    windows = LatentWindows(latents, actions, np.array([0, 2]))
    batch_latents, batch_blocks = windows.batch(np.array([1]), device="cpu")

    assert batch_latents.shape == (1, NUM_STEPS, LATENT_DIM)
    assert batch_blocks.shape == (1, NUM_STEPS, 5 * ENV_ACTION_DIM)
    # observations are 5 raw steps apart, actions are the 5 raw actions in between, row major
    np.testing.assert_array_equal(batch_latents[0].numpy(), latents[[2, 7, 12, 17]])
    np.testing.assert_array_equal(batch_blocks[0, 0].numpy(), actions[2:7].reshape(-1))
    np.testing.assert_array_equal(batch_blocks[0, 3].numpy(), actions[17:22].reshape(-1))


def test_latent_cache_refuses_a_cache_built_elsewhere(tmp_path) -> None:
    dataset, other, cache = tmp_path / "d.h5", tmp_path / "other.pt", tmp_path / "cache.h5"
    with h5py.File(dataset, "w") as file:
        file["pixels"] = np.zeros((6, 2, 2, 3), dtype=np.uint8)
        file["ep_len"], file["ep_offset"] = [6], [0]
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    other.write_bytes(b"other weights")

    with h5py.File(cache, "w") as file:
        file["latent"] = np.zeros((6, LATENT_DIM), dtype=np.float32)
        file["ep_len"], file["ep_offset"] = [6], [0]
        file.attrs.update(cache_identity(dataset, checkpoint))

    assert len(load_latent_cache(cache, dataset, checkpoint).latents) == 6
    with pytest.raises(ValueError, match="checkpoint"):
        load_latent_cache(cache, dataset, other)
