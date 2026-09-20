"""Training the LeWM latent dynamics with the representation frozen.

The released checkpoint is the initialization. The encoder and its projector keep the released
weights, so every trained member predicts in the same latent coordinate system the Atlas costs are
measured in, and only the action embedder, the predictor and the predictor projection learn.

The objective is the upstream one (le-wm@8edfeb3 `train.py:lejepa_forward`, `config/train/lewm.yaml`
and `config/train/data/*.yaml`), restricted to that frozen representation. A window holds
num_steps = history_size + num_preds = 4 observations, 5 raw actions apart, inside one episode.
With z_t the projected encoder latent of observation t and a_t the normalized action block applied
at it,

    L = mean over t < 3 and over latent dimensions of
        ( pred_proj(predictor(z_0..z_t, a_0..a_t))_t - z_{t+1} )^2 ,

teacher forced and one step ahead: upstream predicts every position of the causal window in a
single pass and compares with the encoded latents one step later. Nothing is detached upstream, so
its gradients also reach the encoder through the targets; with the encoder frozen the targets are
constants. There is no target encoder, no EMA and no stop-gradient anywhere in that objective.

Upstream adds `0.09 * SIGReg(emb)`, a regularizer on encoder outputs that keeps the representation
from collapsing into a trivially predictable one. Here its input is constant, so its gradient with
respect to every trainable parameter is exactly zero: it is omitted rather than approximated.

One further deliberate departure from upstream: the batch normalization inside pred_proj keeps the
released running statistics for every forward pass, training included, while its affine parameters
go on training (see TrainableDynamics).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the compression filter of the official pixels)
import numpy as np
import torch
from torch import nn

from planner_atlas.data import BLOCK_STEPS, ActionStats, normalize_actions
from planner_atlas.evaluation import frames_to_tensor
from planner_atlas.models.reference_lewm import LATENT_DIM, ReferenceLeWM

HISTORY = 3  # observations the predictor conditions on, upstream history_size
NUM_STEPS = HISTORY + 1  # observations per window, upstream history_size + num_preds
WINDOW_ROWS = NUM_STEPS * BLOCK_STEPS  # rows one window spans inside an episode, upstream span
GRADIENT_CLIP = 1.0  # upstream trainer.gradient_clip_val
FROZEN = ("encoder", "projector")  # the representation, kept exactly as released
TRAINABLE = ("action_encoder", "predictor", "pred_proj")  # the latent dynamics
FRAME_CONVENTION = "live render of the replayed state; uint8/255, imagenet, vit cls, projector"
DYNAMICS_PROTOCOL = "frozen_rep_v2"  # frozen representation, held normalization statistics


@dataclass(frozen=True)
class LatentCache:
    """Frozen latents of every frame of one dataset, with its episode boundaries."""

    latents: np.ndarray  # [rows, 192] float32
    lengths: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class LatentWindows:
    """Training windows over cached latents, addressed by the row each one starts at."""

    latents: np.ndarray  # [rows, 192], frozen
    actions: np.ndarray  # [rows, 2], normalized, boundary NaNs zeroed as upstream does
    starts: np.ndarray  # [W] rows where a window begins

    def __len__(self) -> int:
        return len(self.starts)

    def batch(
        self, indices: np.ndarray, *, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Latents [B, 4, 192] and the 3 action blocks [B, 3, 10] the loss conditions on."""
        starts = self.starts[indices][:, None]
        latents = self.latents[starts + np.arange(NUM_STEPS) * BLOCK_STEPS]
        actions = self.actions[starts + np.arange(HISTORY * BLOCK_STEPS)]
        return (
            torch.from_numpy(latents).to(device),
            torch.from_numpy(actions).to(device).view(len(starts), HISTORY, -1),
        )


class TrainableDynamics(nn.Module):
    """A released LeWM whose representation is frozen and whose latent dynamics train.

    The frozen half gets no gradients and never leaves eval mode, so its batch normalization
    statistics stay exactly as released. Rollout and cost delegate to the wrapped model, which
    keeps one dynamics implementation for training, planning and the Atlas.

    The batch normalization of pred_proj also keeps the released running statistics, in training
    as in evaluation, which is a deliberate departure from upstream: upstream normalizes by batch
    statistics and lets the running estimates follow the training distribution. This experiment
    freezes the representation and compares repair distributions against each other, so a
    normalization layer that silently adapts its own state to whatever data a member is trained on
    would be a second, distribution-dependent repair mechanism mixed into the one being measured.
    Its affine weight and bias still train, and gradients still flow through the normalization and
    into the predictor.
    """

    def __init__(self, model: ReferenceLeWM) -> None:
        super().__init__()
        self.model = model
        for name in FROZEN:
            getattr(self.model, name).requires_grad_(False).eval()
        self._hold_normalization()

    @classmethod
    def from_checkpoint(cls, path: Path, *, device: torch.device | str) -> "TrainableDynamics":
        return cls(ReferenceLeWM.from_checkpoint(path, device=device))

    def train(self, mode: bool = True) -> "TrainableDynamics":
        super().train(mode)
        self._hold_normalization()
        return self

    def _hold_normalization(self) -> None:
        """Keep the frozen half and every batch normalization on their released statistics."""
        for name in FROZEN:
            getattr(self.model, name).eval()
        for module in self.model.pred_proj.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()

    def predict(self, latents: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        """Predicted next latents [B, T, 192] of a causal window of latents and action blocks."""
        return self.model.predict(latents, self.model.action_encoder(action_blocks))

    def rollout(self, latent: torch.Tensor, action_blocks: torch.Tensor) -> torch.Tensor:
        return self.model.rollout(latent, action_blocks)

    def cost(
        self, latent: torch.Tensor, goal: torch.Tensor, action_blocks: torch.Tensor
    ) -> torch.Tensor:
        return self.model.cost(latent, goal, action_blocks)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def dynamics_state_dict(self) -> dict[str, torch.Tensor]:
        """Only the trained weights: the frozen half stays in the base checkpoint."""
        state = self.model.state_dict()
        return {name: state[name] for name in sorted(dynamics_keys(self.model))}


def protocol_metadata() -> dict:
    """What every saved member states about how it was trained.

    Members trained under different protocols are not comparable, so this travels with each
    checkpoint and load_dynamics refuses anything else.
    """
    return {
        "dynamics_protocol": DYNAMICS_PROTOCOL,
        "normalization": {"running_stats": "frozen", "affine_parameters": "trainable"},
    }


def dynamics_keys(model: ReferenceLeWM) -> set[str]:
    """State dict entries of the latent dynamics, the half that trains."""
    prefixes = tuple(f"{module}." for module in TRAINABLE)
    return {name for name in model.state_dict() if name.startswith(prefixes)}


def predictor_loss(
    model: TrainableDynamics, latents: torch.Tensor, action_blocks: torch.Tensor
) -> torch.Tensor:
    """The frozen-representation objective of the module docstring, on one batch of windows.

    latents are the 4 observations of a window and action_blocks the 3 blocks applied at the
    first 3 of them, which is exactly what the objective conditions on.
    """
    predicted = model.predict(latents[:, :HISTORY], action_blocks)
    return (predicted - latents[:, 1:]).square().mean()


def dynamics_optimizer(
    model: TrainableDynamics, *, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    """The optimizer every run uses, so repair branches cannot drift apart on this."""
    return torch.optim.AdamW(
        model.trainable_parameters(), lr=learning_rate, weight_decay=weight_decay
    )


def optimizer_step(
    model: TrainableDynamics,
    optimizer: torch.optim.Optimizer,
    latents: torch.Tensor,
    action_blocks: torch.Tensor,
) -> float:
    """One step on the prediction loss, clipped as upstream clips; returns the loss."""
    loss = predictor_loss(model, latents, action_blocks)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.trainable_parameters(), GRADIENT_CLIP)
    optimizer.step()
    return loss.item()


def build_latent_cache(
    dataset: Path,
    checkpoint: Path,
    cache: Path,
    frames: Iterable[np.ndarray],
    *,
    device: torch.device | str,
    batch_size: int = 512,
    log: Callable[[str], None] = print,
) -> None:
    """Encode every frame of dataset once, storing latents beside the identity they came from.

    frames yields the dataset's rows in order, from the environment's own renderer (see
    pusht.replay_frames): dynamics then learn on the same latent manifold that planning,
    acquisition and evaluation produce. Encoding the dataset's recorded pixels instead puts them
    on a measurably different one - on 60 random PushT states the recorded frame sits 0.70 in
    latent L2 from the live render of that same state, where rendering it twice gives 0.

    The file is written aside and renamed at the end, so an interrupted build leaves no cache that
    would later look complete.
    """
    model = ReferenceLeWM.from_checkpoint(checkpoint, device=device)
    identity = cache_identity(dataset, checkpoint)
    partial = cache.with_name(cache.name + ".partial")
    with h5py.File(dataset, "r") as source, h5py.File(partial, "w") as target:
        rows = source["pixels"].shape[0]
        latents = target.create_dataset("latent", shape=(rows, LATENT_DIM), dtype="float32")
        target["ep_len"], target["ep_offset"] = source["ep_len"][:], source["ep_offset"][:]
        written, batch = 0, []
        for frame in frames:
            batch.append(frame)
            if len(batch) == batch_size:
                written = _encode_into(model, latents, batch, written, device=device, log=log)
                batch = []
        if batch:
            written = _encode_into(model, latents, batch, written, device=device, log=log)
        if written != rows:
            raise ValueError(f"the frame source yielded {written} frames, the dataset has {rows}")
        target.attrs.update(identity)
    partial.replace(cache)
    log(f"wrote {cache} ({rows:,} latents)")


def _encode_into(model, latents, batch, written, *, device, log) -> int:
    encoded = model.encode(frames_to_tensor(np.stack(batch)[None], device))[0]
    latents[written : written + len(batch)] = encoded.float().cpu().numpy()
    if (written // len(batch)) % 100 == 0:
        log(f"encoded {written:,}/{len(latents):,} frames")
    return written + len(batch)


def load_latent_cache(cache: Path, dataset: Path, checkpoint: Path) -> LatentCache:
    """Load a cache, refusing it unless it was built from this dataset, checkpoint and convention."""
    with h5py.File(cache, "r") as file:
        expected = cache_identity(dataset, checkpoint)
        stored = {key: file.attrs.get(key) for key in expected}
        if stored != expected:
            differing = [key for key in expected if stored[key] != expected[key]]
            raise ValueError(f"latent cache {cache} was built for a different {differing}")
        return LatentCache(file["latent"][:], file["ep_len"][:], file["ep_offset"][:])


def cache_identity(dataset: Path, checkpoint: Path) -> dict[str, str]:
    """What a latent cache must agree with: the dataset, the encoder weights, the frame convention."""
    with h5py.File(dataset, "r") as file:
        episodes = np.concatenate([file["ep_len"][:], file["ep_offset"][:]]).tobytes()
        rows = file["pixels"].shape[0]
    return {
        "dataset": dataset.name,
        "dataset_rows": str(rows),
        "dataset_episodes": sha256(episodes).hexdigest(),
        "checkpoint": file_digest(checkpoint),
        "convention": FRAME_CONVENTION,
    }


def file_digest(path: Path) -> str:
    """SHA-256 of a file, which is how a checkpoint identifies itself here."""
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def window_starts(lengths: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Rows where a whole window fits inside one episode, as upstream indexes its clips."""
    starts = [
        offset + np.arange(length - WINDOW_ROWS + 1)
        for length, offset in zip(lengths, offsets, strict=True)
        if length >= WINDOW_ROWS
    ]
    if not starts:
        raise ValueError(f"no episode is {WINDOW_ROWS} steps long")
    return np.concatenate(starts)


def split_episodes(num_episodes: int, *, validation_fraction: float, seed: int) -> np.ndarray:
    """Mask of validation episodes: a whole episode goes to one side, never to both."""
    validation = np.zeros(num_episodes, dtype=bool)
    chosen = np.random.default_rng(seed).permutation(num_episodes)
    validation[chosen[: round(validation_fraction * num_episodes)]] = True
    return validation


def bootstrap_indices(num_windows: int, *, member: int, seed: int) -> np.ndarray:
    """A standard bootstrap of the training windows for one member: N draws with replacement."""
    return np.random.default_rng([seed, member]).integers(num_windows, size=num_windows)


def latent_windows(
    cache: LatentCache,
    dataset: Path,
    stats: ActionStats,
    *,
    episodes: np.ndarray | None = None,
) -> LatentWindows:
    """Windows over the cached latents, restricted to the episodes marked in episodes."""
    with h5py.File(dataset, "r") as file:
        actions = normalize_actions(torch.from_numpy(file["action"][:]), stats)
    starts = window_starts(cache.lengths, cache.offsets)
    if episodes is not None:
        starts = starts[episodes[np.searchsorted(cache.offsets, starts, side="right") - 1]]
    # upstream replaces the NaN actions of episode boundaries with zero, after normalizing
    return LatentWindows(cache.latents, torch.nan_to_num(actions, 0.0).numpy(), starts)


def subsample_windows(windows: LatentWindows, limit: int, *, seed: int) -> LatentWindows:
    """A deterministic random subset of windows; a prefix would cover only the first episodes."""
    if not limit or limit >= len(windows):
        return windows
    chosen = np.random.default_rng(seed).choice(len(windows), size=limit, replace=False)
    return LatentWindows(windows.latents, windows.actions, windows.starts[np.sort(chosen)])


def windows_episodes(windows: LatentWindows, offsets: np.ndarray) -> int:
    """How many distinct episodes a window set covers."""
    return len(np.unique(np.searchsorted(offsets, windows.starts, side="right") - 1))


def train_dynamics(
    model: TrainableDynamics,
    windows: LatentWindows,
    *,
    indices: np.ndarray,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    validation: LatentWindows | None = None,
    log: Callable[[str], None] = print,
) -> list[dict[str, float]]:
    """AdamW on the trainable dynamics over the given windows; one record per epoch.

    seed drives both the batch order and torch's own generator: the predictor keeps the released
    dropout, so an unseeded run is not reproducible.
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = dynamics_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(indices))
        losses = []
        for start in range(0, len(order) - batch_size + 1, batch_size):  # drop_last, as upstream
            latents, blocks = windows.batch(
                indices[order[start : start + batch_size]], device=device
            )
            losses.append(optimizer_step(model, optimizer, latents, blocks))
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "steps": len(losses)}
        if validation is not None:
            record["validation_loss"] = validation_loss(model, validation, batch_size=batch_size)
        history.append(record)
        log(str(record))
    return history


@torch.no_grad()
def validation_loss(model: TrainableDynamics, windows: LatentWindows, *, batch_size: int) -> float:
    """Mean prediction loss over whole windows, with the dynamics in eval mode."""
    device = next(model.parameters()).device
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(windows), batch_size):
        indices = np.arange(start, min(start + batch_size, len(windows)))
        latents, blocks = windows.batch(indices, device=device)
        total += predictor_loss(model, latents, blocks).item() * len(indices)
        count += len(indices)
    return total / count


def save_dynamics(path: Path, model: TrainableDynamics, metadata: dict) -> None:
    """The trained dynamics and what reproduces them; the frozen half is the base checkpoint."""
    saved = {"dynamics": model.dynamics_state_dict(), "metadata": protocol_metadata() | metadata}
    torch.save(saved, path)


def load_dynamics(
    base_checkpoint: Path, dynamics: Path, *, device: torch.device | str
) -> ReferenceLeWM:
    """The released model with trained dynamics loaded over it: planners take it unchanged."""
    model = ReferenceLeWM.from_checkpoint(base_checkpoint, device=device)
    saved = torch.load(dynamics, map_location=device, weights_only=True)
    protocol = saved.get("metadata", {}).get("dynamics_protocol")
    if protocol != DYNAMICS_PROTOCOL:
        raise ValueError(
            f"{dynamics} was trained under protocol {protocol!r}, not {DYNAMICS_PROTOCOL!r}: "
            "dynamics from different protocols are not comparable"
        )
    trained = saved["dynamics"]
    expected = dynamics_keys(model)
    if set(trained) != expected:
        raise ValueError(
            f"{dynamics} does not hold this model's dynamics: "
            f"{len(expected - set(trained))} missing, {len(set(trained) - expected)} unexpected"
        )
    model.load_state_dict({**model.state_dict(), **trained}, strict=True)
    return model.eval()


def load_metadata(dynamics: Path) -> dict:
    """The metadata saved next to a member's dynamics."""
    return torch.load(dynamics, map_location="cpu", weights_only=True)["metadata"]
