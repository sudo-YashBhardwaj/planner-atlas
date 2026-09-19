"""TwoRoom evaluation: dataset cases, frames, executed action blocks, counterfactuals and MPC.

Frames enter the model as float32 [..., 3, 224, 224] in [0, 1], scaled exactly as the reference
evaluation scales them. A normalized action block [10] executes as 5 raw environment actions, and
executions record the frame after each block, aligned with the model's predicted latents z_1..z_T.
TwoRoom success is an agent-goal distance below 16 pixels, where the environment terminates; the
task cost is that distance / 224.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401  (registers the compression filter of the official pixels)
import numpy as np
import torch

from planner_atlas.data import ActionStats, denormalize_actions, unpack_action_blocks
from planner_atlas.envs import set_state_and_goal
from planner_atlas.models.reference_lewm import ReferenceLeWM, terminal_cost
from planner_atlas.planning import PlanResult

GOAL_OFFSET = 25  # raw environment steps from start to goal, as in the reference evaluation
SUCCESS_RADIUS = 16.0  # pixels, TwoRoom's own termination radius
ARENA_SIZE = 224.0  # pixels, normalizes the task cost


@dataclass(frozen=True)
class TwoRoomCase:
    """A start and the goal GOAL_OFFSET steps later in one episode of the official dataset."""

    episode: int
    start_step: int
    goal_step: int
    start_state: np.ndarray  # agent xy
    goal_state: np.ndarray
    start_frame: np.ndarray  # uint8 [224, 224, 3]
    goal_frame: np.ndarray

    @property
    def initial_distance(self) -> float:
        return float(np.linalg.norm(self.start_state - self.goal_state))


@dataclass(frozen=True)
class CandidateEvaluation:
    """Model and simulator evaluation of candidate plans; every field is an array [S]."""

    predicted_cost: np.ndarray  # sum_d (predicted z_T - z_goal)^2
    realized_cost: np.ndarray  # sum_d (z_T - z_goal)^2, z_T encoding the real final frame
    rollout_error: np.ndarray  # mean over t = 1..T and d of (predicted z_t - z_t)^2
    terminal_error: np.ndarray  # mean over d of (predicted z_T - z_T)^2
    distance: np.ndarray  # final agent-goal distance in pixels
    reached: np.ndarray  # whether the goal was reached at any raw step of the plan

    @property
    def optimism(self) -> np.ndarray:
        """Positive when the model predicts a finish closer to the goal latent than realized."""
        return self.realized_cost - self.predicted_cost


@dataclass(frozen=True)
class MPCResult:
    success: bool
    raw_steps: int
    final_distance: float


def sample_tworoom_cases(
    path: Path, *, num_cases: int, seed: int, goal_offset: int = GOAL_OFFSET
) -> list[TwoRoomCase]:
    """Distinct start rows drawn uniformly among those whose goal lies in the same episode and is
    not already reached at the start (agent-goal distance at least SUCCESS_RADIUS)."""
    with h5py.File(path, "r") as file:
        lengths, offsets, proprio = file["ep_len"][:], file["ep_offset"][:], file["proprio"][:]
        starts = np.concatenate(
            [
                offset + np.arange(length - goal_offset)
                for length, offset in zip(lengths, offsets, strict=True)
            ]
        )
        distances = np.linalg.norm(proprio[starts + goal_offset] - proprio[starts], axis=1)
        starts = starts[distances >= SUCCESS_RADIUS]
        rows = np.sort(np.random.default_rng(seed).choice(starts, size=num_cases, replace=False))
        episodes = np.searchsorted(offsets, rows, side="right") - 1
        pixels = file["pixels"]
        return [
            TwoRoomCase(
                episode=int(episode),
                start_step=int(row - offsets[episode]),
                goal_step=int(row - offsets[episode] + goal_offset),
                start_state=proprio[row],
                goal_state=proprio[row + goal_offset],
                start_frame=pixels[row],
                goal_frame=pixels[row + goal_offset],
            )
            for episode, row in zip(episodes, rows, strict=True)
        ]


def frames_to_tensor(frames: np.ndarray, device: torch.device | str) -> torch.Tensor:
    """uint8 frames [..., 224, 224, 3] -> float32 [..., 3, 224, 224] in [0, 1]."""
    if frames.dtype != np.uint8:
        raise ValueError(f"expected uint8 frames, got {frames.dtype}")
    return torch.from_numpy(frames).to(device).movedim(-1, -3).float().mul_(1.0 / 255.0)


def encode_frame(
    model: ReferenceLeWM, frame: np.ndarray, device: torch.device | str
) -> torch.Tensor:
    """Latent [1, 192] of one uint8 frame [224, 224, 3]."""
    return model.encode(frames_to_tensor(frame[None, None], device))[:, 0]


def execute_blocks(
    env: gym.Env, blocks: torch.Tensor, stats: ActionStats
) -> tuple[np.ndarray, list[dict], bool]:
    """Execute all normalized action blocks [T, 10], 5 raw actions each.

    Returns the frame and step info after each block, and whether the environment terminated
    (reached its goal) at any raw step; execution continues past it to keep block-end alignment.
    """
    frames, infos, reached = [], [], False
    for block in _raw_actions(blocks, stats):
        for action in block:
            _, _, terminated, _, info = env.step(action)
            reached = reached or terminated
        frames.append(env.render())
        infos.append(info)
    return np.stack(frames), infos, reached


def rollout_in_env(
    env: gym.Env, case: TwoRoomCase, blocks: torch.Tensor, stats: ActionStats
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Open-loop execution of blocks [T, 10] from the case start: block-end frames, final agent
    xy (TwoRoom's state), and whether the goal was reached at any raw step."""
    env.reset(seed=0)
    set_state_and_goal(env, case.start_state, case.goal_state)
    frames, infos, reached = execute_blocks(env, blocks, stats)
    return frames, infos[-1]["state"], reached


def evaluate_candidates(
    model: ReferenceLeWM,
    env: gym.Env,
    case: TwoRoomCase,
    stats: ActionStats,
    latent: torch.Tensor,
    goal: torch.Tensor,
    plans: torch.Tensor,
) -> CandidateEvaluation:
    """Predict plans [S, H, 10] from latent [1, 192] and execute them from the case start."""
    predicted = model.rollout(latent, plans[None])[0]  # [S, H + 1, D], z_0 first
    frames, positions, reached = zip(
        *(rollout_in_env(env, case, plan, stats) for plan in plans), strict=True
    )
    real = model.encode(frames_to_tensor(np.stack(frames), latent.device))  # [S, H, D]
    return CandidateEvaluation(
        predicted_cost=terminal_cost(predicted[None, :, -1], goal)[0].cpu().numpy(),
        realized_cost=terminal_cost(real[None, :, -1], goal)[0].cpu().numpy(),
        rollout_error=(predicted[:, 1:] - real).square().mean(dim=(1, 2)).cpu().numpy(),
        terminal_error=(predicted[:, -1] - real[:, -1]).square().mean(dim=1).cpu().numpy(),
        distance=np.linalg.norm(np.stack(positions) - case.goal_state, axis=1),
        reached=np.array(reached),
    )


def run_mpc(
    env: gym.Env,
    model: ReferenceLeWM,
    case: TwoRoomCase,
    stats: ActionStats,
    goal: torch.Tensor,
    plan: Callable[[torch.Tensor, torch.Tensor], PlanResult],
    *,
    max_raw_steps: int = 50,
) -> MPCResult:
    """Block-level MPC: encode the current frame, plan, execute only the first block, repeat.

    Replanning happens only between blocks, but the episode ends at the first raw step where the
    environment terminates (reaches the goal), even mid-block, or after max_raw_steps actions.
    """
    env.reset(seed=0)
    set_state_and_goal(env, case.start_state, case.goal_state)
    position, raw_steps, terminated = case.start_state, 0, False
    while not terminated and raw_steps < max_raw_steps:
        latent = encode_frame(model, env.render(), goal.device)
        for action in _raw_actions(plan(latent, goal).actions[:1], stats)[0]:
            _, _, terminated, _, info = env.step(action)
            position, raw_steps = info["state"], raw_steps + 1
            if terminated:
                break
    distance = float(np.linalg.norm(position - case.goal_state))
    return MPCResult(bool(terminated), raw_steps, distance)


def _raw_actions(blocks: torch.Tensor, stats: ActionStats) -> np.ndarray:
    """Raw actions [T, 5, 2] of normalized blocks [T, 10]; they must lie in [-1, 1], unclipped."""
    actions = denormalize_actions(unpack_action_blocks(blocks), stats).cpu().numpy()
    if np.abs(actions).max() > 1 + 1e-5:
        raise ValueError(f"raw action outside [-1, 1]: max |a| = {np.abs(actions).max()}")
    return actions
