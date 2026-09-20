"""Evaluation of plans against the simulator: frames, executed action blocks, counterfactuals, MPC.

Frames enter the model as float32 [..., 3, 224, 224] in [0, 1], scaled exactly as the reference
evaluation scales them. A normalized action block [10] executes as 5 raw environment actions, and
executions record the frame after each block, aligned with the model's predicted latents z_1..z_T.

Every execution also reports a task outcome: scalars describing what really happened, whose
"task_cost" is the environment's continuous task error, the quantity the Atlas ranks against. The
machinery here is environment independent; TwoRoom's cases and task live below it, PushT's in
planner_atlas.pusht. TwoRoom's task cost is the final agent-goal distance / 224, and its success
is that distance below 16 pixels, where the environment terminates.
"""

from collections.abc import Callable, Iterator, Sequence
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

TaskOutcome = dict[str, float | bool]  # what one execution achieved, including "task_cost"


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
    """Model and simulator evaluation of candidate plans; every array has shape [S]."""

    predicted_cost: np.ndarray  # sum_d (predicted z_T - z_goal)^2
    realized_cost: np.ndarray  # sum_d (z_T - z_goal)^2, z_T encoding the real final frame
    rollout_error: np.ndarray  # mean over t = 1..T and d of (predicted z_t - z_t)^2
    terminal_error: np.ndarray  # mean over d of (predicted z_T - z_T)^2
    task: dict[str, np.ndarray]  # the task outcome of each execution, stacked over candidates

    @property
    def optimism(self) -> np.ndarray:
        """Positive when the model predicts a finish closer to the goal latent than realized."""
        return self.realized_cost - self.predicted_cost

    @property
    def task_cost(self) -> np.ndarray:
        """The environment's continuous task error, which the Atlas ranks against."""
        return self.task["task_cost"]


@dataclass(frozen=True)
class MPCResult:
    raw_steps: int  # control actions only; restoring the case start is not one of them
    task: TaskOutcome


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

    Returns the frame after each block, the step info after every raw action, and whether the
    environment terminated (reached its goal) at any raw step; execution continues past it so that
    the block-end frames stay aligned with the model's predicted latents.
    """
    frames, infos, reached = [], [], False
    for block in _raw_actions(blocks, stats):
        for action in block:
            _, _, terminated, _, info = env.step(action)
            infos.append(info)
            reached = reached or terminated
        frames.append(env.render())
    return np.stack(frames), infos, reached


def tworoom_outcome(position: np.ndarray, goal_state: np.ndarray, *, reached: bool) -> TaskOutcome:
    """TwoRoom's task outcome: how far the agent ended from the goal."""
    distance = float(np.linalg.norm(position - goal_state))
    return {
        "task_cost": distance / ARENA_SIZE,
        "task_distance": distance,
        "success": distance < SUCCESS_RADIUS,
        "reached_goal": bool(reached),
    }


def rollout_tworoom(
    env: gym.Env, case: TwoRoomCase, stats: ActionStats, blocks: torch.Tensor
) -> tuple[np.ndarray, TaskOutcome]:
    """Open-loop execution of blocks [T, 10] from a TwoRoom case start, whose state is exactly
    restorable: block-end frames and the task outcome."""
    env.reset(seed=0)
    set_state_and_goal(env, case.start_state, case.goal_state)
    frames, infos, reached = execute_blocks(env, blocks, stats)
    return frames, tworoom_outcome(infos[-1]["state"], case.goal_state, reached=reached)


def tworoom_frames(env: gym.Env, dataset: Path) -> Iterator[np.ndarray]:
    """Every frame of a TwoRoom dataset from the live renderer; its states restore exactly."""
    with h5py.File(dataset, "r") as file:
        proprio = file["proprio"][:]
    env.reset(seed=0)
    for state in proprio:
        set_state_and_goal(env, state, state)
        yield env.render()


def evaluate_candidates(
    model: ReferenceLeWM,
    latent: torch.Tensor,
    goal: torch.Tensor,
    execute: Callable[[torch.Tensor], tuple[np.ndarray, TaskOutcome]],
    plans: torch.Tensor,
) -> CandidateEvaluation:
    """Predict plans [S, H, 10] from latent [1, 192] and compare them with real executions.

    execute(blocks) runs one plan from the start of the case being studied and returns its
    block-end frames and task outcome.
    """
    frames, outcomes = zip(*(execute(plan) for plan in plans), strict=True)
    realized = model.encode(frames_to_tensor(np.stack(frames), latent.device))  # [S, H, D]
    return compare_with_model(model, latent, goal, plans, realized, outcomes)


def compare_with_model(
    model: ReferenceLeWM,
    latent: torch.Tensor,
    goal: torch.Tensor,
    plans: torch.Tensor,
    realized: torch.Tensor,
    outcomes: Sequence[TaskOutcome],
) -> CandidateEvaluation:
    """What the model predicted for plans [S, H, 10] against the latents [S, H, 192] their real
    executions produced, so an execution paid for once can be scored by several models."""
    predicted = model.rollout(latent, plans[None])[0]  # [S, H + 1, D], z_0 first
    return CandidateEvaluation(
        predicted_cost=terminal_cost(predicted[None, :, -1], goal)[0].cpu().numpy(),
        realized_cost=terminal_cost(realized[None, :, -1], goal)[0].cpu().numpy(),
        rollout_error=(predicted[:, 1:] - realized).square().mean(dim=(1, 2)).cpu().numpy(),
        terminal_error=(predicted[:, -1] - realized[:, -1]).square().mean(dim=1).cpu().numpy(),
        task={key: np.array([outcome[key] for outcome in outcomes]) for key in outcomes[0]},
    )


def run_mpc(
    env: gym.Env,
    model: ReferenceLeWM,
    stats: ActionStats,
    goal: torch.Tensor,
    plan: Callable[[torch.Tensor, torch.Tensor], PlanResult],
    *,
    max_raw_steps: int = 50,
    stop: Callable[[dict], bool] | None = None,
) -> tuple[list[dict], bool]:
    """Block-level MPC from wherever the environment already is: encode the current frame, plan,
    execute only the first block, repeat.

    Returns the step info after every raw action and whether the environment terminated (reached
    its goal). Replanning happens only between blocks, but the episode ends at the first raw step
    that terminates or that satisfies stop, if one is given, even mid-block, or after
    max_raw_steps actions. A stop predicate lets an environment end on its own task criterion
    rather than only on the environment's termination.
    """
    infos, terminated, done = [], False, False
    while not done and len(infos) < max_raw_steps:
        latent = encode_frame(model, env.render(), goal.device)
        for action in _raw_actions(plan(latent, goal).actions[:1], stats)[0]:
            _, _, terminated, _, info = env.step(action)
            infos.append(info)
            done = bool(terminated) or (stop is not None and stop(info))
            if done:
                break
    return infos, bool(terminated)


def run_tworoom_mpc(
    env: gym.Env,
    model: ReferenceLeWM,
    case: TwoRoomCase,
    stats: ActionStats,
    goal: torch.Tensor,
    plan: Callable[[torch.Tensor, torch.Tensor], PlanResult],
    *,
    max_raw_steps: int = 50,
) -> MPCResult:
    """Block-level MPC from a TwoRoom case start."""
    env.reset(seed=0)
    set_state_and_goal(env, case.start_state, case.goal_state)
    infos, reached = run_mpc(env, model, stats, goal, plan, max_raw_steps=max_raw_steps)
    outcome = tworoom_outcome(infos[-1]["state"], case.goal_state, reached=reached)
    return MPCResult(len(infos), outcome)


def _raw_actions(blocks: torch.Tensor, stats: ActionStats) -> np.ndarray:
    """Raw actions [T, 5, 2] of normalized blocks [T, 10]; they must lie in [-1, 1], unclipped."""
    actions = denormalize_actions(unpack_action_blocks(blocks), stats).cpu().numpy()
    if np.abs(actions).max() > 1 + 1e-5:
        raise ValueError(f"raw action outside [-1, 1]: max |a| = {np.abs(actions).max()}")
    return actions
