"""PushT cases, reconstructed starts and the semantic block task.

A mid-episode PushT state cannot be set: the recorded 7-d state omits the block's velocity and
Pymunk's contact cache, and _set_state runs an extra physics substep on top. A case start is
reconstructed instead - restore the recorded episode start, which is at rest, then replay the
recorded raw actions up to the start step - and every counterfactual execution reconstructs again
from the episode start. Replaying recorded actions re-creates recorded data: it is diagnostic
infrastructure rather than new environment interaction, and it is not part of any action budget,
so MPCResult.raw_steps counts control actions only.

The model's current observation is a live render of the reconstructed state while its goal is the
recorded dataset frame at the goal step. PushT's renderer draws a goal overlay at a fixed pose that
_set_goal_state does not move, so a goal frame cannot be rendered from a goal state; live renders
of a state also differ slightly from the recorded frame of that state. The asymmetry is left as it
is and measured, not repaired.

The semantic task is the T block's pose alone: position error in pixels and wrapped angle error,
the block being directional, with PushT's own tolerances of 20 pixels and pi / 9. The environment's
official success additionally requires the agent to be at its goal position; both are reported, and
the semantic cost d_p / 512 + d_theta / pi is the continuous task error the Atlas ranks against.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401  (registers the compression filter of the official pixels)
import numpy as np
import torch

from planner_atlas.data import ActionStats
from planner_atlas.envs import set_state_and_goal
from planner_atlas.evaluation import (
    GOAL_OFFSET,
    MPCResult,
    TaskOutcome,
    execute_blocks,
    run_mpc,
)
from planner_atlas.models.reference_lewm import ReferenceLeWM
from planner_atlas.planning import PlanResult

ARENA_SIZE = 512.0  # pixels, the PushT window; normalizes the block position error
SUCCESS_RADIUS = 20.0  # pixels, PushT's own position tolerance
SUCCESS_ANGLE = np.pi / 9  # radians, PushT's own angle tolerance
BLOCK_POSE = slice(2, 5)  # block x, y and angle within a PushT state


@dataclass(frozen=True)
class PushTCase:
    """A start and the goal GOAL_OFFSET steps later in one episode of the official dataset.

    The start is reproduced by replaying replay_actions from episode_start_state, so a case carries
    the recorded actions a_0..a_{start_step - 1} next to the recorded states and frames.
    """

    episode: int
    episode_row: int  # dataset row of the episode start
    start_step: int
    goal_step: int
    episode_start_state: np.ndarray  # 7-d state at the episode start, where the scene is at rest
    replay_actions: np.ndarray  # recorded raw actions [start_step, 2] from the episode start
    start_state: np.ndarray  # recorded 7-d state at the start step
    goal_state: np.ndarray
    start_frame: np.ndarray  # uint8 [224, 224, 3], recorded
    goal_frame: np.ndarray

    @property
    def initial_task(self) -> dict[str, float]:
        """How far the block still has to move at the recorded start."""
        position_error, angle_error = block_pose_errors(
            self.start_state[BLOCK_POSE], self.goal_state[BLOCK_POSE]
        )
        return {
            "initial_task_cost": float(task_cost(position_error, angle_error)),
            "initial_position_error": float(position_error),
            "initial_angle_error": float(angle_error),
        }


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Angles wrapped to [-pi, pi); scalars are fine, as everywhere below."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def block_pose_errors(pose: np.ndarray, goal_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position error in pixels and wrapped angle error in radians of block poses [..., 3]."""
    position_error = np.linalg.norm(pose[..., :2] - goal_pose[..., :2], axis=-1)
    angle_error = np.abs(wrap_angle(pose[..., 2] - goal_pose[..., 2]))
    return position_error, angle_error


def task_cost(position_error: np.ndarray, angle_error: np.ndarray) -> np.ndarray:
    """Semantic task cost: position error in arena widths plus angle error in half turns."""
    return position_error / ARENA_SIZE + angle_error / np.pi


def semantic_success(position_error: np.ndarray, angle_error: np.ndarray) -> np.ndarray:
    """Whether the block sits at its goal pose, within PushT's own tolerances."""
    return (position_error < SUCCESS_RADIUS) & (angle_error < SUCCESS_ANGLE)


def pusht_outcome(poses: np.ndarray, goal_pose: np.ndarray, *, reached: bool) -> TaskOutcome:
    """PushT's task outcome from the block poses [K, 3] after each raw action of an execution."""
    position_error, angle_error = block_pose_errors(poses, goal_pose)
    success = semantic_success(position_error, angle_error)
    return {
        "task_cost": float(task_cost(position_error[-1], angle_error[-1])),
        "position_error": float(position_error[-1]),
        "angle_error": float(angle_error[-1]),
        "semantic_final_success": bool(success[-1]),
        "semantic_reached_success": bool(success.any()),
        "official_reached_goal": bool(reached),
    }


def sample_pusht_cases(
    path: Path, *, num_cases: int, seed: int, goal_offset: int = GOAL_OFFSET
) -> list[PushTCase]:
    """Distinct start rows drawn uniformly among those whose goal lies in the same episode and
    whose block is not already at the goal pose (the semantic task is not already solved)."""
    with h5py.File(path, "r") as file:
        lengths, offsets, states = file["ep_len"][:], file["ep_offset"][:], file["state"][:]
        starts = np.concatenate(
            [
                offset + np.arange(length - goal_offset)
                for length, offset in zip(lengths, offsets, strict=True)
            ]
        )
        errors = block_pose_errors(
            states[starts][:, BLOCK_POSE], states[starts + goal_offset][:, BLOCK_POSE]
        )
        starts = starts[~semantic_success(*errors)]
        rows = np.sort(np.random.default_rng(seed).choice(starts, size=num_cases, replace=False))
        episodes = np.searchsorted(offsets, rows, side="right") - 1
        actions, pixels = file["action"], file["pixels"]
        return [
            PushTCase(
                episode=int(episode),
                episode_row=int(offsets[episode]),
                start_step=int(row - offsets[episode]),
                goal_step=int(row - offsets[episode] + goal_offset),
                episode_start_state=states[offsets[episode]],
                replay_actions=actions[offsets[episode] : row],
                start_state=states[row],
                goal_state=states[row + goal_offset],
                start_frame=pixels[row],
                goal_frame=pixels[row + goal_offset],
            )
            for episode, row in zip(episodes, rows, strict=True)
        ]


def reconstruct_pusht_start(env: gym.Env, case: PushTCase) -> gym.Env:
    """Put env at the case start by replaying its episode.

    Reset, restore the recorded episode start together with the case goal, then apply the recorded
    raw actions up to the start step. The recorded actions are replayed exactly as recorded, a few
    of which leave [-1, 1]; only planned actions are held to that bound.
    """
    env.reset(seed=0)
    set_state_and_goal(env, case.episode_start_state, case.goal_state)
    for action in case.replay_actions:
        env.step(action)
    return env


def rollout_pusht(
    env: gym.Env, case: PushTCase, stats: ActionStats, blocks: torch.Tensor
) -> tuple[np.ndarray, TaskOutcome]:
    """Open-loop execution of blocks [T, 10] from a freshly reconstructed case start: block-end
    frames and the task outcome."""
    reconstruct_pusht_start(env, case)
    frames, infos, reached = execute_blocks(env, blocks, stats)
    return frames, pusht_outcome(_block_poses(infos), case.goal_state[BLOCK_POSE], reached=reached)


def run_pusht_mpc(
    env: gym.Env,
    model: ReferenceLeWM,
    case: PushTCase,
    stats: ActionStats,
    goal: torch.Tensor,
    plan: Callable[[torch.Tensor, torch.Tensor], PlanResult],
    *,
    max_raw_steps: int = 50,
) -> MPCResult:
    """Block-level MPC from a reconstructed case start, stopping at semantic success.

    The block reaching its goal pose ends the episode at that raw step, even mid-block, and even
    though PushT itself would carry on until the agent also stands at its goal position. Whether
    the environment terminated before the episode stopped is kept as metadata under
    official_reached_before_stop: it is not an official-protocol score, because the episode is
    meant to stop early.
    """
    goal_pose = case.goal_state[BLOCK_POSE]

    def at_goal(info: dict) -> bool:
        return bool(semantic_success(*block_pose_errors(info["block_pose"], goal_pose)))

    reconstruct_pusht_start(env, case)
    infos, reached = run_mpc(
        env, model, stats, goal, plan, max_raw_steps=max_raw_steps, stop=at_goal
    )
    outcome = pusht_outcome(_block_poses(infos), goal_pose, reached=reached)
    outcome["official_reached_before_stop"] = outcome.pop("official_reached_goal")
    return MPCResult(len(infos), outcome)


def _block_poses(infos: list[dict]) -> np.ndarray:
    """Block poses [K, 3] after each raw action; PushT reports the angle unwrapped."""
    return np.array([info["block_pose"] for info in infos])
