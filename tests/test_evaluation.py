from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from planner_atlas.data import ActionStats
from planner_atlas.evaluation import (
    GOAL_OFFSET,
    SUCCESS_RADIUS,
    TwoRoomCase,
    execute_blocks,
    frames_to_tensor,
    run_mpc,
    sample_tworoom_cases,
)

IDENTITY = ActionStats(mean=torch.zeros(2), std=torch.ones(2))


class FakeEnv:
    """Records raw actions and terminates at a chosen raw step; frames show the step count."""

    def __init__(self, terminate_at: int | None = None) -> None:
        self.actions: list[np.ndarray] = []
        self.terminate_at = terminate_at
        self.unwrapped = self  # set_state_and_goal calls the setters below

    def reset(self, seed=None):
        self.actions.clear()
        return None, {}

    def _set_state(self, state) -> None:
        pass

    def _set_goal_state(self, goal_state) -> None:
        pass

    def step(self, action):
        self.actions.append(action)
        terminated = len(self.actions) == self.terminate_at
        return None, 0.0, terminated, False, {"state": np.zeros(2)}

    def render(self) -> np.ndarray:
        return np.full((224, 224, 3), len(self.actions), dtype=np.uint8)


def test_frames_to_tensor_scales_like_the_reference() -> None:
    frames = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=-1)
    tensor = frames_to_tensor(frames, "cpu")
    assert tensor.shape == (3, 16, 16)
    expected = torch.arange(256).view(16, 16).float().mul_(1.0 / 255.0)
    assert all(torch.equal(channel, expected) for channel in tensor)


def test_frames_to_tensor_rejects_float_frames() -> None:
    with pytest.raises(ValueError):
        frames_to_tensor(np.zeros((224, 224, 3), dtype=np.float32), "cpu")


def test_execute_blocks_takes_five_raw_actions_per_block_in_time_order() -> None:
    stats = ActionStats(mean=torch.tensor([0.1, -0.2]), std=torch.tensor([0.5, 0.25]))
    blocks = torch.linspace(-1.0, 1.0, 20).view(2, 10)
    env = FakeEnv(terminate_at=3)
    frames, infos, reached = execute_blocks(env, blocks, stats)
    raw = blocks.view(10, 2) * stats.std + stats.mean
    np.testing.assert_allclose(np.stack(env.actions), raw.numpy(), rtol=0, atol=1e-7)
    assert [frame[0, 0, 0] for frame in frames] == [5, 10] and len(infos) == 2
    assert reached  # and execution continued past the termination to keep block alignment

    one_block = FakeEnv()
    assert not execute_blocks(one_block, blocks[:1], stats)[2]
    assert len(one_block.actions) == 5


def test_execute_blocks_refuses_non_executable_actions() -> None:
    env = FakeEnv()
    with pytest.raises(ValueError):
        execute_blocks(env, torch.full((1, 10), 1.5), IDENTITY)
    assert env.actions == []


def test_mpc_stops_mid_block_at_success_and_replans_only_between_blocks() -> None:
    case = TwoRoomCase(0, 0, GOAL_OFFSET, np.zeros(2), np.full(2, 50.0), None, None)
    model = SimpleNamespace(encode=lambda frames: torch.zeros(*frames.shape[:2], 192))

    def run(env):
        planned_at = []

        def plan(latent, goal):
            planned_at.append(len(env.actions))
            return SimpleNamespace(actions=torch.zeros(5, 10))

        return run_mpc(env, model, case, IDENTITY, torch.zeros(1, 192), plan), planned_at

    result, planned_at = run(FakeEnv(terminate_at=7))  # second raw action of the second block
    assert planned_at == [0, 5] and result.success and result.raw_steps == 7

    result, planned_at = run(FakeEnv())
    assert planned_at == list(range(0, 50, 5)) and not result.success and result.raw_steps == 50


def test_case_sampling_is_deterministic_and_skips_solved_starts(tmp_path) -> None:
    lengths = [30, 40, 26, 10]  # 5, 15, 1 and 0 starts with an in-episode goal
    offsets = np.cumsum([0, *lengths[:-1]])
    total = sum(lengths)
    proprio = np.stack([3.0 * np.arange(total), np.zeros(total)], axis=1).astype(np.float32)
    # the agent never moves in episode 1, so every start there is already solved
    proprio[offsets[1] : offsets[1] + lengths[1]] = 100.0
    with h5py.File(tmp_path / "tworoom.h5", "w") as file:
        file["ep_len"], file["ep_offset"], file["proprio"] = lengths, offsets, proprio
        file["pixels"] = np.broadcast_to(
            np.arange(total, dtype=np.uint8)[:, None, None, None], (total, 2, 2, 3)
        )

    first, second = (
        sample_tworoom_cases(tmp_path / "tworoom.h5", num_cases=6, seed=3) for _ in range(2)
    )
    assert [(case.episode, case.start_step) for case in first] == [
        (case.episode, case.start_step) for case in second
    ]
    assert {case.episode for case in first} == {0, 2}  # all six unsolved starts
    for case in first:
        assert case.goal_step == case.start_step + GOAL_OFFSET < lengths[case.episode]
        assert case.initial_distance >= SUCCESS_RADIUS
        start, goal = (
            offsets[case.episode] + case.start_step,
            offsets[case.episode] + case.goal_step,
        )
        np.testing.assert_array_equal(case.start_state, proprio[start])
        np.testing.assert_array_equal(case.goal_state, proprio[goal])
        assert case.start_frame[0, 0, 0] == start and case.goal_frame[0, 0, 0] == goal
    with pytest.raises(ValueError):
        sample_tworoom_cases(tmp_path / "tworoom.h5", num_cases=7, seed=3)
