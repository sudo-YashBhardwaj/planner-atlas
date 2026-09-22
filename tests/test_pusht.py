from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from planner_atlas.data import ActionStats
from planner_atlas.envs import env_state, make_env, set_state_and_goal
from planner_atlas.evaluation import GOAL_OFFSET
from planner_atlas.pusht import (
    ARENA_SIZE,
    BLOCK_POSE,
    SUCCESS_ANGLE,
    SUCCESS_RADIUS,
    PushTCase,
    block_pose_errors,
    pusht_outcome,
    reconstruct_pusht_start,
    render_pusht_goal,
    replay_frames,
    rollout_pusht,
    run_pusht_mpc,
    sample_pusht_cases,
    semantic_success,
    task_cost,
    wrap_angle,
)

IDENTITY = ActionStats(mean=torch.zeros(2), std=torch.ones(2))


GOAL_STATE = np.array([0.0, 0.0, 256.0, 256.0, 0.5, 0.0, 0.0])  # block pose (256, 256, 0.5)
AWAY_POSE = np.array([100.0, 100.0, 0.5])  # far outside the 20 pixel success radius


class FakeEnv:
    """Records every raw action it is given; its block jumps onto the goal pose at succeed_at."""

    def __init__(self, terminate_at: int | None = None, succeed_at: int | None = None) -> None:
        self.actions: list[np.ndarray] = []
        self.terminate_at = terminate_at
        self.succeed_at = succeed_at
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
        at_goal = self.succeed_at is not None and len(self.actions) >= self.succeed_at
        pose = GOAL_STATE[BLOCK_POSE] if at_goal else AWAY_POSE
        return None, 0.0, terminated, False, {"block_pose": pose}

    def render(self) -> np.ndarray:
        return np.zeros((224, 224, 3), dtype=np.uint8)


def make_case(replay_actions: np.ndarray, goal_state: np.ndarray = GOAL_STATE) -> PushTCase:
    state = np.zeros(7)
    return PushTCase(
        episode=0,
        episode_row=0,
        start_step=len(replay_actions),
        goal_step=len(replay_actions) + GOAL_OFFSET,
        episode_start_state=state,
        replay_actions=replay_actions,
        goal_actions=np.zeros((GOAL_OFFSET, 2), dtype=np.float32),
        start_state=state,
        goal_state=goal_state,
        start_frame=None,
    )


def constant_plan(env: FakeEnv, planned_at: list[int]):
    """A planner that records when it was called, in raw actions the env has already taken."""

    def plan(latent, goal):
        planned_at.append(len(env.actions))
        return SimpleNamespace(actions=torch.zeros(5, 10))

    return plan


def test_wrap_angle_near_pi() -> None:
    assert wrap_angle(np.pi - 1e-9) == pytest.approx(np.pi - 1e-9)
    assert wrap_angle(np.pi + 1e-9) == pytest.approx(-np.pi + 1e-9)
    assert wrap_angle(-np.pi + 1e-9) == pytest.approx(-np.pi + 1e-9)
    np.testing.assert_allclose(
        wrap_angle(np.array([0.0, 3 * np.pi, -3 * np.pi, 2 * np.pi - 1e-9])),
        [0.0, -np.pi, -np.pi, -1e-9],
        atol=1e-12,
    )
    # the error itself is symmetric across the cut
    for angle in (np.pi - 1e-6, np.pi + 1e-6):
        assert abs(wrap_angle(angle)) == pytest.approx(np.pi - 1e-6)


def test_task_cost_on_hand_computed_poses() -> None:
    goal_pose = np.array([200.0, 100.0, 0.5])
    pose = np.array([203.0, 104.0, 0.5 + np.pi / 2])  # 5 pixels away, a quarter turn off
    position_error, angle_error = block_pose_errors(pose, goal_pose)
    assert position_error == pytest.approx(5.0) and angle_error == pytest.approx(np.pi / 2)
    assert task_cost(position_error, angle_error) == pytest.approx(5.0 / ARENA_SIZE + 0.5)

    # the block is directional: half a turn is the worst angle, not a match
    _, half_turn = block_pose_errors(goal_pose + np.array([0.0, 0.0, np.pi]), goal_pose)
    assert half_turn == pytest.approx(np.pi) and task_cost(0.0, half_turn) == pytest.approx(1.0)


def test_semantic_success_thresholds_are_strict() -> None:
    assert semantic_success(SUCCESS_RADIUS - 1e-9, SUCCESS_ANGLE - 1e-9)
    assert not semantic_success(SUCCESS_RADIUS, 0.0)
    assert not semantic_success(0.0, SUCCESS_ANGLE)
    np.testing.assert_array_equal(
        semantic_success(np.array([0.0, 25.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        [True, False, False],
    )


def test_outcome_separates_final_reached_and_official_success() -> None:
    goal_pose = np.array([256.0, 256.0, 0.0])
    poses = np.array([[256.0, 256.0, 0.0], [256.0, 300.0, 0.0]])  # at the goal, then pushed away
    outcome = pusht_outcome(poses, goal_pose, reached=False)
    assert outcome["semantic_reached_success"] and not outcome["semantic_final_success"]
    assert not outcome["official_reached_goal"]
    assert outcome["position_error"] == pytest.approx(44.0)
    assert outcome["angle_error"] == pytest.approx(0.0)
    assert outcome["task_cost"] == pytest.approx(44.0 / ARENA_SIZE)
    # the environment's own success is reported as it is, next to the semantic one
    assert pusht_outcome(poses[:1], goal_pose, reached=True) == {
        "task_cost": 0.0,
        "position_error": 0.0,
        "angle_error": 0.0,
        "semantic_final_success": True,
        "semantic_reached_success": True,
        "official_reached_goal": True,
    }


def test_reconstruction_replays_the_episode_and_never_sets_a_mid_episode_state() -> None:
    episode_start = np.array([256.0, 255.0, 256.0, 300.0, 0.0, 0.0, 0.0])
    goal_state = np.array([256.0, 256.0, 300.0, 320.0, np.pi / 2, 0.0, 0.0])
    actions = np.array([[0.0, 1.0]] * 8, dtype=np.float32)  # drives the agent into the block
    states = []

    with make_env("pusht") as env:
        env.reset(seed=0)
        set_state_and_goal(env, episode_start, goal_state)
        recorded = np.stack([env.step(action)[0]["state"] for action in actions])
        case = PushTCase(
            episode=0,
            episode_row=0,
            start_step=5,
            goal_step=5 + GOAL_OFFSET,
            episode_start_state=episode_start,
            replay_actions=actions[:5],
            goal_actions=actions[5:],
            start_state=recorded[4],
            goal_state=goal_state,
            start_frame=None,
        )
        # the replayed actions really push and turn the block: contact, not free motion
        assert np.linalg.norm(recorded[4, 2:4] - episode_start[2:4]) > 100
        assert abs(recorded[4, 4] - episode_start[4]) > 0.5

        set_states = []
        env.unwrapped._set_state = _recording(env.unwrapped._set_state, set_states)
        for seed in (1, 2):  # a different episode in between: only the replay may set the state
            env.reset(seed=seed)
            env.step(np.array([0.7, -0.3], dtype=np.float32))
            reconstruct_pusht_start(env, case)
            states.append((env_state(env), env.render()))

    (state, frame), (again, frame_again) = states
    np.testing.assert_array_equal(state, again)  # identical, bit for bit
    np.testing.assert_array_equal(frame, frame_again)
    np.testing.assert_array_equal(state, recorded[4])  # and it is the recorded start
    assert any(np.array_equal(each, episode_start) for each in set_states)
    assert not any(np.array_equal(each, case.start_state) for each in set_states)


def test_goal_is_the_live_render_of_the_replayed_goal_step(tmp_path) -> None:
    """The goal observation is the frame the latent cache holds for the goal row, bit for bit."""
    episode_start = np.array([256.0, 255.0, 256.0, 300.0, 0.0, 0.0, 0.0])
    length, start_step = 34, 6
    generator = np.random.default_rng(0)
    actions = np.zeros((length, 2), dtype=np.float32)
    actions[:8] = [0.0, 1.0]  # push into the block, so the goal pose differs from the start
    actions[8:] = generator.uniform(-1, 1, size=(length - 8, 2))
    with make_env("pusht") as env:
        env.reset(seed=0)
        set_state_and_goal(env, episode_start, episode_start)
        states = [episode_start]
        for action in actions[:-1]:
            states.append(env.step(action)[0]["state"])
        states = np.stack(states)
        actions[-1] = np.nan  # the boundary action of an episode, as in the official data
        with h5py.File(tmp_path / "pusht.h5", "w") as file:
            file["ep_len"], file["ep_offset"] = [length], [0]
            file["state"], file["action"] = states, actions

        goal_step = start_step + GOAL_OFFSET
        case = PushTCase(
            episode=0,
            episode_row=0,
            start_step=start_step,
            goal_step=goal_step,
            episode_start_state=episode_start,
            replay_actions=actions[:start_step],
            goal_actions=actions[start_step:goal_step],
            start_state=states[start_step],
            goal_state=states[goal_step],
            start_frame=None,
        )
        env.reset(seed=3)
        env.step(np.array([0.5, 0.5], dtype=np.float32))  # a stale episode in between
        goal = render_pusht_goal(env, case)
        np.testing.assert_array_equal(env_state(env), states[goal_step])  # replay is exact here
        cached = list(replay_frames(env, tmp_path / "pusht.h5"))
        np.testing.assert_array_equal(goal, cached[goal_step])
        assert not np.array_equal(goal, cached[start_step])

        reconstruct_pusht_start(env, case)  # and the start is still reachable afterwards
        np.testing.assert_array_equal(env.render(), cached[start_step])


def test_replayed_actions_are_not_part_of_the_action_budget() -> None:
    case = make_case(np.zeros((7, 2), dtype=np.float32))
    model = SimpleNamespace(encode=lambda frames: torch.zeros(*frames.shape[:2], 192))
    env = FakeEnv()  # its block never reaches the goal, so only the budget can end the episode
    plan = constant_plan(env, [])
    result = run_pusht_mpc(env, model, case, IDENTITY, torch.zeros(1, 192), plan, max_raw_steps=10)
    assert result.raw_steps == 10  # the 7 replayed actions are reconstruction, not control
    assert len(env.actions) == 7 + 10

    frames, _ = rollout_pusht(env, case, IDENTITY, torch.zeros(2, 10))
    assert len(env.actions) == 7 + 10 and len(frames) == 2  # replay again, then 2 blocks


def test_mpc_stops_mid_block_at_semantic_success_without_official_termination() -> None:
    case = make_case(np.zeros((3, 2), dtype=np.float32))
    model = SimpleNamespace(encode=lambda frames: torch.zeros(*frames.shape[:2], 192))
    env = FakeEnv(succeed_at=3 + 7)  # the block lands on its goal at the seventh control action
    planned_at: list[int] = []
    result = run_pusht_mpc(
        env,
        model,
        case,
        IDENTITY,
        torch.zeros(1, 192),
        constant_plan(env, planned_at),
        max_raw_steps=50,
    )
    assert result.raw_steps == 7  # stopped inside the second block, not at its end
    assert len(env.actions) == 3 + 7
    assert planned_at == [3, 8]  # replanning still happens only between blocks
    assert result.task["semantic_reached_success"]
    assert "semantic_final_success" not in result.task  # stopping at success makes it redundant
    # the environment never terminated: its official success also wants the agent at its goal
    assert result.task["official_reached_before_stop"] is False


def test_case_sampling_is_deterministic_and_skips_solved_starts(tmp_path) -> None:
    lengths = [30, 40, 26]  # 5, 15 and 1 starts with an in-episode goal
    offsets = np.cumsum([0, *lengths[:-1]])
    total = sum(lengths)
    states = np.zeros((total, 7), dtype=np.float32)
    states[:, 2] = 3.0 * np.arange(total)  # the block slides 3 pixels per step
    states[:, 4] = 0.01 * np.arange(total)
    states[offsets[1] : offsets[1] + lengths[1]] = 7.0  # episode 1 never moves: already solved
    actions = np.stack([np.arange(total), -np.arange(total)], axis=1).astype(np.float32)
    with h5py.File(tmp_path / "pusht.h5", "w") as file:
        file["ep_len"], file["ep_offset"], file["state"], file["action"] = (
            lengths,
            offsets,
            states,
            actions,
        )
        file["pixels"] = np.broadcast_to(
            np.arange(total, dtype=np.uint8)[:, None, None, None], (total, 2, 2, 3)
        )

    first, second = (
        sample_pusht_cases(tmp_path / "pusht.h5", num_cases=6, seed=3) for _ in range(2)
    )
    assert [(case.episode, case.start_step) for case in first] == [
        (case.episode, case.start_step) for case in second
    ]
    assert {case.episode for case in first} == {0, 2}  # all six unsolved starts
    for case in first:
        start = offsets[case.episode] + case.start_step
        assert case.goal_step == case.start_step + GOAL_OFFSET < lengths[case.episode]
        assert case.episode_row == offsets[case.episode]
        assert case.initial_task["initial_position_error"] >= SUCCESS_RADIUS
        np.testing.assert_array_equal(case.episode_start_state, states[case.episode_row])
        np.testing.assert_array_equal(case.replay_actions, actions[case.episode_row : start])
        np.testing.assert_array_equal(case.goal_actions, actions[start : start + GOAL_OFFSET])
        np.testing.assert_array_equal(case.start_state, states[start])
        np.testing.assert_array_equal(case.goal_state, states[start + GOAL_OFFSET])
        assert case.start_frame[0, 0, 0] == start
        assert not hasattr(case, "goal_frame")  # the recorded goal frame is never a model input
    with pytest.raises(ValueError):
        sample_pusht_cases(tmp_path / "pusht.h5", num_cases=7, seed=3)


def _recording(method, calls: list[np.ndarray]):
    """method, wrapped so that every state it is called with is recorded."""

    def wrapper(state):
        calls.append(np.asarray(state, dtype=float).copy())
        return method(state)

    return wrapper
