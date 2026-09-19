import numpy as np
import pytest

from planner_atlas.envs import EnvName, make_env, set_state_and_goal

NAMES: list[EnvName] = ["tworoom", "pusht"]
ZERO_ACTION = np.zeros(2, dtype=np.float32)

# A valid (state, goal_state) pair per environment, in the format set_state_and_goal takes.
STATES = {
    "tworoom": (np.array([60.0, 112.0]), np.array([164.0, 112.0])),
    "pusht": (
        np.array([100.0, 100.0, 256.0, 300.0, np.pi / 4, 0.0, 0.0]),
        np.array([100.0, 100.0, 256.0, 256.0, np.pi / 4, 0.0, 0.0]),
    ),
}


def assert_rgb_frame(frame: np.ndarray) -> None:
    assert frame.shape == (224, 224, 3)
    assert frame.dtype == np.uint8


@pytest.mark.parametrize("name", NAMES)
def test_make_env(name: EnvName) -> None:
    with make_env(name) as env:
        env.reset(seed=0)
        assert_rgb_frame(env.render())
        assert env.action_space.shape == (2,)
        np.testing.assert_array_equal(env.action_space.low, -1.0)
        np.testing.assert_array_equal(env.action_space.high, 1.0)


@pytest.mark.parametrize("name", NAMES)
def test_step(name: EnvName) -> None:
    with make_env(name) as env:
        env.reset(seed=0)
        obs, reward, terminated, truncated, _ = env.step(ZERO_ACTION)
        assert env.observation_space.contains(obs)
    assert np.isscalar(reward) and np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))


@pytest.mark.parametrize("name", NAMES)
def test_set_state_and_goal(name: EnvName) -> None:
    state, goal_state = STATES[name]
    with make_env(name) as env:
        env.reset(seed=0)
        set_state_and_goal(env, state, goal_state)
        assert_rgb_frame(env.render())
        *_, info = env.step(ZERO_ACTION)
    np.testing.assert_allclose(info["goal_state"], goal_state)


def test_tworoom_state_and_goal_determine_trajectory() -> None:
    state, goal_state = STATES["tworoom"]
    # Into the wall, up along it, then through the door: exercises the collision logic.
    actions = np.array([[1, 0]] * 12 + [[0, -1]] * 10 + [[1, 0]] * 12, dtype=np.float32)
    trajectories = []
    with make_env("tworoom") as env:
        for seed in (0, 1):  # different resets: the set state alone must determine the rollout
            env.reset(seed=seed)
            set_state_and_goal(env, state, goal_state)
            observations = np.stack([np.asarray(env.step(action)[0]) for action in actions])
            trajectories.append((observations, env.render()))
    (obs_a, frame_a), (obs_b, frame_b) = trajectories
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(frame_a, frame_b)
