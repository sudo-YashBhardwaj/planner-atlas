"""TwoRoom and PushT from stable-worldmodel.

This module owns the upstream environment IDs, the import that registers them, and the
private state setters used by the LeWM reference evaluation.
"""

from typing import Literal

import gymnasium as gym
import numpy as np

# Importing stable_worldmodel registers its swm/* environments with Gymnasium.
import stable_worldmodel  # noqa: F401

EnvName = Literal["tworoom", "pusht"]

_ENV_IDS: dict[EnvName, str] = {"tworoom": "swm/TwoRoom-v1", "pusht": "swm/PushT-v1"}


def make_env(name: EnvName) -> gym.Env:
    return gym.make(_ENV_IDS[name], render_mode="rgb_array")


def set_state_and_goal(env: gym.Env, state: np.ndarray, goal_state: np.ndarray) -> None:
    """Set the start and goal of a freshly reset env, as the LeWM reference evaluation does.

    TwoRoom states are agent xy and target xy. PushT states are [agent xy, block xy,
    block angle, agent velocity]; they omit the block's velocities and Pymunk's contact
    state, so an arbitrary mid-episode PushT state cannot be restored exactly.

    The upstream setters are private; this is deliberately the only place that calls them.
    """
    env.unwrapped._set_state(state)
    env.unwrapped._set_goal_state(goal_state)
