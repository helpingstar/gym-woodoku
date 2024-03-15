from typing import Any, SupportsFloat
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AddStraight(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        pre_obs_space = self.observation_space
        self.observation_space = spaces.Dict(
            {"board": pre_obs_space, "straight": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)}
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        new_obs = {"board": obs, "straight": np.array([info["straight"]])}
        return new_obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, termination, truncation, info = super().step(action)
        new_obs = {"board": obs, "straight": np.array([info["straight"]])}
        return new_obs, reward, termination, truncation, info
