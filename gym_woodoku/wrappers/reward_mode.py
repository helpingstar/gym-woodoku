import gymnasium as gym
import numpy as np


class RewardMode(gym.Wrapper):
    def __init__(self, env, mode):
        super().__init__(env)
        self.mode = mode

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        straight = info['straight']
        combo = info['combo']
        n_cell = info['n_cell']

        if self.mode == 'woodoku':
            if combo == 0:
                reward = self.n_cell
            else:
                reward = 28 * combo + 10 * straight + n_cell - 20
        elif self.mode == 'non_straight':
            if combo == 0:
                reward = 0.1
            else:
                reward = 1 + combo


        return obs, reward, terminated, truncated, info
