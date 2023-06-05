import gymnasium as gym


class TerminateIllegalWoodoku(gym.Wrapper):
    def __init__(self, env, illegal_reward):
        super().__init__(env)
        self._illegal_reward = illegal_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info['is_legal']:
            return obs, reward, terminated, truncated, info
        else:
            return obs, self._illegal_reward, True, truncated, info
