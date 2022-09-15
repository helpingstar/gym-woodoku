import gym
from gym import error, spaces, utils
from gym.utils import seeding


class WoodokuEnv(gym.Env):

    def __init__(self):
        pass

    def step(self, action):
        reward = action
        self.state += action
        return (reward)

    def reset(self):
        self.state = 0

    def render(self):
        pass
