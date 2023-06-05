import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class ObservationMode(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=(1, 15, 15), dtype=np.float32)

    def observation(self, obs):
        board = obs['board']
        block_1 = obs['block_1']
        block_2 = obs['block_2']
        block_3 = obs['block_3']
        total_board = np.zeros((15, 15), dtype=np.uint8)
        total_board[0:9, 3:12] = board
        total_board[10:15, 0:5] = block_1
        total_board[10:15, 5:10] = block_2
        total_board[10:15, 10:15] = block_3
        total_board = total_board.astype(np.float32)
        return np.expand_dims(total_board, axis=0)
