import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete


class ObservationMode(gym.ObservationWrapper):
    def __init__(self, env, n_channel=1):
        super().__init__(env)
        assert n_channel in [1, 4]
        self.n_channel = n_channel

        if self.n_channel == 1:
            self.observation_space = Box(low=0, high=1, shape=(1, 15, 15), dtype=np.float32)
        else:
            self.observation_space = Box(low=0, high=1, shape=(4, 9, 9), dtype=np.float32)

    def observation(self, obs):
        board = obs['board']
        block_1 = obs['block_1']
        block_2 = obs['block_2']
        block_3 = obs['block_3']
        if self.n_channel == 1:
            total_board = np.zeros((15, 15), dtype=np.uint8)
            total_board[0:9, 3:12] = board
            total_board[10:15, 0:5] = block_1
            total_board[10:15, 5:10] = block_2
            total_board[10:15, 10:15] = block_3
            total_board = total_board.astype(np.float32)
            return np.expand_dims(total_board, axis=0)
        else:
            channel_1 = np.zeros((9, 9), dtype=np.uint8)
            channel_2 = np.zeros((9, 9), dtype=np.uint8)
            channel_3 = np.zeros((9, 9), dtype=np.uint8)
            channel_1[2:7, 2:7] = block_1
            channel_2[2:7, 2:7] = block_2
            channel_3[2:7, 2:7] = block_3
            return np.stack([board, channel_1, channel_2, channel_3])
            
