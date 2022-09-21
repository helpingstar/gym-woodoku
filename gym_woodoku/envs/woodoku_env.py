import gym
from gym import spaces
# import pygame
import numpy as np


class WoodokuEnv(gym.Env):

    def __init__(self, render_mode=None):

        # observation_space : (관찰의 경우의 수)
        # board : 블록을 놓을 공간
        # block : 3개의 블록들
        self.observation_space = spaces.Dict(
            {
                "board": spaces.MultiBinary([9, 9]),
                "block_1": spaces.MultiBinary([5, 5]),
                "block_2": spaces.MultiBinary([5, 5]),
                "block_3": spaces.MultiBinary([5, 5])
            }
        )

        # action_space : (액션 경우의 수)
        # 3개의 블록중 하나를 (9x9의 위치중 하나에 배치한다)
        self.action_space = spaces.MultiDiscrete(np.array([3, 9, 9]))

    def reset(self, seed=None, options=None):
        # 시드 초기화
        super().reset(seed=seed)

        # make board clean
        self._board = np.zeros((9, 9), dtype=np.uint8)

        # get 3 blocks
        self._get_3_blocks_random()

        # get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # 콤보를 나타냄
        self.combo = 0

        return observation, info

    def _get_3_blocks_random(self):
        # randomly select three blocks

        # Below code is for test, will be replaced later.
        self.block_1 = np.zeros((5, 5), np.uint8)
        self.block_2 = np.zeros((5, 5), np.uint8)
        self.block_3 = np.zeros((5, 5), np.uint8)

    def _get_obs(self):
        return {
            "board": self._board,
            "block_1": self.block_1,
            "block_2": self.block_2,
            "block_3": self.block_3
        }

    def _get_info(self):
        return {}

    def _is_terminated(self) -> bool:
        # 소유한 블록으로 더 이상 게임이 진행 가능한 지 체크한다
        # is_valid_position함수를 3X9X9 경우의 수에 대입해서 모두 false가 나오면
        # true를 리턴한다.
        pass

    def _is_valid_position(self, action) -> bool:
        # 해당 블록을 해당 action을 통해 가능한 위치인지 판단한다.
        pass

    def _is_valid_block(self, action) -> bool:
        # 해당 블록이 현재 유효한 블록인지 판단한다.
        pass

    def _crash_block(self, action) -> int:
        # 부술 블록이 있으면 부수고 추가점수를 리턴한다.
        pass

    def step(self, action):
        """
        https://www.gymlibrary.dev/api/core/#gym.Env.step
        return (observation, reward, terminated, truncated, info)
        """
        # if) action에 해당하는 블록이 존재하는가?
        # yes -> return observation, 0, False, False, info

        # if) n을 action위치에 놓을 수 있는가?
        # is_valid_position 이용
        # no -> return observation, 0, False, False, info

        # OR 연산 수행하여 블록을 놓는다.
        # 사용한 해당 블록칸을 0으로 만든다.
        # reward += 놓으면서 얻는 점수

        # if) 파괴할 블록이 있는가?
        # _crash_block 이용
        # yes -> 파괴를 반영한다.
        #           & reward += 파괴하며 얻는 점수

        # 여기서 점수 산정 방식이 블록 놓기와 블록 파괴가 독립적이라면 reward를 따로 산정하고
        # 독립적이지 않고 파괴에 한정된다면 파괴시에만 reward를 추가한다.

        # if) 블록 3개를 다 썼는가?
        # yes -> _get_3_blocks_random를 통해 리필한다.

        # if 더이상 게임을 수행할 수 있는가?
        # _is_terminated 이용
        # terminated = _is_terminated

        # return observation, reward, terminated, False, info

    def render(self):
        pass

    def close(self):
        pass
