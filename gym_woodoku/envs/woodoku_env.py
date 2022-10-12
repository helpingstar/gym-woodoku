import gym
# import pygame
import numpy as np
from gym import spaces
from blocks import get_3_blocks

MAX_BLOCK_NUM = 3


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
        self._block_exist = [True, True, True]

        # get observation and info
        observation = self._get_obs()
        info = self._get_info()

        # 연속으로 몇 개를 부쉈는지를 나타낸다(한번에 몇개를 부쉈는지랑 다르다)
        self.straight = 0

        return observation, info

    def _get_3_blocks_random(self):
        # randomly select three blocks
        self._block_1, self._block_2, self._block_3 = get_3_blocks()

    def _get_obs(self):
        return {
            "board": self._board,
            "block_1": self._block_1,
            "block_2": self._block_2,
            "block_3": self._block_3
        }

    def _get_combined_obs(self):
        # Four states are combined into one 15x15 array as states for deep learning.
        comb_state = np.zeros((15, 15))
        comb_state[0:9, 3:11] = self._board
        comb_state[10:15, 0:5] = self._block_1
        comb_state[10:15, 5:10] = self._block_2
        comb_state[10:15, 10:15] = self._block_3
        return comb_state

    def _get_info(self):
        return {}

    def _is_terminated(self) -> bool:
        # 소유한 블록으로 더 이상 게임이 진행 가능한 지 체크한다
        # is_valid_position함수를 3X9X9 경우의 수에 대입해서 모두 false가 나오면
        # true를 리턴한다.
        for blk_num in range(MAX_BLOCK_NUM):
            if self._block_exist[blk_num]:
                for act in range(blk_num*81, (blk_num+1) * 81):
                    if self._is_valid_position(act):
                        return False
        return True

    def _nonexist_block(self, action):
        self._block_exist[action // 81] = False

    def _is_valid_position(self, action) -> bool:
        # 해당 블록을 해당 action을 통해 가능한 위치인지 판단한다.
        block = None
        # block의 중심이 놓일 위치 [0~8, 0~8]
        location = None

        # 첫 번째 블록 선택 시
        if 0 <= action and action <= 80:
            block = self._block_1
            location = [action // 9, action % 9]

        # 두 번째 블록 선택 시
        elif 81 <= action and action <= 161:
            block = self._block_2
            location = [(action - 81) // 9, (action - 81) % 9]

        # 세 번째 블록 선택 시
        else:
            block = self._block_3
            location = [(action - 162) // 9, (action - 162) % 9]

        # board와 block 비교
        for col in range(0, 5):
            for row in range(0, 5):
                # 5*5에 block이 존재할 때
                if block[row][col] == 1:
                    # location - 2 : 놓을 위치가 차지하는 5x5 중 (0,0)=(왼쪽위)
                    # board 위에 존재하지 않을 때
                    if not (0 <= (location[0] - 2 + col) < 9 and 0 <= (location[1] - 2 + row) < 9):
                        return False

                    # board 위에 존재하지만 block이 있을 때
                    if self._board[location[0] - 2 + col][location[1] - 2 + row] == 1:
                        return False

        return True

    def _is_valid_block(self, action) -> bool:
        # 해당 블록이 현재 유효한 블록인지 판단한다.

        if self._block_exist[action // 81]:
            return True
        else:
            return False

    def _crash_block(self, action) -> int:
        # 부술 블록이 있으면 부수고 추가점수를 리턴한다.
        # 부순 상태에 따라 self.combo를 업데이트한다.
        pass

    def step(self, action):
        """
        https://www.gymlibrary.dev/api/core/#gym.Env.step
        return (observation, reward, terminated, truncated, info)
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        terminated = False

        # if) action에 해당하는 블록이 존재하는가?
        # no -> return observation, 0, False, False, info
        if not self._is_valid_block(self, action):
            return (self._get_obs(), self.reward, terminated, False, self._get_info())

        # if) n을 action위치에 놓을 수 있는가?
        # is_valid_position 이용
        # no -> return observation, 0, False, False, info
        if not self._is_valid_position(self, action):
            return (self._get_obs(), self.reward, terminated, False, self._get_info())

        self._nonexist_block(action)

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
        if sum(self._block_exist) == 0:
            self._get_3_blocks_random()

        # if 더이상 게임을 수행할 수 있는가?
        # _is_terminated 이용
        # terminated = _is_terminated
        terminated = self._is_terminated()

        # return observation, reward, terminated, False, info

    def _line_printer(line: np.ndarray):
        return np.array2string(line, separator='', formatter={'str_kind': lambda x: x})

    @property
    def block_exist(self):
        return self._block_exist

    def render(self):
        display_height = 17
        display_width = 21
        display_score_top = 1

        new_board = np.where(self._board == 1, '■', '□')
        new__block_1 = np.where(self._block_1 == 1, '■', '□')
        new__block_2 = np.where(self._block_2 == 1, '■', '□')
        new__block_3 = np.where(self._block_3 == 1, '■', '□')

        game_display = np.full(
            (display_height, display_width), ' ', dtype='<U1')

        # copy board
        game_display[1:10, 1:10] = new_board

        # copy block
        for i, block in enumerate([new__block_1, new__block_2, new__block_3]):
            game_display[11:16, 7*i+1:7*i+6] = block

        # create score_board
        game_display[display_score_top+1,
                     11:20] = np.array(list('┌'+'─'*7+'┐'))
        game_display[display_score_top+2,
                     11:20] = np.array(list('│'+' SCORE '+'│'))
        game_display[display_score_top+3,
                     11:20] = np.array(list('├'+'─'*7+'┤'))
        game_display[display_score_top+4,
                     11:20] = np.array(list('│'+'0'*7+'│'))
        game_display[display_score_top+5,
                     11:20] = np.array(list('└'+'─'*7+'┘'))

        # Display game_display
        for i in range(display_height):
            print(self._line_printer(game_display[i])[1:-1])

    def close(self):
        pass
