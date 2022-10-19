import gym
# import pygame
import numpy as np
from gym import spaces
from .blocks import blocks
import random

MAX_BLOCK_NUM = 3
BLOCK_LENGTH = 5
BOARD_LENGTH = 9


class WoodokuEnv(gym.Env):

    # TODO what is render_fps
    metadata = {"game_modes": ['woodoku'],
                "obs_modes": ['divided', 'total_square'],
                "reward_modes": ['one', 'woodoku'],
                "render_modes": ['ansi', 'plot', 'pygame'],
                "render_fps": 1}

    def __init__(self, game_mode='woodoku', obs_mode='total_square', reward_mode='woodoku', render_mode='ansi', crash33=True):

        # ASSERT
        err_msg = f"{game_mode} is not in {self.metadata['game_modes']}"
        assert game_mode in self.metadata['game_modes'], err_msg
        self.game_mode = game_mode

        err_msg = f"{obs_mode} is not in {self.metadata['obs_modes']}"
        assert obs_mode in self.metadata['obs_modes'], err_msg
        self.obs_mode = obs_mode

        err_msg = f"{reward_mode} is not in {self.metadata['reward_modes']}"
        assert reward_mode in self.metadata['reward_modes']
        self.reward_mode = reward_mode

        err_msg = f"{render_mode} is not in {self.metadata['render_modes']}"
        assert render_mode is None or render_mode in self.metadata['render_modes'], err_msg
        self.render_mode = render_mode

        self.crash33 = crash33

        # define observation_space by obs_mode
        if self.obs_mode == 'divided':
            self.observation_space = spaces.Dict(
                {
                    "board": spaces.MultiBinary([9, 9]),
                    "block_1": spaces.MultiBinary([5, 5]),
                    "block_2": spaces.MultiBinary([5, 5]),
                    "block_3": spaces.MultiBinary([5, 5])
                }
            )
        elif self.obs_mode == 'total_square':
            self.observation_space = spaces.Dict(
                {
                    "total_square": spaces.MultiBinary([15, 15])
                }
            )

        # action_space : (Block X Width X Height)
        self.action_space = spaces.Discrete(243)

        # get kind of blocks by `game_mode`
        self._block_list = blocks[game_mode]

    def _get_3_blocks(self) -> tuple:
        a = random.sample(range(self._block_list.shape[0]), 3)
        return self._block_list[a[0]].copy(), self._block_list[a[1]].copy(), self._block_list[a[2]].copy()

    def reset(self, seed=None, options=None):
        # reset seed
        super().reset(seed=seed)

        # make board clean
        self._board = np.zeros((9, 9))

        # get 3 blocks
        self._get_3_blocks_random()
        self._block_exist = [True, True, True]

        # get observation and info
        observation = self._get_obs()

        info = self._get_info()

        # Shows how many pieces are broken in a row
        # (this is different from how many pieces are broken at once)
        self.straight = 0

        # score
        self.score = 0

        return observation, info

    def _get_3_blocks_random(self):
        # randomly select three blocks
        self._block_exist = [True, True, True]
        self._block_1, self._block_2, self._block_3 = self._get_3_blocks()

    def _get_obs(self):
        if self.obs_mode == 'divided':
            return {
                "board": self._board,
                "block_1": self._block_1,
                "block_2": self._block_2,
                "block_3": self._block_3
            }
        elif self.obs_mode == 'total_square':
            # Four states are combined into one 15x15 array as states   for deep learning.
            comb_state = np.zeros((15, 15))
            comb_state[0:9, 3:12] = self._board
            comb_state[10:15, 0:5] = self._block_1
            comb_state[10:15, 5:10] = self._block_2
            comb_state[10:15, 10:15] = self._block_3
            return {'total_square': comb_state}

    def _get_info(self):
        return {}

    def _is_terminated(self) -> bool:
        # Check if the game can be continued with the blocks you own
        # If any number of cases can proceed, False is returned.
        for blk_num in range(MAX_BLOCK_NUM):
            if self._block_exist[blk_num]:
                for act in range(blk_num*81, (blk_num+1) * 81):
                    if self._is_valid_position(act):
                        return False
        return True

    def _nonexist_block(self, action):
        # Deactivate the block corresponding to the action and set the array to 0.
        self._block_exist[action // 81] = False
        block, _ = self.action_to_blk_pos(action)
        block[:, :] = 0

    def _is_valid_position(self, action) -> bool:
        block, location = self.action_to_blk_pos(action)

        # board와 block 비교
        for row in range(0, BLOCK_LENGTH):
            for col in range(0, BLOCK_LENGTH):
                # Condition for block position in block array
                if block[row][col] == 1:
                    # location - 2 : leftmost top (0, 0)
                    # When the block is located outside the board
                    if not (0 <= (location[0] - 2 + row) < 9 and 0 <= (location[1] - 2 + col) < 9):
                        return False

                    # When there is already another block
                    if self._board[location[0] - 2 + row][location[1] - 2 + col] == 1:
                        return False

        return True

    # Check whether the block corresponding to the action is valid.
    def _is_valid_block(self, action) -> bool:
        if self._block_exist[action // 81]:
            return True
        else:
            return False

    # If there is a block to destroy, destroy it and get the corresponding reward.
    def _crash_block(self, action) -> int:
        # combo : Number of broken blocks in one action
        combo = 0
        dup_check = np.zeros((9, 9))
        rows = []
        cols = []
        square33 = []

        block, _ = self.action_to_blk_pos(action)

        # check row
        for r in range(BOARD_LENGTH):
            if self._board[r, :].sum() == 9:
                if dup_check[r, :].sum() != 9:
                    combo += 1
                    dup_check[r, :] = 1
                    rows.append(r)
        # check col
        for c in range(BOARD_LENGTH):
            if self._board[:, c].sum() == 9:
                if dup_check[:, c].sum() != 9:
                    combo += 1
                    dup_check[:, c] = 1
                    cols.append(c)
        # check square33
        for r in range(0, BOARD_LENGTH, 3):
            for c in range(0, BOARD_LENGTH, 3):
                if self._board[r:r+3, c:c+3].sum() == 9:
                    if dup_check[r:r+3, c:c+3].sum() != 9:
                        combo += 1
                        dup_check[r:r+3, c:c+3] = 1
                        square33.append((r, c))

        for r in rows:
            self._board[r, :] = 0
        for c in cols:
            self._board[:, c] = 0
        for r, c in square33:
            self._board[r:r+3, c:c+3] = 0

        combo = len(rows) + len(cols) + len(square33)
        if combo > 0:
            self.straight += 1

        if self.reward_mode == 'one':
            if combo == 0:
                return 1
            else:
                return 1 + combo + self.straight
        elif self.reward_mode == 'woodoku':
            if combo == 0:
                return block.sum()
            else:
                return 28 * combo + 10 * self.straight + block.sum() - 20

    def action_to_blk_pos(self, action) -> tuple:
        # First Block
        if 0 <= action and action <= 80:
            block = self._block_1
            location = [action // 9, action % 9]

        # Second Block
        elif 81 <= action and action <= 161:
            block = self._block_2
            location = [(action - 81) // 9, (action - 81) % 9]

        # Third Block
        else:
            block = self._block_3
            location = [(action - 162) // 9, (action - 162) % 9]

        return block, location

    # Gets the position relative to the center of the border of the block.
    def get_block_square(self, block) -> tuple:
        row = []
        col = []

        for r in range(block.shape[0]):
            if block[r, :].sum() > 0:
                row.append(r)
        for c in range(block.shape[1]):
            if block[:, c].sum() > 0:
                col.append(c)
        return (row[0], row[-1], col[0], col[-1])

    def place_block(self, action):
        # c_loc : where the center of the block is placed
        block, c_loc = self.action_to_blk_pos(action)
        r1, r2, c1, c2 = self.get_block_square(block)
        self._board[c_loc[0]+r1-2:c_loc[0]+r2-1, c_loc[1]+c1 - 2: c_loc[1]+c2-1] \
            += block[r1:r2+1, c1:c2+1]

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        terminated = False

        # Checks whether there is a block corresponding to the action,
        #   and returns if there is not.
        if not self._is_valid_block(action):
            return (self._get_obs(), 0, terminated, False, self._get_info())

        # Check if the block can be placed at the location corresponding to the action.
        #   Return if not possible.
        if not self._is_valid_position(action):
            return (self._get_obs(), 0, terminated, False, self._get_info())

        self.place_block(action)

        # If there is a block to destroy, destroy it and get the corresponding reward.
        reward = self._crash_block(action)
        self.score += reward

        # make block zero and _block_exist to False
        self._nonexist_block(action)

        # Check if all 3 blocks have been used.
        # If the block does not exist, a new block is obtained.
        if sum(self._block_exist) == 0:
            self._get_3_blocks_random()

        # Check if the game is terminated.
        terminated = self._is_terminated()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _line_printer(self, line: np.ndarray):
        return np.array2string(line, separator='', formatter={'str_kind': lambda x: x})

    @property
    def block_exist(self):
        return self._block_exist

    @property
    def get_score(self):
        return self.score

    def render(self):
        if self.render_mode == 'ansi':
            display_height = 17
            display_width = 21
            display_score_top = 1

            new_board = np.where(self._board == 1, '■', '□')
            new_block_1 = np.where(self._block_1 == 1, '■', '□')
            new_block_2 = np.where(self._block_2 == 1, '■', '□')
            new_block_3 = np.where(self._block_3 == 1, '■', '□')

            game_display = np.full(
                (display_height, display_width), ' ', dtype='<U1')

            # copy board
            game_display[1:10, 1:10] = new_board

            # copy block
            for i, block in enumerate([new_block_1, new_block_2, new_block_3]):
                game_display[11:16, 7*i+1:7*i+6] = block

            # create score_board
            game_display[display_score_top+1,
                         11:20] = np.array(list('┌'+'─'*7+'┐'))
            game_display[display_score_top+2,
                         11:20] = np.array(list('│'+' SCORE '+'│'))
            game_display[display_score_top+3,
                         11:20] = np.array(list('├'+'─'*7+'┤'))
            game_display[display_score_top+4,
                         11:20] = np.array(list(f'│{self.score:07d}│'))
            game_display[display_score_top+5,
                         11:20] = np.array(list('└'+'─'*7+'┘'))

            # Display game_display
            for i in range(display_height):
                print(self._line_printer(game_display[i])[1:-1])

    def close(self):
        pass
