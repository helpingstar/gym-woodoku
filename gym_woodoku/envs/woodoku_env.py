"""
This code is written to reference board positions based on `r`(row) and `c`(column) instead of `x` and `y`. 
The top left corner is represented as [0, 0], where the first axis indicates `r` and the second axis indicates `c`, 
with indices increasing downwards and to the right, respectively.
"""

from typing import Tuple, Dict
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from numpy.typing import NDArray
from .blocks import blocks

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)


class WoodokuEnv(gym.Env):
    metadata = {
        "game_modes": ["woodoku"],
        "render_modes": ["ansi", "rgb_array", "human"],
        "render_fps": 10,
        "score_modes": ["woodoku"],
    }

    def __init__(self, game_mode="woodoku", render_mode=None, crash33=True, score_mode="woodoku"):
        # ASSERT
        err_msg = f"{game_mode} is not in {self.metadata['game_modes']}"
        assert game_mode in self.metadata["game_modes"], err_msg
        self.game_mode = game_mode

        err_msg = f"{render_mode} is not in {self.metadata['render_modes']}"
        assert render_mode is None or render_mode in self.metadata["render_modes"], err_msg
        self.render_mode = render_mode

        err_msg = f"{score_mode} is not in {self.metadata['score_modes']}"
        assert score_mode in self.metadata["score_modes"], err_msg
        self.score_mode = score_mode

        self.crash33 = crash33
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(9, 9), dtype=np.int8),
                "block_1": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                "block_2": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                "block_3": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
            }
        )

        # action_space : (Block X Width X Height)
        self.action_space = spaces.Discrete(243)

        # get kind of blocks by `game_mode`
        self._block_list = blocks[game_mode]

        # render

        self.window = None
        self.clock = None

        self.window_size = 512  # The size of the PyGame window

        self.MAX_BLOCK_NUM = 3
        self.BLOCK_LENGTH = 5
        self.BOARD_LENGTH = 9

    def _get_3_blocks(self) -> Tuple[NDArray[np.int8], NDArray[np.int8], NDArray[np.int8]]:
        a = self.np_random.choice(range(self._block_list.shape[0]), 3, replace=False)
        self._block_valid_pos = []
        for i in range(3):
            valid_list = []
            for r in range(5):
                for c in range(5):
                    if self._block_list[a[i]][r][c] == 1:
                        valid_list.append((r, c))
            self._block_valid_pos.append(valid_list)

        return (
            self._block_list[a[0]].copy(),
            self._block_list[a[1]].copy(),
            self._block_list[a[2]].copy(),
        )

    def reset(self, seed=None, options=None):
        # reset seed
        super().reset(seed=seed)

        self.step_count = 0

        # score
        self._score = 0

        # make board clean
        self._board = np.zeros((9, 9), dtype=np.int8)

        # get 3 blocks
        self._get_3_blocks_random()
        self._block_exist = [True, True, True]

        # Whether a block can be placed in its place.
        self.legality = np.zeros((243,), dtype=np.int8)
        self._get_legal_actions()

        # get observation and info
        observation = self._get_obs()

        # Shows how many pieces are broken in a row
        # (this is different from how many pieces are broken at once)
        self.straight = 0

        # combo : Number of broken blocks in one action
        self.combo = 0
        self.is_legal = True
        self.n_cell = 0

        if self.render_mode == "human":
            self.render()

        return observation, self._get_info()

    def _get_3_blocks_random(self):
        # randomly select three blocks
        self._block_exist = [True, True, True]
        self._block_1, self._block_2, self._block_3 = self._get_3_blocks()

    def _get_legal_actions(self):
        """
        Checks whether there is a block corresponding to the action
        Check if the block can be placed at the location.
        """
        for action in range(243):
            if self._is_valid_block(action) and self._is_valid_position(action):
                self.legality[action] = 1
            else:
                self.legality[action] = 0

    def _get_obs(self) -> Dict[str, NDArray[np.int8]]:
        return {
            "board": self._board,
            "block_1": self._block_1,
            "block_2": self._block_2,
            "block_3": self._block_3,
        }

    def _get_info(self):
        return {
            "action_mask": self.legality,
            "score": self._score,
            "straight": self.straight,
            "combo": self.combo,
            "is_legal": self.is_legal,
            "n_cell": self.n_cell,
        }

    def _is_terminated(self) -> bool:
        # Check if the game can be continued with the blocks you own
        # If any number of cases can proceed, False is returned.
        for blk_num in range(self.MAX_BLOCK_NUM):
            if self._block_exist[blk_num]:
                for act in range(blk_num * 81, (blk_num + 1) * 81):
                    if self._is_valid_position(act):
                        return False
        return True

    def _nonexist_block(self, action: int):
        # Deactivate the block corresponding to the action and set the array to 0.
        self._block_exist[action // 81] = False
        block, _ = self.action_to_blk_pos(action)
        block.fill(0)

    def _is_valid_position(self, action: int) -> bool:
        block, location = self.action_to_blk_pos(action)
        # board와 block 비교
        for row, col in self._block_valid_pos[action // 81]:
            # location - 2 : leftmost top (0, 0)
            # When the block is located outside the board
            if not (0 <= (location[0] - 2 + row) < 9 and 0 <= (location[1] - 2 + col) < 9):
                return False

            # When there is already another block
            if self._board[location[0] - 2 + row][location[1] - 2 + col] == 1:
                return False

        return True

    # Check whether the block corresponding to the action is valid.
    def _is_valid_block(self, action: int) -> bool:
        if self._block_exist[action // 81]:
            return True
        else:
            return False

    # If there is a block to destroy, destroy it and get the reward.
    def _crash_block(self, action: int) -> int:
        """Checks the board and if there are blocks to break, it verifies duplicates and removes them, then returns the reward value.
        If there are broken blocks, increment the straight value.
        Args:
            action (int): action

        Returns:
            int: reward
        """
        dup_check = np.zeros((9, 9))
        rows = []
        cols = []
        square33 = []

        block, _ = self.action_to_blk_pos(action)

        self.n_cell = block.sum()

        # check row
        for r in range(self.BOARD_LENGTH):
            if self._board[r, :].sum() == 9:
                if dup_check[r, :].sum() != 9:
                    self.combo += 1
                    dup_check[r, :] = 1
                    rows.append(r)
        # check col
        for c in range(self.BOARD_LENGTH):
            if self._board[:, c].sum() == 9:
                if dup_check[:, c].sum() != 9:
                    self.combo += 1
                    dup_check[:, c] = 1
                    cols.append(c)
        # check square33
        for r in range(0, self.BOARD_LENGTH, 3):
            for c in range(0, self.BOARD_LENGTH, 3):
                if self._board[r : r + 3, c : c + 3].sum() == 9:
                    if dup_check[r : r + 3, c : c + 3].sum() != 9:
                        self.combo += 1
                        dup_check[r : r + 3, c : c + 3] = 1
                        square33.append((r, c))

        for r in rows:
            self._board[r, :].fill(0)
        for c in cols:
            self._board[:, c].fill(0)
        for r, c in square33:
            self._board[r : r + 3, c : c + 3].fill(0)

        self.combo = len(rows) + len(cols) + len(square33)
        if self.combo > 0:
            self.straight += 1
        else:
            self.straight = 0

        # return reward
        if self.combo == 0:
            return self.n_cell
        else:
            return 28 * self.combo + 10 * self.straight + self.n_cell - 20

    def _get_score(self):
        if self.score_mode == "woodoku":
            if self.combo == 0:
                return self.n_cell
            else:
                return 28 * self.combo + 10 * self.straight + self.n_cell - 20

    def action_to_blk_pos(self, action: int) -> Tuple[NDArray[np.int8], Tuple[int, int]]:
        """Converts an integer representing the action into the corresponding `_block_*` and the desired placement coordinates, and returns them.
        The placement coordinates refer to the location of the index (2, 2), which is the center of the 5x5 block.

        Args:
            action (int): action, [0, 242]

        Returns:
            Tuple[NDArray[np.int8], Tuple[int, int]]: (corresponding `_block_*`, (r coordinate, c coordinate))
        """
        # First Block
        if 0 <= action <= 80:
            block = self._block_1
            location = (action // 9, action % 9)

        # Second Block
        elif 81 <= action <= 161:
            block = self._block_2
            location = ((action - 81) // 9, (action - 81) % 9)

        # Third Block
        # 162 <= action < 243
        else:
            block = self._block_3
            location = ((action - 162) // 9, (action - 162) % 9)

        return block, location

    def get_block_square(self, block: NDArray[np.int8]) -> Tuple[int, int, int, int]:
        """Creates a rectangle that encloses only the actual blocks within a 5x5 grid,
        and returns the relative edge coordinates to the center.

        Args:
            block (NDArray[np.int8]): An array representing the block shape with a size of 5x5.

        Returns:
            Tuple[int, int, int, int]: (r_min, r_max, c_min, c_max),
            The position of the corner enclosing the block in the 5x5 array representing the block.
            The rows of the actual block are in the range [r_min, r_max] and the columns are in the range [c_min, c_max].
            In r and c, index 0 represents the top, leftmost corner.
        """

        r_min = 6
        r_max = -1
        c_min = 6
        c_max = -1

        # `.sum()` is faster than `.any()`
        for r in range(block.shape[0]):
            if block[r, :].sum() > 0:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
        for c in range(block.shape[1]):
            if block[:, c].sum() > 0:
                c_min = min(c_min, c)
                c_max = max(c_max, c)

        return (r_min, r_max, c_min, c_max)

    def place_block(self, action: int):
        """Finds the block and its position corresponding to the action, and places the block at that location.

        Args:
            action (int): action
        """
        # c_loc : where the center of the block is placed
        block, c_loc = self.action_to_blk_pos(action)
        r_min, r_max, c_min, c_max = self.get_block_square(block)
        self._board[c_loc[0] - 2 + r_min : c_loc[0] + r_max - 1, c_loc[1] + c_min - 2 : c_loc[1] + c_max - 1] += block[
            r_min : r_max + 1, c_min : c_max + 1
        ]

    def step(
        self, action: int
    ) -> Tuple[Dict[str, NDArray[np.int8]], int, bool, bool, Dict[str, NDArray[np.int8] | int | bool]]:
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg

        terminated = False

        self.combo = 0
        self.n_cell = 0

        # invalid action, attempting to place a block where a block is already placed
        if not self.legality[action]:
            self.is_legal = False
            reward = 0
            terminated = False
        else:
            self.is_legal = True
            self.place_block(action)

            # If there is a block to destroy, destroy it and get the corresponding reward.
            reward = self._crash_block(action)
            self._score += self._get_score()

            # make block zero and _block_exist to False
            self._nonexist_block(action)

            # Check if all 3 blocks have been used.
            # If the block does not exist, a new block is obtained.
            if sum(self._block_exist) == 0:
                self._get_3_blocks_random()

            # Check if the game is terminated.
            terminated = self._is_terminated()

            self._get_legal_actions()

        self.step_count += 1
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _line_printer(self, line: NDArray) -> str:
        return np.array2string(line, separator="", formatter={"str_kind": lambda x: x})

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        if self.render_mode == "ansi":
            self._render_text()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_text(self):
        display_height = 17
        display_width = 21
        display_score_top = 1
        new_board = np.where(self._board == 1, "■", "□")
        new_block_1 = np.where(self._block_1 == 1, "■", "□")
        new_block_2 = np.where(self._block_2 == 1, "■", "□")
        new_block_3 = np.where(self._block_3 == 1, "■", "□")
        game_display = np.full((display_height, display_width), " ", dtype="<U1")

        # copy board
        game_display[1:10, 1:10] = new_board

        # copy block
        for i, block in enumerate([new_block_1, new_block_2, new_block_3]):
            game_display[11:16, 7 * i + 1 : 7 * i + 6] = block

        # create score_board
        game_display[display_score_top + 1, 11:20] = np.array(list("┌" + "─" * 7 + "┐"))
        game_display[display_score_top + 2, 11:20] = np.array(list("│" + " SCORE " + "│"))
        game_display[display_score_top + 3, 11:20] = np.array(list("├" + "─" * 7 + "┤"))
        game_display[display_score_top + 4, 11:20] = np.array(list(f"│{self._score:07d}│"))
        game_display[display_score_top + 5, 11:20] = np.array(list("└" + "─" * 7 + "┘"))

        # Display game_display
        for i in range(display_height):
            print(self._line_printer(game_display[i])[1:-1])

    def _render_gui(self, mode: str):
        pygame.font.init()
        if self.window is None:
            pygame.init()
            # render
            self.board_square_size = 32
            self.block_square_size = 24

            board_total_size = self.board_square_size * 9
            block_total_size = self.block_square_size * 5

            board_left_margin = (self.window_size - board_total_size) // 2
            block_left_margin = (self.window_size - block_total_size * 3) // 4

            top_margin = (self.window_size - board_total_size - block_total_size) // 3

            # Initialize the positions of the squares on the board.
            self.board_row_pos = np.zeros(self.BOARD_LENGTH, dtype=np.uint32)
            self.board_col_pos = np.zeros(self.BOARD_LENGTH, dtype=np.uint32)

            for i in range(self.BOARD_LENGTH):
                self.board_col_pos[i] = board_left_margin + self.board_square_size * i
                self.board_row_pos[i] = top_margin + self.board_square_size * i

            # Initializes the position of the square in the block.
            self.block_row_pos = np.zeros(self.BLOCK_LENGTH, dtype=np.uint32)
            self.block_col_pos = np.zeros((self.MAX_BLOCK_NUM, self.BLOCK_LENGTH), dtype=np.uint32)

            for i in range(self.BLOCK_LENGTH):
                self.block_row_pos[i] = self.window_size - top_margin - block_total_size + self.block_square_size * i

            for b in range(self.MAX_BLOCK_NUM):
                for i in range(self.BLOCK_LENGTH):
                    self.block_col_pos[b][i] = (
                        block_left_margin + (block_left_margin + block_total_size) * b + self.block_square_size * i
                    )

            if mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            elif mode == "rgb_array":
                self.window = pygame.Surface((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill(WHITE)

        # draw board square
        for r in range(self.BOARD_LENGTH):
            for c in range(self.BOARD_LENGTH):
                if self._board[r][c] == 1:
                    pygame.draw.rect(
                        canvas,
                        BLACK,
                        pygame.Rect(
                            (self.board_col_pos[c], self.board_row_pos[r]),  # pos
                            (self.board_square_size, self.board_square_size),
                        ),
                    )
                pygame.draw.rect(
                    canvas,
                    GRAY,
                    pygame.Rect(
                        (self.board_col_pos[c], self.board_row_pos[r]),  # pos
                        (self.board_square_size, self.board_square_size),
                    ),
                    2,
                )

        # draw block square
        for idx, block in enumerate([self._block_1, self._block_2, self._block_3]):
            for r in range(self.BLOCK_LENGTH):
                for c in range(self.BLOCK_LENGTH):
                    if block[r][c] == 1:
                        pygame.draw.rect(
                            canvas,
                            BLACK,
                            pygame.Rect(
                                (
                                    self.block_col_pos[idx][c],
                                    self.block_row_pos[r],
                                ),  # pos
                                (self.block_square_size, self.block_square_size),
                            ),
                        )
                    pygame.draw.rect(
                        canvas,
                        GRAY,
                        pygame.Rect(
                            (self.block_col_pos[idx][c], self.block_row_pos[r]),  # pos
                            (self.block_square_size, self.block_square_size),
                        ),
                        2,
                    )

        myFont = pygame.font.SysFont(None, 30)
        num = myFont.render(f"step: {self.step_count}", True, (0, 0, 0))
        score = myFont.render(f"score: {self._score}", True, (0, 0, 0))
        straight = myFont.render(f"straight: {self.straight}", True, (0, 0, 0))
        canvas.blit(score, (10, 5))
        canvas.blit(num, (200, 5))
        canvas.blit(straight, (340, 5))

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
