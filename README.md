# Installation

```bash
git clone https://github.com/helpingstar/gym-woodoku.git
cd gym-woodoku
pip install -e .
```

## Colab

```bash
!git clone https://github.com/helpingstar/gym-woodoku.git
%cd gym-woodoku
!pip install -e .
```

# Usage

```python
import gym_woodoku
import gym

env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku')
observation, info = env.reset()
env.render()
for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
```

# environment

## `state`

### `board`

9 X 9 `numpy.array`, `dtype=uint8`

// TODO: picture

### `block`

5 X 5 `numpy.array`, `dtype=uint8`

// TODO: picture

**Location of Blocks in Array :**

Wrap the block in a rectangle.
* If side length is **odd** : `len(side) // 2`
* If side length is **even** : `len(side) // 2 - 1`

Each point is placed at the index `(2,2)` of the size array `(5,5)`. (zero based)

**example)**

// TODO

## `action`
Integers ranging from 0 to 242

* 0~80 : use `block_1`
  * Place `block_1` at (`action`-0 // 9, `action`-0 % 9)
* 81~161 : use `block_2`
  * Place `block_2` at (`action`-81 // 9, `action`-81 % 9)
* 162~242 : use `block_3`
  * Place `block_3` at (`action`-162 // 9, `action`-162 % 9)

## `metadata`

### `game_mode`
Gets the type of block to use in the game.
* `woodoku`

### `obs_mode`
Decide how to obtain the observation.

* `divided` : 1 board and 3 blocks
  * `{"board": (9 X 9), block_(1/2/3): (5 X 5)}`
* `total_square` : (15 X 15)

### `reward_modes`
Determine the scoring method.
* `one` : Get a reward of 1 for each block destroyed.
* `woodoku` : Follow Woodoku's scoring system.

### `render_modes`
Determines gym rendering method.
* `console` : The game screen appears on the console.
* `plot` : // TODO
* `pygame` :  Render through the pygame library.
