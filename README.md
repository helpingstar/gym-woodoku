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

env = gym.make('gym_woodoku/Woodoku-v0')
print(env.action_space)
print(env.observation_space)
observation, info = env.reset()

print(observation)
print(info)

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
Integers ranging from `0` to `242`

* `0~80` : use `block_1`
  * Place `block_1` at (`action`-`0` // 9, `action`-`0` % 9)
* `81~161` : use `block_2`
  * Place `block_2` at (`action`-`81` // 9, `action`-`81` % 9)
* `162~242` : use `block_3`
  * Place `block_3` at (`action`-`162` // 9, `action`-`162` % 9)
