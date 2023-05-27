# Welcome to gym-woodoku!

![woodoku_ex_video](https://user-images.githubusercontent.com/54899900/202888885-bd8d18eb-68aa-4dd0-963e-61f311716296.gif)

# Installation

```bash
git clone https://github.com/helpingstar/gym-woodoku.git
cd gym-woodoku
pip install -r requirements.txt
pip install -e .
```

## Colab

```bash
!git clone https://github.com/helpingstar/gym-woodoku.git
%cd gym-woodoku
!pip install -r requirements.txt
!pip install -e .
```

# Usage

```python
import gym_woodoku
import gymnasium as gym

env = gym.make('gym_woodoku/Woodoku-v0', game_mode='woodoku', render_mode='human')
# env = gym.wrappers.RecordVideo(env, video_folder='./video_folder')

observation, info = env.reset()
for i in range(100000):
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    if terminated:
        env.reset()
env.close()
```

# environment

## `state`

### `board`

9 X 9 `numpy.array`, `dtype=uint8`

<img src = "https://user-images.githubusercontent.com/76930260/202095781-ca987cc9-0233-4aa6-a754-e81ad35b1587.png" width = "200" height = "200"/>



### `block`

5 X 5 `numpy.array`, `dtype=uint8`

<image src = "https://user-images.githubusercontent.com/76930260/202096150-8d654bf0-ba08-428e-9367-760eff35756c.png" height = "100" weight = "100"/>

**Location of Blocks in Array :**

Wrap the block in a rectangle.
* If side length is **odd** : `len(side) // 2`
* If side length is **even** : `len(side) // 2 - 1`

Each point is placed at the index `(2,2)` of the size array `(5,5)`. (zero based)

**example)**

<image src = "https://user-images.githubusercontent.com/76930260/202097501-704866dc-927e-490a-9664-e2397c46dc93.png" height = "100" weight = "100"/>         <image src = "https://user-images.githubusercontent.com/76930260/202097633-e8438eba-d080-421b-8786-b081962e9c13.png" height = "100" weight = "100"/>


## action

`action_space` : `Discrete(243)`

* 0~80 : use `block_1`
  * Place `block_1` at ((action-0) // 9, (action-0) % 9)
* 81~161 : use `block_2`
  * Place `block_2` at ((action-81) // 9, (action-81) % 9)
* 162~242 : use `block_3`
  * Place `block_3` at ((action-162) // 9, (action-162) % 9)

**example**
1. `action == 0`
    * Take `block_1` and place it at position `(0, 0)` based on the center of the block.

<img src="https://user-images.githubusercontent.com/54899900/202887249-b4ed5f56-5bd7-4c4e-a0c3-b20132d417d2.jpg" width="150" height="150"/>

2. `action == 212`
    * Take `block_3` and place it at position `(5, 5)` based on the center of the block.

<img src="https://user-images.githubusercontent.com/54899900/202888125-affd44e6-d2ef-4103-a336-d2402366386a.jpg" width="150" height="150"/>

## `metadata`

### `game_mode`
Gets the type of block to use in the game.
* `woodoku`

### `obs_mode`
Decide how to obtain the observation.

* `divided` : 1 board and 3 blocks
  * `observation_space` : `Dict({"board": MultiBinary([9, 9]), "block_(1/2/3)": MultiBinary([5, 5])})`
* `total_square`
  * `observation_space` : `MultiBinary([15, 15, 1])`

<img src="https://user-images.githubusercontent.com/54899900/202887172-c62ad9ee-673c-41dc-9b68-e00d32cf8854.jpg" width="250" height="250"/>

### `reward_modes`
Determine the scoring method.
* `one` : Get a reward of 1 for each block destroyed.
  * 1 + combo + straight
* `woodoku` : Follow Woodoku's scoring system.
  * 28 * combo + 10 * straight + (number of square) - 20

combo : Number of broken pieces at once

straight : broken turns in a row

### `render_modes`
Determines gym rendering method.
* `ansi` : The game screen appears on the console.
* `human` : continuously rendered in the current display
* `rgb_array` : return a single frame representing the current state of the environment. Use with RecordVideo Wrapper if you want to save episodes.
