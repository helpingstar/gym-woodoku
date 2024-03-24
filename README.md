## Welcome to gym-woodoku!

https://github.com/helpingstar/gym-woodoku/assets/54899900/57d6b2fb-f150-4769-b80f-c4115cc4e64a

## Installation

```bash
git clone https://github.com/helpingstar/gym-woodoku.git
cd gym-woodoku
pip install -e .
```

## Usage

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

## Observation

```
Dict({
  "board": Box(low=0, high=1, shape=(9, 9), dtype=np.int8),
  "block_1": Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
  "block_2": Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
  "block_3": Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
})
```

**Location of Blocks in Array :**

Wrap the block in a rectangle.
* If side length is **odd** : `len(side) // 2`
* If side length is **even** : `len(side) // 2 - 1`

Each point is placed at the index `(2,2)` of the size array `(5,5)`. (zero based)

**example)**

<image src = "https://user-images.githubusercontent.com/76930260/202097501-704866dc-927e-490a-9664-e2397c46dc93.png" height = "100" weight = "100"/>         <image src = "https://user-images.githubusercontent.com/76930260/202097633-e8438eba-d080-421b-8786-b081962e9c13.png" height = "100" weight = "100"/>


## action

action_space : `Discrete(243)`

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



## Arguments

```python
import gym_woodoku
import gymnasium as gym

env = gym.make('gym_woodoku/Woodoku-v0', 
                game_mode: str = 'woodoku',
                render_mode: str = None,
                crash33: bool = True)
```

* `game_mode` : Gets the type of block to use in the game.
  * `woodoku`
* `crash33` : If true, when a 3x3 cell is filled, that portion will be broken.
* `render_modes` : Determines gym rendering method.
  * `ansi` : The game screen appears on the console.
  * `human` : continuously rendered in the current display
  * `rgb_array` : return a single frame representing the current state of the environment. Use with RecordVideo Wrapper if you want to save episodes.
