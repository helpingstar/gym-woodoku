# Installation

```bash
git clone https://github.com/helpingstar/gym-woodoku.git
cd gym-woodoku
pip install -e .
```

## Colab

!git clone https://github.com/helpingstar/gym-woodoku.git
%cd gym-woodoku
!pip install -e .

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
