# Installation

```bash
cd gym-woodoku
pip install -e .
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
