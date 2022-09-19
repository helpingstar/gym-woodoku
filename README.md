# Installation

`gym <= 0.21`


```bash
cd gym-woodoku
pip install -e .
```

# Usage

```python
import gym

print(gym.__version__)

env = gym.make('gym_woodoku:woodoku-v0')
env.reset()
print(env.step(1))
```
