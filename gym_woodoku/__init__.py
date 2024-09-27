from gymnasium.envs.registration import register

__version__ = "0.0.1"

register(
    id='gym_woodoku/Woodoku-v0',
    entry_point='gym_woodoku.envs:WoodokuEnv',
)
