from gymnasium.envs.registration import register

register(
    id='gym_woodoku/Woodoku-v0',
    entry_point='gym_woodoku.envs:WoodokuEnv',
)
