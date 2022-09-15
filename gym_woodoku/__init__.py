from gym.envs.registration import register

register(
    id='woodoku-v0',
    entry_point='gym_woodoku.envs:WoodokuEnv',
)
