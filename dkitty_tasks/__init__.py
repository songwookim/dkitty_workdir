from gymnasium.envs.registration import register

register(
    id="dkitty_tasks/dkitty_stand-v0",
    entry_point="dkitty_tasks.envs:DkittyStandEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)