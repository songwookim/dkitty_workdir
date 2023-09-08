import gymnasium as gym
import dkitty_tasks
from stable_baselines3.common.env_checker import check_env
from dkitty_tasks.envs import DkittyStandEnv

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np


# env = gym.make('dkitty_tasks/HalfCheetah-song')
env = DkittyStandEnv(render_mode="human")
check_env(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG("MlpPolicy", env, verbose=1, action_noise=action_noise)
model.learn(total_timesteps=10000, log_interval=10)
model.save("ddpg_pendulum")

observation, info = env.reset(seed=42)
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

