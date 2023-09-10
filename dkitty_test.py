import gymnasium as gym
import dkitty_tasks
from stable_baselines3.common.env_checker import check_env
from dkitty_tasks.envs import DkittyStandEnv
from stable_baselines3 import SAC

import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np


# env = gym.make('dkitty_tasks/HalfCheetah-song')
env = DkittyStandEnv(render_mode="human")
check_env(env)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#model = SAC("MlpPolicy", env, verbose=1, action_noise=action_noise)
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000, log_interval=1, progress_bar=True)
model.save("ddpg_pendulum")

observation, info = env.reset(seed=42)
# for _ in range(1000):
#     action, _states = model.predict(observation, deterministic=True)
#     observation, reward, terminated, truncated, info = env.step(action)
#
#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()