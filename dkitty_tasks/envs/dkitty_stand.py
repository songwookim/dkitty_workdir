__credits__ = ["Rushiv Arora"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DKITTY_ASSET_PATH = '~/Desktop/git_reposits/robel_sim/dkitty/dkitty_stand-v0.xml'
FRAME_SKIP = 40
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


# robot_infot = dict(
#     name='dkitty',
#     actuator_indices=range(12],
#     qpos_indices=range(6, 18),
#     qpos_range=[
#         # FR
#         (-0.5, 0.279),
#         (0.0, PI / 2),
#         (-2.0, 0.0),
#         # FL
#         (-0.279, 0.5),
#         (0.0, PI / 2),
#         (-2.0, 0.0),
#         # BL
#         (-0.279, 0.5),
#         (0.0, PI / 2),
#         (-2.0, 0.0),
#         # BR
#         (-0.5, 0.279),
#         (0.0, PI / 2),
#         (-2.0, 0.0),
#     ],
#     qvel_range=[(-PI, PI)] * 12,
# )

class DkittyStandEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 10,
    }

    def __init__(
            self,
            forward_reward_weight=1.0,
            ctrl_cost_weight=0.1,
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(35,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            DKITTY_ASSET_PATH,
            FRAME_SKIP,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=noise_low, high=noise_high, size=self.model.nq
        # )
        qpos = np.array(
            [0, 0, 0, 0, 0, 0, 0, np.pi / 4, -np.pi / 2, 0, np.pi / 4, -np.pi / 2, 0, np.pi / 4, -np.pi / 2, 0,
             np.pi / 4, -np.pi / 2])
        # qpos[0, 3, 6, 9] = 0
        # qpos[1, 4, 7, 10] = np.pi / 4
        # qpos[2, 5, 8, 11] = -np.pi / 2
        qvel = (
                self.init_qvel
                + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

# class DKittyStandFixed(BaseDKittyStand):
#         """Stand up from a fixed position."""
#
#         def _reset(self):
#             """Resets the environment."""
#             self._initial_pose[[0, 3, 6, 9]] = 0
#             self._initial_pose[[1, 4, 7, 10]] = np.pi / 4
#             self._initial_pose[[2, 5, 8, 11]] = -np.pi / 2
#             super()._reset()
