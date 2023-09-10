__credits__ = ["Rushiv Arora"]

import math

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import mujoco
DKITTY_ASSET_PATH = '~/Desktop/git_reposits/robel_sim/dkitty/dkitty_stand-v0.xml'
FRAME_SKIP = 40
DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
_desire_pose = np.zeros(12,)
_initial_pose = np.zeros(12,)

_initial_pose[0] = -1.5 # 10
_initial_pose[3] = 1.5 # 20
_initial_pose[6] = 1.5 # 30
_initial_pose[9] = -1.5 # 40


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
        self.time_step = 0
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
                low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64     # 35
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(36,), dtype=np.float64     # 35
            )

        MujocoEnv.__init__(
            self,
            DKITTY_ASSET_PATH,
            FRAME_SKIP,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
    def _initialize_simulation(self) :  ############override###################
        model = mujoco.MjModel.from_xml_path(self.fullpath)
        # MjrContext will copy model.vis.global_.off* to con.off*
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)

        data.qpos[6:18] = np.ones(12,)#_initial_pose

        return model, data
    def _reset_simulation(self):
        model, data = self._initialize_simulation()
        mujoco.mj_resetData(model, data)
        # self.sim.reset()
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        self.time_step += 1

        # x_position_before = self.data.qpos[0]
        action = np.clip(action, -1.0, 1.0)

        self.do_simulation(action, self.frame_skip)

        #ctrl_cost = self.control_cost(action) # ???
        # forward_reward = self._forward_reward_weight * x_velocity # ????

        observation = self._get_obs()

        # upright of rewards
        upright = self.data.qpos[2]
        upright_reward = 2  # alpha
        upright_threshold = -0.02 # beta
        falling_reward = -100*(upright<-2)

        upright_rewards = falling_reward + upright_reward * (upright - upright_threshold) / (1 - upright_threshold)

        # term 1 mean_pose_error
        e_pose = _desire_pose-self.data.qpos[6:18] # 12's euler angle
        pose_mean_error = np.abs(e_pose).mean() # = e_pose_bar at paper

        # term 2 center_distance_cost
        center_dist = np.linalg.norm(self.data.qpos[:2])
        center_distance_cost = center_dist

        # term 3 bonus_small
        bonus_small = 5 * (pose_mean_error < (np.pi / 6)) * upright

        # term 4 bonus_big
        bonus_big = 10 * (pose_mean_error < (np.pi / 12)) * (upright > 0.9)

        # reward = upright_rewards - 4 * pose_mean_error - 2 * center_distance_cost + bonus_small + bonus_big
        c1 = 3
        c2 = 0.3
        c3 = 5
        reward = math.exp(c1 * upright) \
                - c2 * math.exp(self.data.qvel[0] * self.data.qvel[0]) \
                - c2 * math.exp(self.data.qvel[1] * self.data.qvel[1]) \
                - c2 * math.exp(self.data.qvel[6] * self.data.qvel[6]) \
                - c2 * math.exp(self.data.qvel[7] * self.data.qvel[7]) \
                - c2 * math.exp(self.data.qvel[8] * self.data.qvel[8]) \
                - c2 * math.exp(self.data.qvel[9] * self.data.qvel[9]) \
                - c2 * math.exp(self.data.qvel[10] * self.data.qvel[10]) \
                - c2 * math.exp(self.data.qvel[11] * self.data.qvel[11]) \
                - c2 * math.exp(self.data.qvel[12] * self.data.qvel[12]) \
                - c2 * math.exp(self.data.qvel[13] * self.data.qvel[13]) \
                - c2 * math.exp(self.data.qvel[14] * self.data.qvel[14]) \
                - c2 * math.exp(self.data.qvel[15] * self.data.qvel[15]) \
                - c2 * math.exp(self.data.qvel[16] * self.data.qvel[16]) \
                - c2 * math.exp(self.data.qvel[17] * self.data.qvel[17]) \
                 - c2 * math.exp(self.data.qpos[0] * self.data.qpos[0]) \
                 - c2 * math.exp(self.data.qpos[1] * self.data.qpos[1]) \
                 - c2 * math.exp(self.data.qpos[6] * self.data.qpos[6]) \
                 - c2 * math.exp(self.data.qpos[7] * self.data.qpos[7]) \
                 - c2 * math.exp(self.data.qpos[8] * self.data.qpos[8]) \
                 - c2 * math.exp(self.data.qpos[9] * self.data.qpos[9]) \
                 - c2 * math.exp(self.data.qpos[10] * self.data.qpos[10]) \
                 - c2 * math.exp(self.data.qpos[11] * self.data.qpos[11]) \
                 - c2 * math.exp(self.data.qpos[12] * self.data.qpos[12]) \
                 - c2 * math.exp(self.data.qpos[13] * self.data.qpos[13]) \
                 - c2 * math.exp(self.data.qpos[14] * self.data.qpos[14]) \
                 - c2 * math.exp(self.data.qpos[15] * self.data.qpos[15]) \
                 - c2 * math.exp(self.data.qpos[16] * self.data.qpos[16]) \
                 - c2 * math.exp(self.data.qpos[17] * self.data.qpos[17]) \
                 - c3 * np.linalg.norm(self.data.qpos[:2])
            # print(reward)
        # reward = forward_reward - ctrl_cost
        terminated = False

        if upright < -0.28 or self.time_step > 1000:
            terminated = True
            self.time_step = 0

        info = {
            #"x_position": x_position_after,
            #"x_velocity": x_velocity,
            "reward_run": reward,
            #"reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        # if self._exclude_current_positions_from_observation:
        #     position = position[1:]

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
