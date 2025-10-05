# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Booster T1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.g1_12dof import base as g1_base
from mujoco_playground._src.locomotion.g1_12dof import g1_constants as consts



def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      obs_history_length=1,
      priv_obs_history_length=1,
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=1.0,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=0.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              survival=0.25,
              tracking_lin_vel=0.0,#1.2,
              tracking_lin_vel_x=2.0,
              tracking_lin_vel_y=1.0,
              tracking_ang_vel=2.0,  # original 0.5
              base_height=0.2,
              orientation=-5.0,
              torques=0.0,#-2.0e-4 / 2,  # original -2.0e-4
              torque_tiredness=0.0,#-1.0e-2 / 2,  # original -1.0e-2
              power=0.0,#-2.0e-3 / 2,  # original -2.0e-3
              lin_vel_z=0.0,#-2.0,
              ang_vel_xy=0.0,#-0.2,
              dof_vel=0.0,#-1.0e-4,
              dof_acc=0.0,#-1.0e-7,
              root_acc=0.0,#-1.0e-4,
              action_rate=0.0,#-1.0 / 2,  # original -1.0
              dof_pos_limits=0.0,#-1.0,
              feet_slip=-0.1 * 10,
              feet_vel_z=0.0,#-1.0,     # disabled in Isaac config
              feet_yaw_diff=-1.0,
              feet_yaw_mean=-1.0,
              feet_roll=-0.1 * 10.0,  # original -0.1
              feet_distance=-1.0 * 10.0,  # original -1.0
              feet_swing=3.0,
              feet_height=-20,
              joint_deviation_hip=-1.0,

          ),
          tracking_sigma = 0.1,
          base_height_target = 0.5,
          swing_period = 0.6,
          max_foot_height=0.08,

      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 1.0],
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.8, 0.8],
      ang_vel_yaw=[-1.0, 1.0],
  )

class G1Utils:
    """Utility functions for the Go1."""

    """
    Properties
    """

    # constant: the lower joint angle limits for a leg (G1, 12-DoF)
    LOWER_JOINT_LIMITS = jp.array([
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,   # left leg
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618,   # right leg
    ])

    UPPER_JOINT_LIMITS = jp.array([
        2.8798,  2.9671,  2.7576,  2.8798,    0.5236,   0.2618,   # left leg
        2.8798,  2.9671,  2.7576,  2.8798,    0.5236,   0.2618,   # right leg
    ])

    """constant: the upper joint angle limits for a leg"""


class Joystick(g1_base.G1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

    self._gravity = jp.s_[0:3]
    self._rpy_rate_idxs = jp.s_[3:6]
    self._roll_rate_idx = jp.s_[3]
    self._pitch_rate_idx = jp.s_[4]
    self._turn_rate_idx = jp.s_[5]
    self._command_idxs = jp.s_[6:9]
    self._cos_phase_idx = jp.s_[9]
    self._sin_phase_idx = jp.s_[10]
    self._q_idxs = jp.s_[11:23]
    self._qd_idxs = jp.s_[23:35]
    self._last_action_idxs = jp.s_[35:47]

    self._priv_rpy_rate_idxs = jp.s_[47:50]
    self._priv_roll_rate_idx = jp.s_[47]
    self._priv_pitch_rate_idx = jp.s_[48]
    self._priv_turn_rate_idx = jp.s_[49]
    self._priv_gravity = jp.s_[50:53]
    self._priv_quat_idxs = jp.s_[53:57]
    self._priv_base_vel_idxs = jp.s_[57:60]
    self._forward_vel_idx = jp.s_[57]
    self._y_vel_idx = jp.s_[58]
    self._z_vel_idx = jp.s_[59]
    self._priv_q_idxs = jp.s_[60:72]
    self._priv_qd_idxs = jp.s_[72:84]
    self._priv_h_idx = jp.s_[84]
    #self._priv_torque_idxs = jp.s_[85:97]
    #self._priv_feet_vel_idx = jp.s_[97:103]
    self._priv_gait_process_idx = jp.s_[85]
    self._priv_gait_frequency_idx = jp.s_[86]
    self.privileged_state_size = 87
    self.obs_limits = jp.ones((self.privileged_state_size, 2))
    self.obs_limits = self.obs_limits.at[:, 0].set(-1.)

    self.obs_limits = self.obs_limits.at[self._q_idxs, 0].set(
        G1Utils.LOWER_JOINT_LIMITS - 0.25)
    self.obs_limits = self.obs_limits.at[self._q_idxs, 1].set(
        G1Utils.UPPER_JOINT_LIMITS + 0.25)
    self.obs_limits = self.obs_limits.at[self._priv_q_idxs, 0].set(
        G1Utils.LOWER_JOINT_LIMITS - 0.25)
    self.obs_limits = self.obs_limits.at[self._priv_q_idxs, 1].set(
        G1Utils.UPPER_JOINT_LIMITS + 0.25)

    self.obs_limits = self.obs_limits.at[self._forward_vel_idx, 0].set(-0.2)
    self.obs_limits = self.obs_limits.at[self._forward_vel_idx, 1].set(2.5)
    self.obs_limits = self.obs_limits.at[self._y_vel_idx, 0].set(-0.5)
    self.obs_limits = self.obs_limits.at[self._y_vel_idx, 1].set(0.5)
    self.obs_limits = self.obs_limits.at[self._z_vel_idx, 0].set(-0.5)
    self.obs_limits = self.obs_limits.at[self._z_vel_idx, 1].set(0.5)
    self.obs_limits = self.obs_limits.at[self._roll_rate_idx, 0].set(-1.5)
    self.obs_limits = self.obs_limits.at[self._roll_rate_idx, 1].set(1.5)
    self.obs_limits = self.obs_limits.at[self._pitch_rate_idx, 0].set(-1.5)
    self.obs_limits = self.obs_limits.at[self._pitch_rate_idx, 1].set(1.5)
    self.obs_limits = self.obs_limits.at[self._turn_rate_idx, 0].set(-1.5)
    self.obs_limits = self.obs_limits.at[self._turn_rate_idx, 1].set(1.5)
    self.obs_limits = self.obs_limits.at[self._priv_roll_rate_idx, 0].set(-1.5)
    self.obs_limits = self.obs_limits.at[self._priv_roll_rate_idx, 1].set(1.5)
    self.obs_limits = self.obs_limits.at[self._priv_pitch_rate_idx, 0].set(-1.5)
    self.obs_limits = self.obs_limits.at[self._priv_pitch_rate_idx, 1].set(1.5)
    self.obs_limits = self.obs_limits.at[self._priv_turn_rate_idx, 0].set(-1.5)
    self.obs_limits = self.obs_limits.at[self._priv_turn_rate_idx, 1].set(1.5)
    self.obs_limits = self.obs_limits.at[self._qd_idxs, 0].set(-20.0)
    self.obs_limits = self.obs_limits.at[self._qd_idxs, 1].set(20.0)
    self.obs_limits = self.obs_limits.at[self._priv_qd_idxs, 0].set(-20.0)
    self.obs_limits = self.obs_limits.at[self._priv_qd_idxs, 1].set(20.0)
    self.obs_limits = self.obs_limits.at[self._cos_phase_idx, 0].set(-1.0)
    self.obs_limits = self.obs_limits.at[self._cos_phase_idx, 1].set(1.0)
    self.obs_limits = self.obs_limits.at[self._sin_phase_idx, 0].set(-1.0)
    self.obs_limits = self.obs_limits.at[self._sin_phase_idx, 1].set(1.0)


    self.obs_limits = self.obs_limits.at[self._priv_h_idx, 0].set(0.0)
    self.obs_limits = self.obs_limits.at[self._priv_h_idx, 1].set(0.8)


  def _normalize_obs(self, obs: jp.ndarray, obs_limits:jp.ndarray) -> jp.ndarray:
      return (2*(obs - obs_limits[:, 0])
              / (obs_limits[:, 1] - obs_limits[:, 0])
              - 1)

  def _denormalize_obs(self, obs: jp.ndarray, obs_limits:jp.ndarray) -> jp.ndarray:
      return ((obs + 1)*(obs_limits[:, 1] - obs_limits[:, 0])/2
              + obs_limits[:, 0])
  
  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(
        self._mj_model.keyframe("knees_bent").qpos[7:]
    )
    
    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    hip_indices = []
    hip_joint_names = ["hip_roll", "hip_yaw"]
    for side in ["left", "right"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    for side in ["left", "right"]:
      knee_indices.append(
          self._mj_model.joint(f"{side}_knee_joint").qposadr - 7
      )
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Left leg.
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # Right leg.
    ])
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._site_id = self._mj_model.site("imu").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._left_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.LEFT_FEET_GEOMS]
    )
    self._right_feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.RIGHT_FEET_GEOMS]
    )
    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    force_range = self._mj_model.actuator_forcerange          # (nact, 2)
    force_limited = self._mj_model.actuator_forcelimited      # (nact,)
    hi = jp.array(force_range[:, 1])
    #   unlimited → treat as very large so the penalty goes to zero
    self._torque_limits = jp.where(force_limited, hi, jp.full_like(hi, 1e6))

  def _reset_if_outside_bounds(self, state: mjx_env.State) -> mjx_env.State:
    qpos = state.data.qpos
    new_x = jp.where(jp.abs(qpos[0]) > 9.5, 0.0, qpos[0])
    new_y = jp.where(jp.abs(qpos[1]) > 9.5, 0.0, qpos[1])
    qpos = qpos.at[0:2].set(jp.array([new_x, new_y]))
    state = state.replace(data=state.data.replace(qpos=qpos))
    return state

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (12,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    # Phase, freq=U(1.25, 1.75)
    rng, key = jax.random.split(rng)
    gait_freq = jax.random.uniform(key, (), minval=1.0, maxval=2.0)
    phase_dt = 2 * jp.pi * self.dt * gait_freq
    phase = jp.array([0, jp.pi])

    gait_process = jp.zeros(())


    rng, cmd_rng = jax.random.split(rng)
    cmd = self.sample_command(cmd_rng)
    do_zero = jp.linalg.norm(cmd) < 0.01
    gait_freq = jp.where(do_zero, jp.asarray(0.0, dtype=gait_freq.dtype), gait_freq)


    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # Phase related.
        "gait_freq": gait_freq,
        "gait_process": gait_process,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
        "filtered_linvel": jp.zeros(3),
        "filtered_angvel": jp.zeros(3),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())
    metrics["vel_err"] = jp.zeros(())
    left_feet_contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._left_feet_geom_id
    ])
    right_feet_contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._right_feet_geom_id
    ])
    contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])

    obs = self._get_obs(data, info, contact)
    obs['state'] = jp.tile(obs['state'], self._config.obs_history_length)
    obs['privileged_state'] = jp.tile(obs['privileged_state'], self._config.priv_obs_history_length)

    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    state = state.replace(data=data)

    # state = self._reset_if_outside_bounds(state)
    action_scale = jp.array([0.54754645, 0.35066146, 0.54754645, 0.35066146, 0.43857732, 0.43857732, 
                    0.54754645, 0.35066146, 0.54754645, 0.35066146, 0.43857732, 0.43857732])
    motor_targets = self._default_pose + action * action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    linvel = self.get_local_linvel(data, "pelvis")
    state.info["filtered_linvel"] = (
        linvel * 1.0 + state.info["filtered_linvel"] * 0.0
    )
    angvel = self.get_gyro(data, "pelvis")
    state.info["filtered_angvel"] = (
        angvel * 1.0 + state.info["filtered_angvel"] * 0.0
    )

    vel_err = jp.sum(jp.square(state.info["command"][:2] - linvel[:2])) 
    state.metrics["vel_err"] = jp.exp(-vel_err)

    left_feet_contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._left_feet_geom_id
    ])
    right_feet_contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._right_feet_geom_id
    ])
    contact = jp.hstack([jp.any(left_feet_contact), jp.any(right_feet_contact)])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1

    state.info["gait_process"] = jp.fmod(state.info["gait_process"] + self.dt * state.info["gait_freq"], 1.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    # state.info["command"] = jp.where(
    #     state.info["step"] > 500,
    #     self.sample_command(cmd_rng),
    #     state.info["command"],
    # )
    # do_zero = jp.all(state.info["command"] == 0)
    # state.info["gait_freq"] = jp.where(do_zero, jp.asarray(0.0, dtype=state.info["gait_freq"].dtype), state.info["gait_freq"])
    
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    obs = self._get_obs(data, state.info, contact)
    obs_stack = jp.concatenate(
      [obs['state'], state.obs['state'][:obs['state'].shape[0]*(self._config.obs_history_length-1)]],
      axis=-1
    )

    priv_obs_stack = jp.concatenate(
      [obs['privileged_state'], state.obs['privileged_state'][:obs['privileged_state'].shape[0]*(self._config.priv_obs_history_length-1)]],
      axis=-1
    )

    obs = {
      'state': obs_stack,
      'privileged_state': priv_obs_stack,
    }

    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

  
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_gravity(data, "pelvis")[-1] < 0.0
    return (
        fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    phase = jp.array([
      jp.cos(2*jp.pi*info["gait_process"]) * (info["gait_freq"] > 1.0e-8).astype(info["gait_process"].dtype) * (jp.linalg.norm(info["command"]) > 0.01).astype(info["gait_process"].dtype),
      jp.sin(2*jp.pi*info["gait_process"]) * (info["gait_freq"] > 1.0e-8).astype(info["gait_process"].dtype) * (jp.linalg.norm(info["command"]) > 0.01).astype(info["gait_process"].dtype)
    ])


    linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )
    # We will disable noisy_linvel
    noisy_linvel = jp.zeros_like(noisy_linvel)

    state = jp.hstack([
        noisy_gravity,  # 3
        noisy_gyro,  # 3
        info["command"],  # 3
        phase, # 2-dimensional 
        noisy_joint_angles,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
    ])
    norm_state = self._normalize_obs(state, self.obs_limits[:state.shape[-1]])
    accelerometer = self.get_accelerometer(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "pelvis")
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]
    quat = data.qpos[3:7]
    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        #accelerometer,  # 3
        gravity,  # 3
        quat, # 4
        linvel,  # 3
        #global_angvel,  # 3
        joint_angles,
        joint_vel,
        root_height,  # 1
        #data.actuator_force,
        #contact,  # 2
        #feet_vel,  # 4*3
        #info["feet_air_time"],  # 2
        info["gait_process"],
        info["gait_freq"],
    ])
    norm_privileged_state = self._normalize_obs(privileged_state, self.obs_limits)

    return {
        "state": norm_state,
        "privileged_state": norm_privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:

    cmd = info["command"]
    lin_f = info["filtered_linvel"]
    ang_f = info["filtered_angvel"]

    return {
        # -------- positive terms --------
        "survival": jp.array(1.0),
        "tracking_lin_vel": self._reward_tracking_lin_vel(cmd, lin_f),
        "tracking_lin_vel_x": self._reward_tracking_lin_vel_axis(0, cmd, lin_f),
        "tracking_lin_vel_y": self._reward_tracking_lin_vel_axis(1, cmd, lin_f),
        "tracking_ang_vel": self._reward_tracking_ang_vel(cmd, ang_f),
        "feet_swing": self._reward_feet_swing(info["gait_process"], info["gait_freq"], contact),
        "feet_height": self._cost_feet_height(data.site_xpos[self._feet_site_id][..., -1], contact, info),
        # -------- penalties ------------ (signed handled by scale)
        "base_height": self._reward_base_height(data),
        "orientation": self._cost_orientation(self.get_gravity(data, "pelvis")),
        "torques": self._cost_torques(data.actuator_force),
        "torque_tiredness": self._cost_torque_tiredness(data.actuator_force),
        "power": self._cost_energy(data.qvel[6:], data.actuator_force),
        "lin_vel_z": self._cost_lin_vel_z(lin_f),
        "ang_vel_xy": self._cost_ang_vel_xy(ang_f),
        "dof_vel": self._cost_dof_vel(data.qvel[6:]),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        "root_acc": self._cost_root_acc(data),
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "feet_vel_z": self._cost_feet_vel_z(data),
        "feet_roll": self._cost_feet_roll(data),
        "feet_yaw_diff": self._cost_feet_yaw_diff(data),
        "feet_yaw_mean": self._cost_feet_yaw_mean(data),
        "feet_distance": self._cost_feet_distance(data, info),
        "joint_deviation_hip": self._cost_joint_deviation_hip(
            data.qpos[7:]
        ),
    }


  # Tracking rewards.
  def _reward_tracking_lin_vel(
      self, command: jax.Array, local_linvel: jax.Array
  ) -> jax.Array:
    """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
    err = jp.sum(jp.square(command[:2] - local_linvel[:2]))
    return jp.exp(-err / self._config.reward_config.tracking_sigma)


  def _reward_tracking_lin_vel_axis(
      self, axis: int, command: jax.Array, local_linvel: jax.Array
  ) -> jax.Array:
    """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
    err = jp.square(command[axis] - local_linvel[axis])
    return jp.exp(-err / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      local_angvel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - local_angvel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
    return jp.square(local_linvel[2])

  def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
    return jp.sum(jp.square(local_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis[:2]))

  def _reward_base_height(self, data: mjx.Data) -> jax.Array:
    h = data.qpos[2]
    return jp.exp(-jp.abs(h - self._config.reward_config.base_height_target)*0.1)

  # Energy related rewards.

  def _cost_torques(self, torques: jp.ndarray) -> jp.ndarray:
    return jp.sum(jp.square(torques))

  def _cost_energy(self, qvel: jp.ndarray, qfrc_actuator: jp.ndarray) -> jp.ndarray:
    power = qvel * qfrc_actuator
    return jp.sum(jp.where(power > 0.0, power, 0.0))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    c1 = jp.sum(jp.square(act - last_act))
    return c1

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qvel))

  # Other rewards.

  def _cost_joint_pos_limits(self, qpos: jp.ndarray) -> jp.ndarray:
    below = qpos < self._soft_lowers
    above = qpos > self._soft_uppers
    return jp.sum((below | above).astype(qpos.dtype))

  def _reward_survival(self) -> jax.Array:
    return jp.array(1.0)



  # Pose-related rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array
  ) -> jax.Array:
    cost = jp.sum(jp.square(qpos[self._hip_indices]))

    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(
        jp.abs(
            qpos[self._knee_indices] - self._default_pose[self._knee_indices]
        )
    )

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose) * self._weights)

  # Feet related rewards.

  def _cost_feet_slip(
      self,
      data: mjx.Data,
      contact: jp.ndarray,
      info: dict[str, Any],
  ) -> jp.ndarray:
    del info  # Unused here.
    v = data.sensordata[self._foot_linvel_sensor_adr]  # shape (2, 3)
    speed2 = jp.sum(jp.square(v), axis=-1)             # per‑foot |v|²
    return jp.sum(speed2 * contact)

  def _cost_feet_height(
      self,
      feet_height: jax.Array,
      contact: jp.ndarray,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = feet_height -  self._config.reward_config.max_foot_height
    return jp.sum(jp.square(error) * ~contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    reward *= cmd_norm > 0.1  # No reward for zero commands.
    return reward

  def _cost_feet_distance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    left_foot_pos = data.site_xpos[self._feet_site_id[0]]
    right_foot_pos = data.site_xpos[self._feet_site_id[1]]
    base_xmat = data.site_xmat[self._site_id]
    base_yaw = jp.arctan2(base_xmat[1, 0], base_xmat[0, 0])
    feet_distance = jp.abs(
        jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
        - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
    )
    return jp.clip(0.2 - feet_distance, min=0.0, max=0.1)

  def sample_command(self, rng: jax.Array) -> jax.Array:
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

    lin_vel_x = jax.random.uniform(
        rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
    )
    lin_vel_y = jax.random.uniform(
        rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
    )
    ang_vel_yaw = jax.random.uniform(
        rng3,
        minval=self._config.ang_vel_yaw[0],
        maxval=self._config.ang_vel_yaw[1],
    )

    # With 10% chance, set everything to zero.
    return jp.where(
        jax.random.bernoulli(rng4, p=0.1),
        jp.zeros(3),
        jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
    )

  def _cost_torque_tiredness(self, torques: jax.Array) -> jax.Array:
    # Σ (τ / τ_max)²  – clipped at 1 so the term stays O(1)
    frac = jp.clip(jp.abs(torques) / self._torque_limits, 0.0, 1.0)
    return jp.sum(jp.square(frac))

  def _cost_root_acc(self, data: mjx.Data) -> jax.Array:
    # Root‑link 6‑D acceleration² (free‑joint entries of qacc)
    return jp.sum(jp.square(data.qacc[:6]))

  # ----- feet kinematics ----------------------------------------------------
  def _feet_site_xmat(self, data: mjx.Data) -> jax.Array:
      """Return the (2,3,3) rotation matrices of the foot *sites*."""
      return data.site_xmat[self._feet_site_id].reshape(2, 3, 3)

  def _feet_roll_yaw(self, data: mjx.Data) -> tuple[jax.Array, jax.Array]:
    """Return (roll, yaw) angles of both feet, radians in [‑π, π]."""
    R = self._feet_site_xmat(data)
    # roll  = atan2(R32, R33)           (x‑rotation)
    # yaw   = atan2(R21, R11)           (z‑rotation)
    roll = jp.arctan2(R[:, 2, 1], R[:, 2, 2])
    yaw  = jp.arctan2(R[:, 1, 0], R[:, 0, 0])
    return roll, yaw

  def _cost_feet_roll(self, data: mjx.Data) -> jax.Array:
    roll, _ = self._feet_roll_yaw(data)
    return jp.sum(jp.square(roll))

  def _cost_feet_yaw_diff(self, data: mjx.Data) -> jax.Array:
    _, yaw = self._feet_roll_yaw(data)
    diff = jp.fmod(yaw[1] - yaw[0] + jp.pi, 2 * jp.pi) - jp.pi
    return jp.square(diff)

  def _cost_feet_yaw_mean(self, data: mjx.Data) -> jax.Array:
    _, feet_yaw = self._feet_roll_yaw(data)
    base_R = data.site_xmat[self._site_id]
    base_yaw = jp.arctan2(base_R[1, 0], base_R[0, 0])
    mean_yaw = jp.mean(feet_yaw)
    err = jp.fmod(base_yaw - mean_yaw + jp.pi, 2 * jp.pi) - jp.pi
    return jp.square(err)

  def _cost_feet_vel_z(self, data: mjx.Data) -> jax.Array:
    # use the same foot linear‑velocity sensors already wired for slip
    vz = data.sensordata[self._foot_linvel_sensor_adr][:, 2]
    return jp.sum(jp.square(vz))

  def _reward_feet_swing(self,
                         gait_process: jp.ndarray,
                         gait_frequency: float,
                         feet_contact: jp.ndarray) -> jp.ndarray:
    left_swing = (jp.abs(gait_process - 0.25) < 0.5 * self._config.reward_config.swing_period) & (gait_frequency > 1.0e-8)
    right_swing = (jp.abs(gait_process - 0.75) < 0.5 * self._config.reward_config.swing_period) & (gait_frequency > 1.0e-8)
    ref_rew = (left_swing & ~feet_contact[0]) + (right_swing & ~feet_contact[1])
    return ref_rew

