from brax.robots.booster.utils import BoosterUtils
from brax.envs.base import RlwamEnv, State

from brax import actuator
from brax import kinematics
from brax.generalized.base import State as GeneralizedState
from brax.generalized import dynamics
from brax.generalized import integrator
from brax.generalized import mass
from brax import base
from brax.math import rotate, inv_rotate, quat_to_eulerzyx, eulerzyx_to_quat, quat_to_euler, euler_to_quat
from brax.generalized.pipeline import step as pipeline_step

from jax import numpy as jp
from typing import Optional, Any, Tuple, Callable
import jax
import flax
from brax.generalized import constraint
from jax import lax
import numpy as np
from ml_collections import config_dict


@flax.struct.dataclass
class ControlCommand:
    """Output of the low level controller which includes gait control and
    inverse kinematics. """
    q_des: jp.ndarray
    qd_des: jp.ndarray
    Kp: jp.ndarray
    Kd: jp.ndarray


class BoosterGoFast(RlwamEnv):
    """ Booster environment"""

    def __init__(
        self,
        policy_repeat=10,
        forward_cmd_vel_type='constant',  # 'constant' or 'sine'
        forward_cmd_vel_range=(0.0, 0.0),  # for now just using the average of this for the gait controller
        forward_cmd_vel_period_range=(5.0, 10.0),  # only used with 'sine'
        turn_cmd_rate_range=(-jp.pi/8, jp.pi/8),
        initial_yaw_range=(-0.0, 0.0),
        contact_time_const=0.02,
        contact_time_const_range=None,
        contact_damping_ratio=1.0,
        friction_range=(0.6, 0.6),
        ground_roll_range=(0.0, 0.0),
        ground_pitch_range=(0.0, 0.0),
        joint_damping_perc_range=(1.0, 1.0),
        joint_gain_range=(1.0, 1.0),
        link_mass_perc_range=(1.0, 1.0),
        fallen_roll=jp.pi/4,
        fallen_pitch=jp.pi/4,
        forces_in_q_coords=False,
        include_height_in_obs=False,
        body_height_in_action_space=True,
        gains_in_action_space=False,
        backend='generalized',
        used_cached_systems=False,
        healthy_delta_radius=2.0,  # not used
        healthy_delta_yaw=1.57,  # not used
        mini_ankle_dist= 0.04599947574240915,
        tracking_sigma=0.1,
        vel_x_command=1.0,
        **kwargs
    ):

        self.sim_dt = 1/500  # simulation dt; 1000 Hz

        # determines high level policy freq; (1/sim_dt)/policy_repeat Hz
        self.policy_repeat = policy_repeat

        sys = BoosterUtils.get_system(used_cached_systems) # load 'robots/go1/xml/go1.xml'
        sys = sys.replace(dt=self.sim_dt)

        # normally this is use by Brax as the number of times to step the
        # physics pipeline for each environment step. However we have
        # overwritten the pipline_step function with our own behaviour which
        # steps the physics self.policy_repeat times. So we set this to 1.
        n_frames = 1
        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._period = 0.55  # period of the gait cycle (sec)
        self._forward_cmd_vel = jp.mean(jp.array(forward_cmd_vel_range))
        self._initial_yaw_range = initial_yaw_range
        if contact_time_const_range is None:
            self._contact_time_const_range = (contact_time_const,
                                              contact_time_const)
        else:
            self._contact_time_const_range = contact_time_const_range
        self._contact_damping_ratio = contact_damping_ratio
        self._friction_range = friction_range
        self._ground_roll_range = ground_roll_range
        self._ground_pitch_range = ground_pitch_range
        self._joint_damping_perc_range = joint_damping_perc_range
        self._joint_gain_range = joint_gain_range
        self._link_mass_perc_range = link_mass_perc_range
        self._fallen_roll = fallen_roll
        self._fallen_pitch = fallen_pitch
        self._include_height_in_obs = include_height_in_obs
        self._body_height_in_action_space = body_height_in_action_space
        self._gains_in_action_space = gains_in_action_space

        if forces_in_q_coords:
            self._qfc_fn = lambda state, forces: forces
        else:
            self._qfc_fn = lambda state, forces: state.con_jac.T @ forces

        # set up slices for the state space, defined in the xml file
        self._xml_quat_idxs = jp.s_[3:7]
        self._xml_q_idxs = jp.s_[7:19]
        self._xml_base_vel_idxs = jp.s_[0:3]
        self._xml_rpy_rate_idxs = jp.s_[3:6]
        self._xml_qd_idxs = jp.s_[6:18]
        self._xml_h_idxs = jp.s_[2:3]

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

        self._priv_gait_process_idx = jp.s_[85]
        self._priv_gait_frequency_idx = jp.s_[86]


        # set up observation normalization limits
        self._observation_size = 47
        self._priv_observation_size = 87

        self.obs_limits = jp.ones((self._priv_observation_size, 2))
        self.obs_limits = self.obs_limits.at[:, 0].set(-1.)

        self.obs_limits = self.obs_limits.at[self._q_idxs, 0].set(
            BoosterUtils.LOWER_JOINT_LIMITS - 0.25)
        self.obs_limits = self.obs_limits.at[self._q_idxs, 1].set(
            BoosterUtils.UPPER_JOINT_LIMITS + 0.25)
        self.obs_limits = self.obs_limits.at[self._priv_q_idxs, 0].set(
            BoosterUtils.LOWER_JOINT_LIMITS - 0.25)
        self.obs_limits = self.obs_limits.at[self._priv_q_idxs, 1].set(
            BoosterUtils.UPPER_JOINT_LIMITS + 0.25)
        # self.obs_limits = self.obs_limits.at[self._priv_torque_idxs, 0].set(
        #     -1*BoosterUtils.MOTOR_TORQUE_LIMIT)
        # self.obs_limits = self.obs_limits.at[self._priv_torque_idxs, 1].set(
        #     BoosterUtils.MOTOR_TORQUE_LIMIT)
        # self.obs_limits = self.obs_limits.at[self._priv_feet_vel_idx, 0].set(
        #     -0.5)
        # self.obs_limits = self.obs_limits.at[self._priv_feet_vel_idx, 1].set(
        #     0.5)
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

        if self._include_height_in_obs:
            self.obs_limits = self.obs_limits.at[self._priv_h_idx, 0].set(0.0)
            self.obs_limits = self.obs_limits.at[self._priv_h_idx, 1].set(0.8)

        self.reward_config=config_dict.create(
            scales=config_dict.create(
                survival=0.25,
                tracking_lin_vel=0.0,#1.2,
                tracking_lin_vel_x=1.0,
                tracking_lin_vel_y=1.0,
                tracking_ang_vel=2.0,  # original 0.5
                base_height=0.2,
                orientation=-5.0,
                torques=0.0, #-2.0e-4 / 2,  # original -2.0e-4
                torque_tiredness=0.0, #-1.0e-2 / 2,  # original -1.0e-2
                power=0.0, #-2.0e-3 / 2,  # original -2.0e-3
                lin_vel_z=0.0, #-2.0,
                ang_vel_xy=0.0, #-0.2,
                dof_vel=0.0, #-1.0e-4,
                dof_acc=0.0, #-1.0e-7,
                root_acc=0.0, #-1.0e-4,
                action_rate=0.0,#-1.0 / 2,  # original -1.0
                dof_pos_limits=0.0, #-1.0,
                collision=0.0, #0.0,#-1.0 * 10.0,  # original -1.0
                feet_slip=-0.1,
                feet_vel_z=0.0, #-1.0,     # disabled in Isaac config
                feet_yaw_diff=-1.0,
                feet_yaw_mean=-1.0,
                feet_roll=-0.1 * 10.0,  # original -0.1
                feet_distance=-1.0 * 10.0,  # original -1.0
                feet_swing=3.0,
            ),
            tracking_sigma = tracking_sigma,
            base_height_target = 0.65,
            swing_period = 0.2,
            foot_collision_radius = 0.1115, # the radius of the geom of foot
        )

        # set up slices for the action space, defined by the table above
        self._action_size = 12

        self.feet_indices = jp.array([6,12])
        self.knee_indices = jp.array([4, 10])
        self.mini_ankle_dist = mini_ankle_dist
        self.default_joint_pd_target = BoosterUtils.ALL_STANDING_JOINT_ANGLES
        self.commands = jp.array([vel_x_command, 0.0, 0.0, 0.0])
        self.terminate_height_min = 0.3
        self.terminate_height_max = 0.8
        self.terminate_lin_vel_max = 10.0
        self.terminate_ang_vel_max = 10.0


        self._soft_joint_pos_limit_factor = 0.95
        c = (BoosterUtils.LOWER_JOINT_LIMITS + BoosterUtils.UPPER_JOINT_LIMITS) / 2
        r = BoosterUtils.UPPER_JOINT_LIMITS - BoosterUtils.LOWER_JOINT_LIMITS
        self._soft_lowers = c - 0.5 * r * self._soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._soft_joint_pos_limit_factor

    def reset(self, rng: jp.ndarray) -> State:

        # randomize initial yaw
        rng, rng_yaw = jax.random.split(rng)
        initial_yaw = jax.random.uniform(
            rng_yaw, shape=(),
            minval=self._initial_yaw_range[0]*180/jp.pi,
            maxval=self._initial_yaw_range[1]*180/jp.pi
        )
        initial_quat = eulerzyx_to_quat(jp.array([0.0, 0.0, initial_yaw]))

        # initialize system with initial q and qd
        q = self.sys.init_q  # init_q is defined in the xml file
        q = q.at[self._xml_quat_idxs].set(initial_quat)
        qd = jp.zeros(self.sys.qd_size())
        pipeline_state = self.pipeline_init(q, qd)

        # domain randomization
        domain_rand_rngs = jax.random.split(rng, 7)
        self._contact_time_const = jax.random.uniform(
            domain_rand_rngs[0], shape=(),
            minval=self._contact_time_const_range[0],
            maxval=self._contact_time_const_range[1]
        )
        self._friction = jax.random.uniform(
            domain_rand_rngs[1], shape=(),
            minval=self._friction_range[0],
            maxval=self._friction_range[1]
        )
        self._ground_roll = jax.random.uniform(
            domain_rand_rngs[2], shape=(),
            minval=self._ground_roll_range[0],
            maxval=self._ground_roll_range[1]
        )
        self._ground_pitch = jax.random.uniform(
            domain_rand_rngs[3], shape=(),
            minval=self._ground_pitch_range[0],
            maxval=self._ground_pitch_range[1]
        )
        self._joint_damping = jax.random.uniform(
            domain_rand_rngs[4], shape=self.sys.dof.damping.shape,
            minval=self._joint_damping_perc_range[0] * self.sys.dof.damping,
            maxval=self._joint_damping_perc_range[1] * self.sys.dof.damping
        )
        self._joint_gain = jax.random.uniform(
            domain_rand_rngs[5], shape=self.sys.actuator.gain.shape,
            minval=self._joint_gain_range[0],
            maxval=self._joint_gain_range[1]
        )
        self._link_mass_perc = jax.random.uniform(
            domain_rand_rngs[6], shape=self.sys.link.inertia.mass.shape,
            minval=self._link_mass_perc_range[0],
            maxval=self._link_mass_perc_range[1]
        )
        # initialize metrics

        # we use info to pass along quantities for domain randomization
        feet_air_time = jp.zeros(2)
        last_contacts = jp.array([False, False], dtype=jp.bool_)
        last_root_vel = jp.zeros(6)
        last_last_actions = jp.zeros(self._action_size)
        last_actions = jp.zeros(self._action_size)
        actions = jp.zeros(self._action_size)
        info = {
            'contact_time_const': self._contact_time_const,
            'contact_damping_ratio': self._contact_damping_ratio,
            'friction': self._friction,
            'ground_roll': self._ground_roll,
            'ground_pitch': self._ground_pitch,
            'joint_damping': self._joint_damping,
            'joint_gain': self._joint_gain,
            'link_mass_perc': self._link_mass_perc,
        }

        # compute mass matrix and bias + passive forces
        sys = self.sys
        x, xd = kinematics.forward(sys, q, qd)
        rew_info = {
            'gait_process': jp.zeros(()),
            'gait_frequency': jp.ones(())*1.5,
            'last_last_actions': last_last_actions,
            'last_actions': last_actions,
            'rigid_state_pos': x.pos,
            'rigid_state_lin_vel': xd.vel,
            'rigid_state_ang_vel': xd.ang,
            'rigid_state_rot': x.rot,
            'rigid_state_qdd': pipeline_state.qdd,

        }

        empty_cmd = ControlCommand(
            q_des=jp.zeros((12,)),
            qd_des=jp.zeros((12,)),
            Kp=jp.zeros((12,)),
            Kd=jp.zeros((12,)),
        )
        info['cmd'] = empty_cmd

        # initial observations, reward, done, and u
        norm_obs = self._get_obs(pipeline_state, rew_info, jp.zeros(self._action_size), jp.zeros(self._action_size))
        # compute cmd for info
        _, rew_components = self.compute_reward(
            norm_obs, norm_obs, jp.zeros(self._action_size), jp.zeros(self._action_size), rew_info)

        metrics = {k: jp.zeros(()) for k, v in rew_components.items()}

        metrics.update(
            step_count=jp.zeros(()),
            forward_vel=jp.zeros(()),
            )

        obs = {
            'state': norm_obs[:self._observation_size],
            'privileged_state': norm_obs
        }
        reward, done = jp.zeros(2)
        u = jp.zeros(self._action_size)

        return State(pipeline_state, obs, reward, done, metrics,
                     info=info, u=u, rew_info=rew_info, torque=jp.zeros(self._action_size))


    def low_level_control(self, scaled_action: jp.ndarray,
                          unused_norm_obs: jp.ndarray) -> jp.ndarray:
        # Here we simply return the action as we are treating as "control
        # inputs" to the env the actions. Low level PD torque control is
        # instead absorbed into the appoximate dynamics.
        return scaled_action
    

    def step(self, state: State, action: jp.ndarray) -> State:

        # overwrite system contact properties with the environment's
        rew_info = state.rew_info
        sys = self._update_system_properties(state)

        # get observations from state
        prev_norm_obs = self._get_obs(state.pipeline_state, rew_info, state.u, state.torque)

        def f(info_list, _):
            pipeline_state, _, _ = info_list
            norm_obs = self._get_obs(pipeline_state, rew_info, state.u, state.torque)
            obs = self._denormalize_obs(norm_obs)
            q = obs[self._q_idxs]
            qd = obs[self._qd_idxs]
            torque, cmd = self.torque_pd_control(action, BoosterUtils.KP, BoosterUtils.KD, q, qd)
            pipeline_state = pipeline_step(sys, pipeline_state, torque)
            return (pipeline_state, torque, cmd), _

        (new_pipeline_state, torque, cmd), _ = jax.lax.scan(f, (state.pipeline_state, state.torque, state.info["cmd"]),
                                             (), self.policy_repeat)

        # get new observations and compute reward
        sys = self.sys
        new_norm_obs = self._get_obs(new_pipeline_state, rew_info, action, torque)
        new_obs = self._denormalize_obs(new_norm_obs)
        
        q, qd = self.q_and_qd_from_obs(new_obs)
        x, xd = kinematics.forward(sys, q, qd)
        reward, rew_components = self.compute_reward(
            norm_obs=new_norm_obs, prev_norm_obs=prev_norm_obs, torques=torque, action=action, info=rew_info)
        metrics = {k: v for k, v in rew_components.items()}
        metrics.update(
            forward_vel=self._denormalize_obs(new_norm_obs)[self._forward_vel_idx],
            step_count=state.metrics['step_count'] + 1,
        )
                             
            
        # compute dones for resets
        done = self.is_done(new_norm_obs)


        info = state.info
        info['cmd'] = cmd
        new_rew_info = {}

        new_rew_info['last_last_actions'] = rew_info['last_actions']
        new_rew_info['last_actions'] = action

        new_rew_info.update({
            'rigid_state_pos': x.pos,
            'rigid_state_lin_vel': xd.vel,
            'rigid_state_ang_vel': xd.ang,
            'rigid_state_rot': x.rot,
            'rigid_state_qdd': new_pipeline_state.qdd,
            'gait_frequency': rew_info['gait_frequency'],
            'gait_process': jp.fmod(rew_info['gait_process'] + self.sim_dt*self.policy_repeat * rew_info['gait_frequency'], 1.0), 

        })

        obs_dict={
            'state': new_norm_obs[:self._observation_size],
            'privileged_state': new_norm_obs
        }

        return State(new_pipeline_state, obs_dict, reward, done, metrics,
                     info=info, u=action, rew_info=new_rew_info, torque=torque)


    def is_done(self, next_norm_obs: jp.ndarray) -> jp.ndarray:
        """Returns the done signal."""
        done = 1.0 - self._is_healthy(next_norm_obs)
        return done


    def _is_healthy(self, next_norm_obs: jp.ndarray) -> jp.ndarray:
        """Returns the healthy signal."""
        next_obs = self._denormalize_obs(next_norm_obs)
        quat = next_obs[self._priv_quat_idxs] / jp.linalg.norm(next_obs[self._priv_quat_idxs])
        roll, pitch, yaw = quat_to_eulerzyx(next_obs[self._priv_quat_idxs])

        base_vel_body = next_obs[self._priv_base_vel_idxs]
        base_vel_global = rotate(base_vel_body, quat)
        ang_vel_body = next_obs[self._priv_rpy_rate_idxs]
        ang_vel_global = rotate(ang_vel_body, quat)

        # Combine all conditions into one single condition using jp.logical_and:
        condition = next_obs[self._priv_h_idx] > self.terminate_height_min
        condition = jp.logical_and(condition, next_obs[self._priv_h_idx] < self.terminate_height_max)
        condition = jp.logical_and(condition, jp.all(jp.abs(base_vel_global) < self.terminate_lin_vel_max))
        condition = jp.logical_and(condition, jp.all(jp.abs(ang_vel_global) < self.terminate_ang_vel_max))
        condition = jp.logical_and(condition, jp.abs(roll) < self._fallen_roll)
        condition = jp.logical_and(condition, jp.abs(pitch) < self._fallen_pitch)
        condition = jp.logical_and(condition, jp.all(next_obs[self._priv_q_idxs] < BoosterUtils.UPPER_JOINT_LIMITS*1.5))
        condition = jp.logical_and(condition, jp.all(next_obs[self._priv_q_idxs] > BoosterUtils.LOWER_JOINT_LIMITS*1.5))
        condition = jp.logical_and(condition, jp.all(next_obs[self._priv_qd_idxs] < BoosterUtils.MOTOR_VEL_LIMIT*4))
        condition = jp.logical_and(condition, jp.all(next_obs[self._priv_qd_idxs] > -BoosterUtils.MOTOR_VEL_LIMIT*4))
        # Now use jp.where with the single composite condition.
        is_healthy = jp.where(condition, 1.0, 0.0)


        return is_healthy

    def _update_system_properties(self, state: State):
        """Updates the system properties used for physics simulation with
        values that were set by the domain randomization"""
        sys = self.sys

        contact_time_const = state.info['contact_time_const']
        contact_damping_ratio = state.info['contact_damping_ratio']
        friction = state.info['friction']
        ground_roll = state.info['ground_roll']
        ground_pitch = state.info['ground_pitch']
        ground_quat = eulerzyx_to_quat(jp.array([ground_roll, ground_pitch, 0.0]))
        new_geoms = [
            g.replace(
                solver_params=g.solver_params.at[0, 0].set(contact_time_const)
            ) for g in sys.geoms
        ]
        new_geoms = [
            g.replace(
                solver_params=g.solver_params.at[0, 1].set(contact_damping_ratio)
            ) for g in new_geoms
        ]
        new_geoms = [
            g.replace(
                friction=g.friction.at[:].set(friction)
            ) for g in new_geoms
        ]
        new_geoms[0] = new_geoms[0].replace(
            transform=new_geoms[0].transform.replace(
                rot=new_geoms[0].transform.rot.at[0, :].set(ground_quat)
            )
        )

        joint_damping = state.info['joint_damping']
        new_dof = sys.dof.replace(
            damping=sys.dof.damping.at[:].set(joint_damping)
        )

        joint_gain = state.info['joint_gain']
        new_actuator = sys.actuator.replace(
            gain=sys.actuator.gain.at[:].set(joint_gain)
        )

        link_mass_perc = state.info['link_mass_perc']
        new_link = sys.link.replace(
            inertia=sys.link.inertia.replace(
                mass=sys.link.inertia.mass.at[:].set(
                    link_mass_perc * sys.link.inertia.mass
                ),
                i=sys.link.inertia.i.at[:, :, :].set(
                    jp.expand_dims(link_mass_perc, axis=(1, 2)) * sys.link.inertia.i
                )
            )
        )

        sys = sys.replace(geoms=new_geoms, dof=new_dof, actuator=new_actuator,
                          link=new_link)
        return sys
    
    def q_and_qd_from_obs(self, obs: jp.ndarray):
        q = self.sys.init_q
        # normalize quat
        quat = obs[self._priv_quat_idxs] / jp.linalg.norm(obs[self._priv_quat_idxs])
        q = q.at[self._xml_quat_idxs].set(quat)
        q = q.at[self._xml_q_idxs].set(obs[self._priv_q_idxs])
        if self._include_height_in_obs:
            q = q.at[self._xml_h_idxs].set(obs[self._priv_h_idx])

        base_vel_body = obs[self._priv_base_vel_idxs]
        base_vel_global = rotate(base_vel_body, quat)
        ang_vel_body = obs[self._priv_rpy_rate_idxs]
        qd = jp.zeros(self.sys.qd_size())
        qd = qd.at[self._xml_base_vel_idxs].set(base_vel_global)
        qd = qd.at[self._xml_rpy_rate_idxs].set(ang_vel_body)
        qd = qd.at[self._xml_qd_idxs].set(obs[self._priv_qd_idxs])

        return q, qd

    def _get_obs(self, pipeline_state: base.State,
                 info: dict,
                 last_action: jp.ndarray, # TODO
                 last_torque: jp.ndarray, # TODO
                 ) -> jp.ndarray:
        """uses metrics to compute phase and desired velocity"""

        quat, q, base_vel_body, ang_vel_body, qd, q_all, qd_all = self._get_basic_obs(pipeline_state)

        x, xd = kinematics.forward(self.sys, q_all, qd_all)


        gait_process = info['gait_process']
        gait_frequency = info['gait_frequency']

        foot_speed = xd.vel[self.feet_indices]
        cos_phase = jp.cos(2*jp.pi*gait_process) * (gait_frequency > 1.0e-8).astype(gait_process.dtype) * (jp.linalg.norm(self.commands[:3]) > 0.01).astype(gait_process.dtype)
        sin_phase = jp.sin(2*jp.pi*gait_process) * (gait_frequency > 1.0e-8).astype(gait_process.dtype) * (jp.linalg.norm(self.commands[:3]) > 0.01).astype(gait_process.dtype)
        phase = jp.array([cos_phase, sin_phase])
        phase = jp.where(
            jp.linalg.norm(self.commands[:3]) > 0.01,
            phase,
            jp.zeros(2),
        )
        #torques, _ = self.torque_pd_control(last_action, q, qd)
        torques = last_torque

        projected_gravity = inv_rotate(jp.array([0.0, 0.0, -1.0]), quat)

        gait = jp.array([
            gait_process, gait_frequency
        ])
        # TODO: align 
        if self._include_height_in_obs:
            h = pipeline_state.q[self._xml_h_idxs]
            obs = jp.concatenate([
                projected_gravity, 
                ang_vel_body, 
                self.commands[:3], 
                phase, 
                q, 
                qd, 
                last_action, 
                ang_vel_body, 
                projected_gravity, 
                quat, 
                base_vel_body, 
                q, 
                qd, 
                h, 
                #torques, 
                gait
                ])

        return self._normalize_obs(obs)


    def _get_basic_obs(self, pipeline_state: base.State) -> jp.ndarray:
        "Returns basic observations without phase and desired velocity"

        positions = pipeline_state.q
        velocities = pipeline_state.qd

        # quat orientation of the base
        quat = positions[self._xml_quat_idxs]

        # joint angles
        q = positions[self._xml_q_idxs]

        # linear velocity of the base in the body frame
        base_vel_global = velocities[self._xml_base_vel_idxs]
        base_vel_body = inv_rotate(base_vel_global, quat)

        # angular velocity of the base in the body frame
        ang_vel_body = velocities[self._xml_rpy_rate_idxs] #rotate(ang_vel_global, quat)

        # joint speeds
        qd = velocities[self._xml_qd_idxs]

        return quat, q, base_vel_body, ang_vel_body, qd, positions, velocities


    def _normalize_obs(self, obs: jp.ndarray) -> jp.ndarray:
        return (2*(obs - self.obs_limits[:, 0])
                / (self.obs_limits[:, 1] - self.obs_limits[:, 0])
                - 1)

    def _denormalize_obs(self, obs: jp.ndarray) -> jp.ndarray:
        return ((obs + 1)*(self.obs_limits[:, 1] - self.obs_limits[:, 0])/2
                + self.obs_limits[:, 0])
    
    def compute_reward(self, norm_obs: jp.ndarray, prev_norm_obs: jp.ndarray,
                       torques: jp.ndarray,
                       action: jp.ndarray,
                    info:dict,
                       ) -> jp.ndarray:
        obs = self._denormalize_obs(norm_obs)
        prev_obs = self._denormalize_obs(prev_norm_obs)
        reward, reward_components = self._reward_normalized(obs=obs, prev_obs=prev_obs, torques=torques, action=action,
                                                    info=info)
            
        #is_healthy = self._is_healthy(norm_obs)
        reward = reward

        return reward, reward_components

    def _reward_normalized(self, obs: jp.ndarray,
                           prev_obs: jp.ndarray,
                           torques: jp.ndarray,
                           action: jp.ndarray,
                            info: dict) -> jp.ndarray:
        gait_process = obs[self._priv_gait_process_idx]
        gait_frequency = obs[self._priv_gait_frequency_idx]
        last_actions = info['last_actions']
        last_last_actions = info['last_last_actions']
        rigid_state_pos = info['rigid_state_pos']
        rigid_state_lin_vel = info['rigid_state_lin_vel']
        _ = info['rigid_state_ang_vel']
        rigid_state_rot = info['rigid_state_rot']
        rigid_state_qdd = info['rigid_state_qdd']

        quat = obs[self._priv_quat_idxs] / jp.linalg.norm(obs[self._priv_quat_idxs])

        base_lin_vel = obs[self._priv_base_vel_idxs]
        base_ang_vel = obs[self._priv_rpy_rate_idxs]
        base_euler_xyz = quat_to_euler(quat)
        projected_gravity = inv_rotate(jp.array([0.0, 0.0, -1.0]), quat)
        contact = rigid_state_pos[self.feet_indices, 2] < self.mini_ankle_dist

        commands = obs[self._command_idxs]

        feet_pos = rigid_state_pos[self.feet_indices, :3]
        feet_quat = rigid_state_rot[self.feet_indices, :4]
        # concat feet_quat
        feet_euler_xyz_0 = quat_to_euler(feet_quat[0])
        feet_euler_xyz_1 = quat_to_euler(feet_quat[1])
        feet_euler_xyz = jp.array([feet_euler_xyz_0, feet_euler_xyz_1])
        feet_roll = (feet_euler_xyz[:,0] + jp.pi) % (2 * jp.pi) - jp.pi
        feet_yaw = (feet_euler_xyz[:,2] + jp.pi) %  (2 * jp.pi) - jp.pi




        reward_components = {
            # -------- positive terms --------

            "survival": jp.array(1.0),
            "tracking_lin_vel": self._reward_tracking_lin_vel(commands, base_lin_vel),
            "tracking_lin_vel_x": self._reward_tracking_lin_vel_axis(0, commands, base_lin_vel),
            "tracking_lin_vel_y": self._reward_tracking_lin_vel_axis(1, commands, base_lin_vel),
            "tracking_ang_vel": self._reward_tracking_ang_vel(commands, base_ang_vel),
            "feet_swing": self._reward_feet_swing(gait_process=gait_process, gait_frequency=gait_frequency, feet_contact=contact),

            # -------- penalties ------------ (signed handled by scale)
            "base_height": self._reward_base_height(height=rigid_state_pos[0, 2]),
            "orientation": self._cost_orientation(projected_gravity),
            "torques": self._cost_torques(torques),
            "torque_tiredness": self._cost_torque_tiredness(torques),
            "power": self._cost_energy(qvel=obs[self._priv_qd_idxs], torques=torques),
            "lin_vel_z": self._cost_lin_vel_z(base_lin_vel),
            "ang_vel_xy": self._cost_ang_vel_xy(base_ang_vel),
            "dof_vel": self._cost_dof_vel(obs[self._priv_qd_idxs]),
            "dof_acc": self._cost_dof_acc(rigid_state_qdd[6:]),
            "root_acc": self._cost_root_acc(rigid_state_qdd),
            "action_rate": self._cost_action_rate(act=action, last_act=last_actions, last_last_act=last_last_actions),
            "dof_pos_limits": self._cost_joint_pos_limits(obs[self._priv_q_idxs]),
            "collision": self._cost_collision(left_foot_pos=feet_pos[0], right_foot_pos=feet_pos[1]),
            "feet_slip": self._cost_feet_slip(feet_vel=rigid_state_lin_vel[self.feet_indices], contact=contact),
            "feet_vel_z": self._cost_feet_vel_z(feet_vel=rigid_state_lin_vel[self.feet_indices]),
            "feet_roll": self._cost_feet_roll(feet_roll=feet_roll),
            "feet_yaw_diff": self._cost_feet_yaw_diff(feet_yaw=feet_yaw),
            "feet_yaw_mean": self._cost_feet_yaw_mean(feet_yaw=feet_yaw, base_yaw=base_euler_xyz[2]),
            "feet_distance": self._cost_feet_distance(feet_pos=feet_pos, base_yaw=base_euler_xyz[2]),
        }

        rewards_comp = {k: v * self.reward_config.scales[k] for k, v in reward_components.items()}

        reward = jp.clip(sum(rewards_comp.values()) * self.sim_dt * self.policy_repeat, 0.0, 10000.0)
        rewards_comp['reward']  = reward
        return reward, rewards_comp

    # Tracking rewards.
    def _reward_tracking_lin_vel(
        self, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
        err = jp.sum(jp.square(command[:2] - local_linvel[:2]))
        return jp.exp(-err / self.reward_config.tracking_sigma)


    def _reward_tracking_lin_vel_axis(
        self, axis: int, command: jax.Array, local_linvel: jax.Array
    ) -> jax.Array:
        """Axis–wise linear‑velocity tracker (matches Isaac Gym x & y trackers)."""
        err = jp.square(command[axis] - local_linvel[axis])
        return jp.exp(-err / self.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        local_angvel: jax.Array,
    ) -> jax.Array:
        ang_vel_error = jp.square(commands[2] - local_angvel[2])
        return jp.exp(-ang_vel_error / self.reward_config.tracking_sigma)

    # Base-related rewards.

    def _cost_lin_vel_z(self, local_linvel) -> jax.Array:
        return jp.square(local_linvel[2])

    def _cost_ang_vel_xy(self, local_angvel) -> jax.Array:
        return jp.sum(jp.square(local_angvel[:2]))

    def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
        return jp.sum(jp.square(torso_zaxis[:2]))

    def _reward_base_height(self, height: float) -> jax.Array:
        return jp.exp(-jp.abs(height - self.reward_config.base_height_target)*0.1)

    # Energy related rewards.

    def _cost_torques(self, torques: jp.ndarray) -> jp.ndarray:
        return jp.sum(jp.square(torques))

    def _cost_energy(self, qvel: jp.ndarray, torques: jp.ndarray) -> jp.ndarray:
        power = qvel * torques
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

    def _cost_collision(self,     
        left_foot_pos: jp.ndarray,
        right_foot_pos: jp.ndarray,
        ) -> jax.Array:
        # 计算脚心之间的欧氏距离
        dist = jp.linalg.norm(left_foot_pos - right_foot_pos)
        # 如果距离小于 2*radius，则产生穿透，返回穿透深度；否则返回 0
        overlap = jp.maximum(2.0 * self.reward_config.foot_collision_radius - dist, 0.0)
        return overlap > 0.0

    # Feet related rewards.

    def _cost_feet_slip(
        self,
        feet_vel: jp.ndarray,
        contact: jp.ndarray,
    ) -> jp.ndarray:
        speed2 = jp.sum(jp.square(feet_vel), axis=-1)             # per‑foot |v|²
        return jp.sum(speed2 * contact)

    def _cost_feet_distance(
        self,
        feet_pos: jax.Array,
        base_yaw: jax.Array,
    ) -> jax.Array:
        left_foot_pos = feet_pos[0]
        right_foot_pos = feet_pos[1]
        feet_distance = jp.abs(
            jp.cos(base_yaw) * (left_foot_pos[1] - right_foot_pos[1])
            - jp.sin(base_yaw) * (left_foot_pos[0] - right_foot_pos[0])
        )
        return jp.clip(0.2 - feet_distance, min=0.0, max=0.1)


    def _cost_torque_tiredness(self, torques: jax.Array) -> jax.Array:
        # Σ (τ / τ_max)²  – clipped at 1 so the term stays O(1)

        frac = jp.clip(jp.abs(torques) / jp.full_like(BoosterUtils.MOTOR_TORQUE_LIMIT, 1e6), 0.0, 1.0)
        return jp.sum(jp.square(frac))

    def _cost_root_acc(self, qdd: jax.Array) -> jax.Array:
        # Root‑link 6‑D acceleration² (free‑joint entries of qacc)
        return jp.sum(jp.square(qdd[:6]))

    def _cost_feet_roll(self, feet_roll: jax.Array) -> jax.Array:
        return jp.sum(jp.square(feet_roll))

    def _cost_feet_yaw_diff(self, feet_yaw: jax.Array) -> jax.Array:
        diff = jp.fmod(feet_yaw[1] - feet_yaw[0] + jp.pi, 2 * jp.pi) - jp.pi
        return jp.square(diff)

    def _cost_feet_yaw_mean(self, feet_yaw: jax.Array, base_yaw: jax.Array) -> jax.Array:
        feet_mean_yaw = jp.fmod(jp.mean(feet_yaw) + jp.pi, 2 * jp.pi) - jp.pi
        err = jp.fmod(base_yaw - feet_mean_yaw + jp.pi, 2 * jp.pi) - jp.pi

        return jp.square(err)

    def _cost_feet_vel_z(self, feet_vel: jp.ndarray) -> jax.Array:
        # use the same foot linear‑velocity sensors already wired for slip
        vz = feet_vel[:, 2]
        return jp.sum(jp.square(vz))

    def _reward_feet_swing(self,
        gait_process: jp.ndarray,
        gait_frequency: float,
        feet_contact: jp.ndarray) -> jp.ndarray:
        left_swing = (jp.abs(gait_process - 0.25) < 0.5 * self.reward_config.swing_period) & (gait_frequency > 1.0e-8)
        right_swing = (jp.abs(gait_process - 0.75) < 0.5 * self.reward_config.swing_period) & (gait_frequency > 1.0e-8)
        ref_rew = (left_swing & ~feet_contact[0]) + (right_swing & ~feet_contact[1])
        return ref_rew

    def torque_pd_control(self, action: jp.ndarray,
                          kp: jp.ndarray,
                          kd: jp.ndarray,
                          q: jp.ndarray,
                          qd: jp.ndarray,
                          limit_Kp: bool = True) -> Tuple[jp.ndarray, ControlCommand]:

        cmd = self.low_level_control_hardware(action, kp=kp, kd=kd, limit_Kp=limit_Kp)

        # torque control
        u = cmd.Kp*(cmd.q_des - q) + cmd.Kd*(cmd.qd_des - qd)

        u = jp.clip(u, -BoosterUtils.MOTOR_TORQUE_LIMIT, BoosterUtils.MOTOR_TORQUE_LIMIT)   
        return u, cmd
    
    def low_level_control_hardware(self, action: jp.ndarray,
                                    kp: jp.ndarray,
                                    kd: jp.ndarray,
                                   limit_Kp: bool = True) -> ControlCommand:

        q_des = action   + BoosterUtils.ALL_STANDING_JOINT_ANGLES

        qd_des = jp.zeros((self.action_size,))



        q_des = jp.clip(q_des,
                        BoosterUtils.LOWER_JOINT_LIMITS,
                        BoosterUtils.UPPER_JOINT_LIMITS)
        # Kp = jax.lax.cond(limit_Kp,
        #                   self._limit_Kp,
        #                   lambda *args: Kp,
        #                   obs, q_des, qd_des, Kp, Kd)

        return ControlCommand(q_des, qd_des, kp, kd)

    def _limit_Kp(self, obs, q_des, qd_des, Kp, Kd):
        # limits kp if torque is too high
        q = obs[self._q_idxs]
        qd = obs[self._qd_idxs]
        torque_theo = Kp*(q_des - q) + Kd*(qd_des - qd)
        q_err = jp.where(q_des - q != 0, q_des - q, 1)  # avoid div by zero
        Kp = jp.where(
            torque_theo > BoosterUtils.MOTOR_TORQUE_LIMIT,
            (BoosterUtils.MOTOR_TORQUE_LIMIT - Kd*(qd_des - qd))/q_err, Kp)
        Kp = jp.where(
            torque_theo < -BoosterUtils.MOTOR_TORQUE_LIMIT,
            (-BoosterUtils.MOTOR_TORQUE_LIMIT - Kd*(qd_des - qd))/q_err, Kp)
        return Kp



    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def controls_size(self) -> int:
        return self._action_size

    @property
    def observation_size(self) -> int:
        return {
            "state": self._observation_size,
            "privileged_state": self._priv_observation_size,
        }

    @property
    def dt(self) -> jp.ndarray:
        """The timestep used for each env step."""
        return self.sim_dt * self.policy_repeat

