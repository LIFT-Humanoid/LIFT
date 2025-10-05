import os
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
#from utils.model import *
from torch.distributions import Normal

class TanhBijector:
    """Tanh Bijector."""

    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, y):
        # Clamping the input to avoid numerical issues with arctanh
        return torch.arctanh(torch.clamp(y, -0.999999, 0.999999))

    def forward_log_det_jacobian(self, x):
        # Computing the log of the absolute value of the Jacobian determinant
        return 2. * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2. * x))


class NormalTanhDistribution:
    """Normal distribution followed by tanh."""

    def __init__(self, min_std=0.001, max_std=None):
        self.min_std = min_std
        self.max_std = max_std
        self.postprocessor = TanhBijector()

    def create_dist(self, parameters):
        loc, scale = torch.chunk(parameters, 2, dim=-1)
        if self.max_std is None:
            scale = F.softplus(scale) + self.min_std
        else:
            scale = torch.sigmoid(scale)
            scale = self.min_std + (self.max_std - self.min_std) * scale
        return Normal(loc, scale)

    def sample(self, parameters):
        dist = self.create_dist(parameters)
        return self.postprocessor.forward(dist.rsample())

    def mode(self, parameters):
        dist = self.create_dist(parameters)
        return self.postprocessor.forward(dist.mean)


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint
    policy = torch.jit.load(cfg["basic"]["checkpoint"])

    #policy = load_policy(cfg["basic"]["checkpoint"])
    #policy = torch.compile(policy)

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)
    normalizer = NormalTanhDistribution()

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0
    obs_mean = np.array([ 2.21614912e-02, -6.62935758e-03, -9.89770353e-01, -1.84123067e-03,
       -1.05458626e-03, -3.48594203e-03, -1.35526909e-02,  5.11697785e-04,
        1.54562516e-03,  1.50413706e-03, -7.06638210e-04, -3.33513528e-01,
        7.78222755e-02, -4.17490155e-02,  5.26312172e-01, -1.20869458e-01,
       -1.03857972e-01, -4.04644668e-01, -7.41391554e-02,  7.49955326e-02,
        5.31088114e-01, -9.97376591e-02,  1.02060944e-01, -2.47895136e-03,
        3.20627441e-04,  8.47006449e-04,  4.78907768e-03, -2.23732460e-03,
        8.21477210e-04, -2.81094946e-03, -2.26590651e-04,  3.80213809e-04,
        4.62141959e-03, -1.09391217e-03, -7.45175115e-04, -3.37301910e-01,
        1.75762802e-01, -9.18873921e-02,  2.82788008e-01, -3.28020975e-02,
       -1.21201731e-01, -4.13269728e-01, -1.69798374e-01,  1.24133684e-01,
        2.68622637e-01, -9.55357030e-02,  1.15391515e-01], dtype=np.float32)
    obs_std = np.array([0.0956538 , 0.09019899, 0.07090962, 0.62016124, 0.6689715 ,
       0.76529443, 0.5644678 , 0.4491155 , 0.55393225, 0.6727647 ,
       0.671542  , 0.16333263, 0.11988021, 0.15301014, 0.2176434 ,
       0.17444059, 0.1672211 , 0.17699417, 0.12580965, 0.14642641,
       0.22388743, 0.17983808, 0.16669112, 0.20540868, 0.16352113,
       0.21124631, 0.29779342, 0.42500436, 0.44682735, 0.21028578,
       0.16216691, 0.2101046 , 0.3005955 , 0.44228187, 0.44563502,
       0.2673667 , 0.2923741 , 0.23901007, 0.5012833 , 0.43059075,
       0.40789968, 0.2820594 , 0.2805095 , 0.24342726, 0.507204  ,
       0.42896807, 0.40686968], dtype=np.float32)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = 1.5#np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: gait_frequency={gait_frequency}, x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            if it % cfg["control"]["decimation"] == 0:
                obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
                obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
                obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:23] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[35:47] = actions
                obs = (obs - obs_mean) / obs_std
                dist = policy(torch.tensor(obs).unsqueeze(0))        # -> shape [1, 2*A]
                #actions = dist.detach().numpy()     # -> shape [1, A]

                actions = normalizer.mode(dist).detach().numpy()     # -> shape [1, A]
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)
