# Copyright 2025 The Brax Authors.
# Modifications Copyright 2025 LIFT Author
#
# This file is ADAPTED FROM the Brax project and includes local modifications.
# Original source licensed under the Apache License, Version 2.0 (the "License").
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"

from jax import config as jax_config
jax_config.update("jax_enable_x64", False)

import time
from pathlib import Path
import argparse
import jax
import jax.numpy as jp  # noqa: F401  # kept for potential downstream use
#from brax.training.acme import running_statistics
from omegaconf import OmegaConf
from utils import running_statistics
import sys
#import utils.running_statistics as rs
sys.modules['brax.training.acme.running_statistics'] = running_statistics
import dill

from policy_pretrain import sac_networks
from brax.io import html

# ---------- OmegaConf resolver (from your originals) ----------
def int_multiply(x, y):
    return int(x * y)

OmegaConf.register_new_resolver("int_multiply", int_multiply)

# ---------- Build SAC networks (shared) ----------
normalize_fn = running_statistics.normalize
env_network = sac_networks.make_sac_networks(
    observation_size=47,
    action_size=12,
    preprocess_observations_fn=normalize_fn,
    policy_hidden_layer_sizes=(512, 256, 128),
    q_hidden_layer_sizes=(1024, 512, 256),
    activation='swish'
)

# ---------- Common env kwargs base (from both files) ----------
COMMON_KWARGS = dict(
    policy_repeat=10,
    forward_cmd_vel_type="constant",
    forward_cmd_vel_range=0.0,
    forward_cmd_vel_period_range=[40.0, 40.0],
    turn_cmd_rate_range=[-0.0, 0.0],
    initial_yaw_range=[-0.0, 0.0],
    contact_time_const=0.02,
    contact_damping_ratio=1.0,
    friction_range=[0.6, 0.6],
    ground_roll_range=[0.0, 0.0],
    ground_pitch_range=[0.0, 0.0],
    joint_damping_perc_range=[1.0, 1.0],
    joint_gain_range=[1.0, 1.0],
    link_mass_perc_range=[1.0, 1.0],
    fallen_roll=0.785,
    fallen_pitch=0.785,
    forces_in_q_coords=True,
    include_height_in_obs=True,
    body_height_in_action_space=False,
    gains_in_action_space=False,
    healthy_delta_radius=2.0,
    healthy_delta_yaw=1.57,
    vel_x_command=1.5,
)

# env-specific overrides pulled from your two files
ENV_OVERRIDES = {
    "g1": dict(mini_ankle_dist=0.04),      # from file1
    "t1": dict(mini_ankle_dist=0.02), # from file2
}

def build_env(env_name: str):
    """Instantiate the chosen Brax environment with merged kwargs."""
    env_name = env_name.lower()
    if env_name == "g1":
        from brax.envs.g1_go_fast import G1GoFast
        kwargs = {**COMMON_KWARGS, **ENV_OVERRIDES["g1"]}
        env = G1GoFast(backend="generalized", **kwargs)
    elif env_name == "t1":
        from brax.envs.booster_go_fast import BoosterGoFast
        kwargs = {**COMMON_KWARGS, **ENV_OVERRIDES["t1"]}
        env = BoosterGoFast(backend="generalized", **kwargs)
    else:
        raise ValueError("env must be one of {'g1','t1'}")
    return env

def evaluate_and_render(env, sac_ts, max_steps=1000, render_height=500, out_name="evaluation_render.html"):
    @jax.jit
    def jit_step(state, action):
        return env.step(state, action)

    @jax.jit
    def jit_reset(key):
        return env.reset(key)

    key = jax.random.PRNGKey(0)
    state = jit_reset(key)
    states = []
    rew = 0.0
    start_time = time.time()

    for i in range(1, max_steps + 1):
        logits = env_network.policy_network.apply(sac_ts[0], sac_ts[1], state.obs)
        action = env_network.parametric_action_distribution.mode(logits)

        state = jit_step(state, action)
        rew += state.reward

        # health check (as in your originals)
        is_healthy = env._is_healthy(state.obs["privileged_state"])
        if not is_healthy:
            break

        states.append(state.pipeline_state)

        if i % 10 == 0:
            now = time.time()
            print(f"Step {i}")
            print(f"Time elapsed: {now - start_time:.2f}s")
            start_time = now

    print("Total reward:", float(rew))

    render_html = html.render(env.sys.replace(dt=env.dt), states, height=render_height)
    output_path = Path(__file__).parent / out_name
    with open(output_path, "w") as f:
        f.write(render_html)
    print(f"Render saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and render policy in Brax envs.")
    parser.add_argument("--env", type=str, default="t1", choices=["t1", "g1"],
                        help="Which environment to run.")
    parser.add_argument("--model", type=str, default="models/step_40000_rew_104.pkl",
                        help="Path to dill-pickled (params_state, params) tuple.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max simulation steps.")
    parser.add_argument("--height", type=int, default=500, help="Render viewport height.")
    parser.add_argument("--out", type=str, default="", help="Output HTML filename.")
    args = parser.parse_args()

    # Load model (dill)
    if not Path(args.model).exists():
        raise FileNotFoundError(
            f"Model file not found: {args.model}\n"
            f"(Set with --model path/to/file.pkl)"
        )
    with open(args.model, "rb") as f:
        sac_ts = dill.load(f)
        print("Loaded model:", type(sac_ts), "keys/types:", [type(x) for x in sac_ts] if hasattr(sac_ts, "__iter__") else None)

    env = build_env(args.env)

    # Default output name based on env if not provided
    out_name = args.out or (f"evaluation_render_{args.env}.html")
    evaluate_and_render(env, sac_ts, max_steps=args.max_steps, render_height=args.height, out_name=out_name)
    
# python eval_in_brax.py --env g1 --model models/g1_40000_rew143.pkl
# python eval_in_brax.py --env g1 --model models/step_40000_rew_104.pkl
if __name__ == "__main__":
    main()
