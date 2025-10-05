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

"""SAC networks."""
from typing import Literal, Sequence, Tuple

import flax
from flax import linen
import jax.numpy as jnp
import jax

from utils import distribution
from utils import types
from policy_pretrain import base_networks as networks
from policy_pretrain.base_networks import make_q_network, make_policy_network


@flax.struct.dataclass
class SACNetworks:
    policy_network: networks.FeedForwardNetwork
    q_network: networks.FeedForwardNetwork
    parametric_action_distribution: distribution.ParametricDistribution


def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(
        params: types.PolicyParams, deterministic: bool = False, output_info: bool = False,
        use_sac_noise: bool = True,
    ) -> types.Policy:

        def policy(
            observations: types.Observation, key_sample: types.PRNGKey, dones: types.Observation = None, noise_scales: types.Observation = None
        ) -> Tuple[types.Action, types.Extra]:
            logits = sac_networks.policy_network.apply(*params, observations)
            mean = sac_networks.parametric_action_distribution.mode(logits)
            std = sac_networks.parametric_action_distribution.stddev(logits)
            if deterministic:
                return mean, {
                    'mean': mean,
                    'std': std
                }
            if use_sac_noise:
                action = sac_networks.parametric_action_distribution.sample(
                    logits, key_sample
                )
                if output_info:
                    return action, {
                        'mean': mean,
                        'std': std,
                    }
                return action, {}
            else:
                if dones is not None:
                    dones_2d = jnp.reshape(dones, (logits.shape[0], 1)).astype(bool)
                    # dones shape is (num_envs,1)
                    noise_scales = jnp.where(
                        dones_2d,
                        jax.random.uniform(
                            key_sample, shape=(logits.shape[0], 1)
                        ) * (0.4 - 0.001) + 0.001,
                        noise_scales
                    )
                action_mean = sac_networks.parametric_action_distribution.mode(logits)
                action = action_mean + noise_scales * jax.random.normal(key_sample, shape=action_mean.shape)

                if output_info:
                    return action, {
                        'noise_scales': noise_scales,
                    }
                else:
                    return action, {}

        return policy

    return make_policy


def make_q_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_q(
        params: types.PolicyParams,
    ) -> types.Policy:

        def q_network(
            observations: types.Observation, actions: types.Action
        ) -> Tuple[types.Action, types.Extra]:
            q = sac_networks.q_network.apply(*params, observations, actions)
            return q

        return q_network

    return make_q


_activations = {
    'swish': linen.swish,
    'tanh': linen.tanh,
    'relu': linen.relu,
}


def make_sac_networks(
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    policy_hidden_layer_sizes: Sequence[int] = (256, 256),
    q_hidden_layer_sizes: Sequence[int] = (256, 256),
    activation: str = 'relu',
    policy_network_layer_norm: bool = False,
    q_network_layer_norm: bool = False,
    distribution_type: Literal['normal', 'tanh_normal'] = 'tanh_normal',
    noise_std_type: Literal['scalar', 'log'] = 'scalar',
    init_noise_std: float = 1.0,
    state_dependent_std: bool = False,
    policy_obs_key: str = 'state',
    value_obs_key: str = 'privileged_state',
    compute_dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.float32,
    last_layer_in_fp32: bool = True,

) -> SACNetworks:
    """Make SAC networks."""
    parametric_action_distribution: distribution.ParametricDistribution
    if distribution_type == 'normal':
        parametric_action_distribution = distribution.NormalDistribution(
            event_size=action_size
        )
    elif distribution_type == 'tanh_normal':
        parametric_action_distribution = distribution.NormalTanhDistribution(
            event_size=action_size
        )
    else:
        raise ValueError(
            f'Unsupported distribution type: {distribution_type}. Must be one'
            ' of "normal" or "tanh_normal".'
        )
    activation = _activations[activation]

    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
        layer_norm=policy_network_layer_norm,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
        state_dependent_std=state_dependent_std,
        obs_key=policy_obs_key,
        dtype=compute_dtype,
        param_dtype=param_dtype,
        last_layer_in_fp32=last_layer_in_fp32,

    )
    q_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        hidden_layer_sizes=q_hidden_layer_sizes,
        activation=activation,
        layer_norm=q_network_layer_norm,
        value_obs_key=value_obs_key,
        dtype=compute_dtype,
        param_dtype=param_dtype,
        last_layer_in_fp32=last_layer_in_fp32,
    )
    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )
