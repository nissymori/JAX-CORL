# source https://github.com/young-geng/JaxCQL
# https://arxiv.org/abs/2006.04779
import os
import time
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple

import d4rl
import distrax
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

class CQLConfig(BaseModel):
    # GENERAL
    also: str = "CQL"
    project: str = "cql-jax"
    env_name: str = "halfcheetah-medium-expert-v2"
    max_traj_length: int = 1000
    seed: int = 42
    batch_size: int = 256
    n_jitted_updates: int = 8
    max_steps: int = 1000000
    eval_interval: int = 10000
    eval_episodes: int = 5
    normalize_state: bool = False
    data_size: int = 1000000
    # NETWORK
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False
    policy_log_std_multiplier: float = 1.0
    policy_log_std_offset: float = -1.0
    # CQL SPECIFIC
    discount: float = 0.99
    alpha_multiplier: float = 1.0
    use_automatic_entropy_tuning: bool = True
    backup_entropy: bool = False
    target_entropy: float = 0.0
    policy_lr: float = 3e-4
    qf_lr: float = 3e-4
    optimizer_type: str = "adam"
    soft_target_update_rate: float = 5e-3
    use_cql: bool = True
    cql_n_actions: int = 10
    cql_importance_sample: bool = True
    cql_lagrange: bool = False
    cql_target_action_gap: float = 1.0
    cql_temp: float = 1.0
    cql_min_q_weight: float = 5.0
    cql_max_target_backup: bool = False
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf

    def __hash__(self):
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = CQLConfig(**conf_dict)


def extend_and_repeat(tensor: jnp.ndarray, axis: int, repeat: int) -> jnp.ndarray:
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.square(val - target))


def value_and_multi_grad(
    fun: Callable, n_outputs: int, argnums=0, has_aux=False
) -> Callable:
    def select_output(index: int) -> Callable:
        def wrapped(*args, **kwargs):
            if has_aux:
                x, *aux = fun(*args, **kwargs)
                return (x[index], *aux)
            else:
                x = fun(*args, **kwargs)
                return x[index]

        return wrapped

    grad_fns = tuple(
        jax.value_and_grad(select_output(i), argnums=argnums, has_aux=has_aux)
        for i in range(n_outputs)
    )

    def multi_grad_fn(*args, **kwargs):
        grads = []
        values = []
        for grad_fn in grad_fns:
            (value, *aux), grad = grad_fn(*args, **kwargs)
            values.append(value)
            grads.append(grad)
        return (tuple(values), *aux), tuple(grads)

    return multi_grad_fn


def update_target_network(main_params: Any, target_params: Any, tau: float) -> Any:
    return jax.tree_util.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


def multiple_action_q_function(forward: Callable) -> Callable:
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(
        self, observations: jnp.ndarray, actions: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1]).reshape(
                -1, observations.shape[-1]
            )
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


class Scalar(nn.Module):
    init_value: float

    def setup(self) -> None:
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self) -> jnp.ndarray:
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, input_tensor: jnp.ndarray) -> jnp.ndarray:
        x = input_tensor
        for h in self.hidden_dims:
            if self.orthogonal_init:
                x = nn.Dense(
                    h,
                    kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
                    bias_init=jax.nn.initializers.zeros,
                )(x)
            else:
                x = nn.Dense(h)(x)
            x = nn.relu(x)

        if self.orthogonal_init:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.orthogonal(1e-2),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        else:
            output = nn.Dense(
                self.output_dim,
                kernel_init=jax.nn.initializers.variance_scaling(
                    1e-2, "fan_in", "uniform"
                ),
                bias_init=jax.nn.initializers.zeros,
            )(x)
        return output


class FullyConnectedQFunction(nn.Module):
    observation_dim: int
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], axis=-1)
        x = FullyConnectedNetwork(
            output_dim=1,
            hidden_dims=self.hidden_dims,
            orthogonal_init=self.orthogonal_init,
        )(x)
        return jnp.squeeze(x, -1)


class TanhGaussianPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    hidden_dims: Tuple[int] = (256, 256)
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0

    def setup(self) -> None:
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim,
            hidden_dims=self.hidden_dims,
            orthogonal_init=self.orthogonal_init,
        )
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)

    def log_prob(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        if actions.ndim == 3:
            observations = extend_and_repeat(observations, 1, actions.shape[1])
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = (
            self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        )
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        return action_distribution.log_prob(actions)

    def __call__(
        self,
        observations: jnp.ndarray,
        rng: jax.random.PRNGKey,
        deterministic=False,
        repeat=None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if repeat is not None:
            observations = extend_and_repeat(observations, 1, repeat)
        base_network_output = self.base_network(observations)
        mean, log_std = jnp.split(base_network_output, 2, axis=-1)
        log_std = (
            self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
        )
        log_std = jnp.clip(log_std, -20.0, 2.0)
        action_distribution = distrax.Transformed(
            distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
            distrax.Block(distrax.Tanh(), ndims=1),
        )
        if deterministic:
            samples = jnp.tanh(mean)
            log_prob = action_distribution.log_prob(samples)
        else:
            samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

        return samples, log_prob


class Transition(NamedTuple):
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


def get_dataset(
    env: gym.Env, config: CQLConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dataset["terminals"], dtype=jnp.float32),
    )
    # shuffle data and select the first data_size samples
    data_size = min(config.data_size, len(dataset.observations))
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_util.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_util.tree_map(lambda x: x[:data_size], dataset)
    # normalize states
    obs_mean, obs_std = 0, 1
    if config.normalize_state:
        obs_mean = dataset.observations.mean(0)
        obs_std = dataset.observations.std(0)
        dataset = dataset._replace(
            observations=(dataset.observations - obs_mean) / (obs_std + 1e-5),
            next_observations=(dataset.next_observations - obs_mean) / (obs_std + 1e-5),
        )
    return dataset, obs_mean, obs_std


def collect_metrics(metrics, names, prefix=None):
    collected = {}
    for name in names:
        if name in metrics:
            collected[name] = jnp.mean(metrics[name])
    if prefix is not None:
        collected = {
            "{}/{}".format(prefix, key): value for key, value in collected.items()
        }
    return collected


class CQLTrainer(object):

    def __init__(self, config, policy, qf, rng):
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "sgd": optax.sgd,
        }[config.optimizer_type]

        rng, policy_rng, q1_rng, q2_rng = jax.random.split(rng, 4)

        policy_params = self.policy.init(
            policy_rng, jnp.zeros((10, self.observation_dim)), policy_rng
        )
        self._train_states["policy"] = TrainState.create(
            params=policy_params, tx=optimizer_class(config.policy_lr), apply_fn=None
        )

        qf1_params = self.qf.init(
            q1_rng,
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )
        self._train_states["qf1"] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(config.qf_lr),
            apply_fn=None,
        )
        qf2_params = self.qf.init(
            q2_rng,
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )
        self._train_states["qf2"] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(config.qf_lr),
            apply_fn=None,
        )
        self._target_qf_params = deepcopy({"qf1": qf1_params, "qf2": qf2_params})

        model_keys = ["policy", "qf1", "qf2"]

        if config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            rng, log_alpha_rng = jax.random.split(rng)
            self._train_states["log_alpha"] = TrainState.create(
                params=self.log_alpha.init(log_alpha_rng),
                tx=optimizer_class(config.policy_lr),
                apply_fn=None,
            )
            model_keys.append("log_alpha")

        if config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            rng, log_alpha_prime_rng = jax.random.split(rng)
            self._train_states["log_alpha_prime"] = TrainState.create(
                params=self.log_alpha_prime.init(log_alpha_prime_rng),
                tx=optimizer_class(config.qf_lr),
                apply_fn=None,
            )
            model_keys.append("log_alpha_prime")

        self._model_keys = tuple(model_keys)

    @partial(jax.jit, static_argnames=("self", "config", "bc"))
    def train(self, train_states, target_q_params, dataset, rng, config, bc=False):
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, update_rng = jax.random.split(rng, 3)
            batch_indices = jax.random.randint(
                batch_rng, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)
            train_states, target_q_params, metrics = self._train_step(
                train_states, target_q_params, update_rng, batch, config, bc
            )
        return train_states, target_q_params, metrics

    def _train_step(
        self, train_states, target_qf_params, _rng, batch, config, bc=False
    ):

        def loss_fn(train_params):
            observations = batch.observations
            actions = batch.actions
            rewards = batch.rewards
            next_observations = batch.next_observations
            dones = batch.dones

            loss_collection = {}

            rng, new_actions_rng = jax.random.split(_rng)
            new_actions, log_pi = self.policy.apply(
                train_params["policy"], observations, new_actions_rng
            )

            if config.use_automatic_entropy_tuning:
                alpha_loss = (
                    -self.log_alpha.apply(train_params["log_alpha"])
                    * (log_pi + config.target_entropy).mean()
                )
                loss_collection["log_alpha"] = alpha_loss
                alpha = (
                    jnp.exp(self.log_alpha.apply(train_params["log_alpha"]))
                    * config.alpha_multiplier
                )
            else:
                alpha_loss = 0.0
                alpha = config.alpha_multiplier

            """ Policy loss """
            if bc:
                rng, bc_rng = jax.random.split(rng)
                log_probs = self.policy.apply(
                    train_params["policy"],
                    observations,
                    actions,
                    bc_rng,
                    method=self.policy.log_prob,
                )
                policy_loss = (alpha * log_pi - log_probs).mean()
            else:
                q_new_actions = jnp.minimum(
                    self.qf.apply(train_params["qf1"], observations, new_actions),
                    self.qf.apply(train_params["qf2"], observations, new_actions),
                )
                policy_loss = (alpha * log_pi - q_new_actions).mean()

            loss_collection["policy"] = policy_loss

            """ Q function loss """
            q1_pred = self.qf.apply(train_params["qf1"], observations, actions)
            q2_pred = self.qf.apply(train_params["qf2"], observations, actions)

            if config.cql_max_target_backup:
                rng, cql_rng = jax.random.split(rng)
                new_next_actions, next_log_pi = self.policy.apply(
                    train_params["policy"],
                    next_observations,
                    cql_rng,
                    repeat=config.cql_n_actions,
                )
                target_q_values = jnp.minimum(
                    self.qf.apply(
                        target_qf_params["qf1"], next_observations, new_next_actions
                    ),
                    self.qf.apply(
                        target_qf_params["qf2"], next_observations, new_next_actions
                    ),
                )
                max_target_indices = jnp.expand_dims(
                    jnp.argmax(target_q_values, axis=-1), axis=-1
                )
                target_q_values = jnp.take_along_axis(
                    target_q_values, max_target_indices, axis=-1
                ).squeeze(-1)
                next_log_pi = jnp.take_along_axis(
                    next_log_pi, max_target_indices, axis=-1
                ).squeeze(-1)
            else:
                rng, cql_rng = jax.random.split(rng)
                new_next_actions, next_log_pi = self.policy.apply(
                    train_params["policy"], next_observations, cql_rng
                )
                target_q_values = jnp.minimum(
                    self.qf.apply(
                        target_qf_params["qf1"], next_observations, new_next_actions
                    ),
                    self.qf.apply(
                        target_qf_params["qf2"], next_observations, new_next_actions
                    ),
                )

            if config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = jax.lax.stop_gradient(
                rewards + (1.0 - dones) * config.discount * target_q_values
            )
            qf1_loss = mse_loss(q1_pred, td_target)
            qf2_loss = mse_loss(q2_pred, td_target)

            ### CQL
            if config.use_cql:
                batch_size = actions.shape[0]
                rng, random_rng = jax.random.split(rng)
                cql_random_actions = jax.random.uniform(
                    random_rng,
                    shape=(batch_size, config.cql_n_actions, self.action_dim),
                    minval=-1.0,
                    maxval=1.0,
                )
                rng, current_rng = jax.random.split(rng)
                cql_current_actions, cql_current_log_pis = self.policy.apply(
                    train_params["policy"],
                    observations,
                    current_rng,
                    repeat=config.cql_n_actions,
                )
                rng, next_rng = jax.random.split(rng)
                cql_next_actions, cql_next_log_pis = self.policy.apply(
                    train_params["policy"],
                    next_observations,
                    next_rng,
                    repeat=config.cql_n_actions,
                )

                cql_q1_rand = self.qf.apply(
                    train_params["qf1"], observations, cql_random_actions
                )
                cql_q2_rand = self.qf.apply(
                    train_params["qf2"], observations, cql_random_actions
                )
                cql_q1_current_actions = self.qf.apply(
                    train_params["qf1"], observations, cql_current_actions
                )
                cql_q2_current_actions = self.qf.apply(
                    train_params["qf2"], observations, cql_current_actions
                )
                cql_q1_next_actions = self.qf.apply(
                    train_params["qf1"], observations, cql_next_actions
                )
                cql_q2_next_actions = self.qf.apply(
                    train_params["qf2"], observations, cql_next_actions
                )

                cql_cat_q1 = jnp.concatenate(
                    [
                        cql_q1_rand,
                        jnp.expand_dims(q1_pred, 1),
                        cql_q1_next_actions,
                        cql_q1_current_actions,
                    ],
                    axis=1,
                )
                cql_cat_q2 = jnp.concatenate(
                    [
                        cql_q2_rand,
                        jnp.expand_dims(q2_pred, 1),
                        cql_q2_next_actions,
                        cql_q2_current_actions,
                    ],
                    axis=1,
                )
                cql_std_q1 = jnp.std(cql_cat_q1, axis=1)
                cql_std_q2 = jnp.std(cql_cat_q2, axis=1)

                if config.cql_importance_sample:
                    random_density = np.log(0.5**self.action_dim)
                    cql_cat_q1 = jnp.concatenate(
                        [
                            cql_q1_rand - random_density,
                            cql_q1_next_actions - cql_next_log_pis,
                            cql_q1_current_actions - cql_current_log_pis,
                        ],
                        axis=1,
                    )
                    cql_cat_q2 = jnp.concatenate(
                        [
                            cql_q2_rand - random_density,
                            cql_q2_next_actions - cql_next_log_pis,
                            cql_q2_current_actions - cql_current_log_pis,
                        ],
                        axis=1,
                    )

                cql_qf1_ood = (
                    jax.scipy.special.logsumexp(cql_cat_q1 / config.cql_temp, axis=1)
                    * config.cql_temp
                )
                cql_qf2_ood = (
                    jax.scipy.special.logsumexp(cql_cat_q2 / config.cql_temp, axis=1)
                    * config.cql_temp
                )

                """Subtract the log likelihood of data"""
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred,
                    config.cql_clip_diff_min,
                    config.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred,
                    config.cql_clip_diff_min,
                    config.cql_clip_diff_max,
                ).mean()

                if config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(
                            self.log_alpha_prime.apply(train_params["log_alpha_prime"])
                        ),
                        a_min=0.0,
                        a_max=1000000.0,
                    )
                    cql_min_qf1_loss = (
                        alpha_prime
                        * config.cql_min_q_weight
                        * (cql_qf1_diff - config.cql_target_action_gap)
                    )
                    cql_min_qf2_loss = (
                        alpha_prime
                        * config.cql_min_q_weight
                        * (cql_qf2_diff - config.cql_target_action_gap)
                    )

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5

                    loss_collection["log_alpha_prime"] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * config.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * config.cql_min_q_weight
                    alpha_prime_loss = 0.0
                    alpha_prime = 0.0

                qf1_loss = qf1_loss + cql_min_qf1_loss
                qf2_loss = qf2_loss + cql_min_qf2_loss

            loss_collection["qf1"] = qf1_loss
            loss_collection["qf2"] = qf2_loss
            return tuple(loss_collection[key] for key in self.model_keys), locals()

        train_params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_values), grads = value_and_multi_grad(
            loss_fn, len(self.model_keys), has_aux=True
        )(train_params)

        new_train_states = {
            key: train_states[key].apply_gradients(grads=grads[i][key])
            for i, key in enumerate(self.model_keys)
        }
        new_target_qf_params = {}
        new_target_qf_params["qf1"] = update_target_network(
            new_train_states["qf1"].params,
            target_qf_params["qf1"],
            config.soft_target_update_rate,
        )
        new_target_qf_params["qf2"] = update_target_network(
            new_train_states["qf2"].params,
            target_qf_params["qf2"],
            config.soft_target_update_rate,
        )

        metrics = collect_metrics(
            aux_values,
            [
                "log_pi",
                "policy_loss",
                "qf1_loss",
                "qf2_loss",
                "alpha_loss",
                "alpha",
                "q1_pred",
                "q2_pred",
                "target_q_values",
            ],
        )

        if config.use_cql:
            metrics.update(
                collect_metrics(
                    aux_values,
                    [
                        "cql_std_q1",
                        "cql_std_q2",
                        "cql_q1_rand",
                        "cql_q2_rand" "cql_qf1_diff",
                        "cql_qf2_diff",
                        "cql_min_qf1_loss",
                        "cql_min_qf2_loss",
                        "cql_q1_current_actions",
                        "cql_q2_current_actions" "cql_q1_next_actions",
                        "cql_q2_next_actions",
                        "alpha_prime",
                        "alpha_prime_loss",
                    ],
                    "cql",
                )
            )

        return new_train_states, new_target_qf_params, metrics

    @partial(jax.jit, static_argnames=("self",))
    def get_actions(self, train_states, obs):
        action, _ = self.policy.apply(
            train_states["policy"].params,
            obs.reshape(1, -1),
            jax.random.PRNGKey(0),
            deterministic=True,
        )
        return action.squeeze(0)

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}


def create_trainer(
    observations: jnp.ndarray, actions: jnp.ndarray, config: CQLConfig
) -> CQLTrainer:
    policy = TanhGaussianPolicy(
        observation_dim=observations.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=config.hidden_dims,
        orthogonal_init=config.orthogonal_init,
        log_std_multiplier=config.policy_log_std_multiplier,
        log_std_offset=config.policy_log_std_offset,
    )
    qf = FullyConnectedQFunction(
        observation_dim=observations.shape[-1],
        action_dim=actions.shape[-1],
        hidden_dims=config.hidden_dims,
        orthogonal_init=config.orthogonal_init,
    )
    rng = jax.random.PRNGKey(config.seed)
    return CQLTrainer(config, policy, qf, rng)


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env: gym.Env,
    num_episodes: int,
    obs_mean=0,
    obs_std=1,
):
    episode_returns = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs = (obs - obs_mean) / obs_std
            action = policy_fn(obs=obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        episode_returns.append(total_reward)
    return env.get_normalized_score(np.mean(episode_returns)) * 100


if __name__ == "__main__":
    wandb.init(project=config.project, config=config)
    rng = jax.random.PRNGKey(config.seed)
    env = gym.make(config.env_name)
    dataset, obs_mean, obs_std = get_dataset(env, config)

    if config.target_entropy >= 0.0:
        config.target_entropy = -np.prod(env.action_space.shape).item()

    sac = create_trainer(dataset.observations, dataset.actions, config)
    train_states, target_qf_params = sac._train_states, sac._target_qf_params

    num_steps = int(config.max_steps // config.n_jitted_updates)
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        metrics = {"step": i}
        rng, update_rng = jax.random.split(rng)
        train_states, target_qf_params, metrics = sac.train(
            train_states, target_qf_params, dataset, update_rng, config, bc=False
        )
        metrics.update(metrics)

        if i == 0 or (i + 1) % config.eval_interval == 0:
            policy_fn = partial(sac.get_actions, train_states=train_states)
            normalized_score = evaluate(
                policy_fn, env, config.eval_episodes, obs_mean=0, obs_std=1
            )
            metrics[f"{config.env_name}/normalized_score"] = normalized_score
            print(config.env_name, i, metrics[f"{config.env_name}/normalized_score"])
        wandb.log(metrics)

    # final evaluation
    policy_fn = partial(sac.get_actions, train_states=train_states)
    normalized_score = evaluate(
        policy_fn, env, config.eval_episodes, obs_mean=0, obs_std=1
    )
    wandb.log({f"{config.env_name}/finel_normalized_score": normalized_score})
    print(config.env_name, i, normalized_score)
    wandb.finish()
