import numpy as np
import jax
import jax.numpy as jnp
from mlxu.jax_utils import (
    JaxRNG,
    wrap_function_with_rng,
    init_rng,
    next_rng,
    collect_metrics,
)

import flax
from flax import linen as nn
import distrax
from functools import partial

from copy import copy, deepcopy
from queue import Queue
import threading

import d4rl

from flax.training.train_state import TrainState
import optax
import distrax
from ml_collections import ConfigDict
from collections import OrderedDict
from copy import deepcopy

import os
import time
from copy import deepcopy
import uuid

import pprint
import mlxu
import gym
from logger import logger, setup_logger


def extend_and_repeat(tensor, axis, repeat):
    return jnp.repeat(jnp.expand_dims(tensor, axis), repeat, axis=axis)


def mse_loss(val, target):
    return jnp.mean(jnp.square(val - target))


def value_and_multi_grad(fun, n_outputs, argnums=0, has_aux=False):
    def select_output(index):
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


@jax.jit
def batch_to_jax(batch):
    return jax.tree_util.tree_map(jax.device_put, batch)


def update_target_network(main_params, target_params, tau):
    return jax.tree_util.tree_map(
        lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
    )


def multiple_action_q_function(forward):
    # Forward the q function with multiple actions on each state, to be used as a decorator
    def wrapped(self, observations, actions, **kwargs):
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

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

    def __call__(self):
        return self.value


class FullyConnectedNetwork(nn.Module):
    output_dim: int
    arch: str = "256-256"
    orthogonal_init: bool = False

    @nn.compact
    def __call__(self, input_tensor):
        x = input_tensor
        hidden_sizes = [int(h) for h in self.arch.split("-")]
        for h in hidden_sizes:
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
    arch: str = "256-256"
    orthogonal_init: bool = False

    @nn.compact
    @multiple_action_q_function
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        x = FullyConnectedNetwork(
            output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init
        )(x)
        return jnp.squeeze(x, -1)

    @nn.nowrap
    def rng_keys(self):
        return ("params",)


class TanhGaussianPolicy(nn.Module):
    observation_dim: int
    action_dim: int
    arch: str = "256-256"
    orthogonal_init: bool = False
    log_std_multiplier: float = 1.0
    log_std_offset: float = -1.0

    def setup(self):
        self.base_network = FullyConnectedNetwork(
            output_dim=2 * self.action_dim,
            arch=self.arch,
            orthogonal_init=self.orthogonal_init,
        )
        self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
        self.log_std_offset_module = Scalar(self.log_std_offset)

    def log_prob(self, observations, actions):
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

    def __call__(self, observations, deterministic=False, repeat=None):
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
            samples, log_prob = action_distribution.sample_and_log_prob(
                seed=self.make_rng("noise")
            )

        return samples, log_prob

    @nn.nowrap
    def rng_keys(self):
        return ("params", "noise")


class SamplerPolicy(object):

    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(self, params, rng, observations, deterministic):
        return self.policy.apply(
            params,
            observations,
            deterministic,
            repeat=None,
            rngs=JaxRNG(rng)(self.policy.rng_keys()),
        )

    def __call__(self, observations, deterministic=False):
        actions, _ = self.act(
            self.params, next_rng(), observations, deterministic=deterministic
        )
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class ReplayBuffer(object):
    def __init__(self, max_size, data=None):
        self._max_size = max_size
        self._next_idx = 0
        self._size = 0
        self._initialized = False
        self._total_steps = 0

        if data is not None:
            if self._max_size < data["observations"].shape[0]:
                self._max_size = data["observations"].shape[0]
            self.add_batch(data)

    def __len__(self):
        return self._size

    def _init_storage(self, observation_dim, action_dim):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._observations = np.zeros(
            (self._max_size, observation_dim), dtype=np.float32
        )
        self._next_observations = np.zeros(
            (self._max_size, observation_dim), dtype=np.float32
        )
        self._actions = np.zeros((self._max_size, action_dim), dtype=np.float32)
        self._rewards = np.zeros(self._max_size, dtype=np.float32)
        self._dones = np.zeros(self._max_size, dtype=np.float32)
        self._next_idx = 0
        self._size = 0
        self._initialized = True

    def add_sample(self, observation, action, reward, next_observation, done):
        if not self._initialized:
            self._init_storage(observation.size, action.size)

        self._observations[self._next_idx, :] = np.array(observation, dtype=np.float32)
        self._next_observations[self._next_idx, :] = np.array(
            next_observation, dtype=np.float32
        )
        self._actions[self._next_idx, :] = np.array(action, dtype=np.float32)
        self._rewards[self._next_idx] = reward
        self._dones[self._next_idx] = float(done)

        if self._size < self._max_size:
            self._size += 1
        self._next_idx = (self._next_idx + 1) % self._max_size
        self._total_steps += 1

    def add_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(
            observations, actions, rewards, next_observations, dones
        ):
            self.add_sample(o, a, r, no, d)

    def add_batch(self, batch):
        self.add_traj(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
        )

    def sample(self, batch_size):
        indices = np.random.randint(len(self), size=batch_size)
        return self.select(indices)

    def select(self, indices):
        return dict(
            observations=self._observations[indices, ...],
            actions=self._actions[indices, ...],
            rewards=self._rewards[indices, ...],
            next_observations=self._next_observations[indices, ...],
            dones=self._dones[indices, ...],
        )

    def generator(self, batch_size, n_batchs=None):
        i = 0
        while n_batchs is None or i < n_batchs:
            yield self.sample(batch_size)
            i += 1

    @property
    def total_steps(self):
        return self._total_steps

    @property
    def data(self):
        return dict(
            observations=self._observations[: self._size, ...],
            actions=self._actions[: self._size, ...],
            rewards=self._rewards[: self._size, ...],
            next_observations=self._next_observations[: self._size, ...],
            dones=self._dones[: self._size, ...],
        )


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    return dict(
        observations=dataset["observations"],
        actions=dataset["actions"],
        next_observations=dataset["next_observations"],
        rewards=dataset["rewards"],
        dones=dataset["terminals"].astype(np.float32),
    )


def index_batch(batch, indices):
    indexed = {}
    for key in batch.keys():
        indexed[key] = batch[key][indices, ...]
    return indexed


def subsample_batch(batch, size):
    indices = np.random.randint(batch["observations"].shape[0], size=size)
    return index_batch(batch, indices)


class StepSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                observation.reshape(1, -1), deterministic=deterministic
            ).reshape(-1)
            next_observation, reward, done, _ = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):

    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env

    def sample(self, policy, n_trajs, deterministic=False, replay_buffer=None):
        trajs = []
        for _ in range(n_trajs):
            observations = []
            actions = []
            rewards = []
            next_observations = []
            dones = []

            observation = self.env.reset()

            for _ in range(self.max_traj_length):
                action = policy(
                    observation.reshape(1, -1), deterministic=deterministic
                ).reshape(-1)
                next_observation, reward, done, _ = self.env.step(action)
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_observations.append(next_observation)

                if replay_buffer is not None:
                    replay_buffer.add_sample(
                        observation, action, reward, next_observation, done
                    )

                observation = next_observation

                if done:
                    break

            trajs.append(
                dict(
                    observations=np.array(observations, dtype=np.float32),
                    actions=np.array(actions, dtype=np.float32),
                    rewards=np.array(rewards, dtype=np.float32),
                    next_observations=np.array(next_observations, dtype=np.float32),
                    dones=np.array(dones, dtype=np.float32),
                )
            )

        return trajs

    @property
    def env(self):
        return self._env


class ConservativeSAC(object):

    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.discount = 0.99
        config.alpha_multiplier = 1.0
        config.use_automatic_entropy_tuning = True
        config.backup_entropy = False
        config.target_entropy = 0.0
        config.policy_lr = 3e-4
        config.qf_lr = 3e-4
        config.optimizer_type = "adam"
        config.soft_target_update_rate = 5e-3
        config.use_cql = True
        config.cql_n_actions = 10
        config.cql_importance_sample = True
        config.cql_lagrange = False
        config.cql_target_action_gap = 1.0
        config.cql_temp = 1.0
        config.cql_min_q_weight = 5.0
        config.cql_max_target_backup = False
        config.cql_clip_diff_min = -np.inf
        config.cql_clip_diff_max = np.inf

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, policy, qf):
        self.config = self.get_default_config(config)
        self.policy = policy
        self.qf = qf
        self.observation_dim = policy.observation_dim
        self.action_dim = policy.action_dim

        self._train_states = {}

        optimizer_class = {
            "adam": optax.adam,
            "sgd": optax.sgd,
        }[self.config.optimizer_type]

        policy_params = self.policy.init(
            next_rng(self.policy.rng_keys()), jnp.zeros((10, self.observation_dim))
        )
        self._train_states["policy"] = TrainState.create(
            params=policy_params,
            tx=optimizer_class(self.config.policy_lr),
            apply_fn=None,
        )

        qf1_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )
        self._train_states["qf1"] = TrainState.create(
            params=qf1_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        qf2_params = self.qf.init(
            next_rng(self.qf.rng_keys()),
            jnp.zeros((10, self.observation_dim)),
            jnp.zeros((10, self.action_dim)),
        )
        self._train_states["qf2"] = TrainState.create(
            params=qf2_params,
            tx=optimizer_class(self.config.qf_lr),
            apply_fn=None,
        )
        self._target_qf_params = deepcopy({"qf1": qf1_params, "qf2": qf2_params})

        model_keys = ["policy", "qf1", "qf2"]

        if self.config.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self._train_states["log_alpha"] = TrainState.create(
                params=self.log_alpha.init(next_rng()),
                tx=optimizer_class(self.config.policy_lr),
                apply_fn=None,
            )
            model_keys.append("log_alpha")

        if self.config.cql_lagrange:
            self.log_alpha_prime = Scalar(1.0)
            self._train_states["log_alpha_prime"] = TrainState.create(
                params=self.log_alpha_prime.init(next_rng()),
                tx=optimizer_class(self.config.qf_lr),
                apply_fn=None,
            )
            model_keys.append("log_alpha_prime")

        self._model_keys = tuple(model_keys)
        self._total_steps = 0

    def train(self, batch, bc=False):
        self._total_steps += 1
        self._train_states, self._target_qf_params, metrics = self._train_step(
            self._train_states, self._target_qf_params, next_rng(), batch, bc
        )
        return metrics

    @partial(jax.jit, static_argnames=("self", "bc"))
    def _train_step(self, train_states, target_qf_params, rng, batch, bc=False):
        rng_generator = JaxRNG(rng)

        def loss_fn(train_params):
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_observations = batch["next_observations"]
            dones = batch["dones"]

            loss_collection = {}

            @wrap_function_with_rng(rng_generator())
            def forward_policy(rng, *args, **kwargs):
                return self.policy.apply(
                    *args, **kwargs, rngs=JaxRNG(rng)(self.policy.rng_keys())
                )

            @wrap_function_with_rng(rng_generator())
            def forward_qf(rng, *args, **kwargs):
                return self.qf.apply(
                    *args, **kwargs, rngs=JaxRNG(rng)(self.qf.rng_keys())
                )

            new_actions, log_pi = forward_policy(train_params["policy"], observations)

            if self.config.use_automatic_entropy_tuning:
                alpha_loss = (
                    -self.log_alpha.apply(train_params["log_alpha"])
                    * (log_pi + self.config.target_entropy).mean()
                )
                loss_collection["log_alpha"] = alpha_loss
                alpha = (
                    jnp.exp(self.log_alpha.apply(train_params["log_alpha"]))
                    * self.config.alpha_multiplier
                )
            else:
                alpha_loss = 0.0
                alpha = self.config.alpha_multiplier

            """ Policy loss """
            if bc:
                log_probs = forward_policy(
                    train_params["policy"],
                    observations,
                    actions,
                    method=self.policy.log_prob,
                )
                policy_loss = (alpha * log_pi - log_probs).mean()
            else:
                q_new_actions = jnp.minimum(
                    forward_qf(train_params["qf1"], observations, new_actions),
                    forward_qf(train_params["qf2"], observations, new_actions),
                )
                policy_loss = (alpha * log_pi - q_new_actions).mean()

            loss_collection["policy"] = policy_loss

            """ Q function loss """
            q1_pred = forward_qf(train_params["qf1"], observations, actions)
            q2_pred = forward_qf(train_params["qf2"], observations, actions)

            if self.config.cql_max_target_backup:
                new_next_actions, next_log_pi = forward_policy(
                    train_params["policy"],
                    next_observations,
                    repeat=self.config.cql_n_actions,
                )
                target_q_values = jnp.minimum(
                    forward_qf(
                        target_qf_params["qf1"], next_observations, new_next_actions
                    ),
                    forward_qf(
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
                new_next_actions, next_log_pi = forward_policy(
                    train_params["policy"], next_observations
                )
                target_q_values = jnp.minimum(
                    forward_qf(
                        target_qf_params["qf1"], next_observations, new_next_actions
                    ),
                    forward_qf(
                        target_qf_params["qf2"], next_observations, new_next_actions
                    ),
                )

            if self.config.backup_entropy:
                target_q_values = target_q_values - alpha * next_log_pi

            td_target = jax.lax.stop_gradient(
                rewards + (1.0 - dones) * self.config.discount * target_q_values
            )
            qf1_loss = mse_loss(q1_pred, td_target)
            qf2_loss = mse_loss(q2_pred, td_target)

            ### CQL
            if self.config.use_cql:
                batch_size = actions.shape[0]
                cql_random_actions = jax.random.uniform(
                    rng_generator(),
                    shape=(batch_size, self.config.cql_n_actions, self.action_dim),
                    minval=-1.0,
                    maxval=1.0,
                )

                cql_current_actions, cql_current_log_pis = forward_policy(
                    train_params["policy"],
                    observations,
                    repeat=self.config.cql_n_actions,
                )
                cql_next_actions, cql_next_log_pis = forward_policy(
                    train_params["policy"],
                    next_observations,
                    repeat=self.config.cql_n_actions,
                )

                cql_q1_rand = forward_qf(
                    train_params["qf1"], observations, cql_random_actions
                )
                cql_q2_rand = forward_qf(
                    train_params["qf2"], observations, cql_random_actions
                )
                cql_q1_current_actions = forward_qf(
                    train_params["qf1"], observations, cql_current_actions
                )
                cql_q2_current_actions = forward_qf(
                    train_params["qf2"], observations, cql_current_actions
                )
                cql_q1_next_actions = forward_qf(
                    train_params["qf1"], observations, cql_next_actions
                )
                cql_q2_next_actions = forward_qf(
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

                if self.config.cql_importance_sample:
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
                    jax.scipy.special.logsumexp(
                        cql_cat_q1 / self.config.cql_temp, axis=1
                    )
                    * self.config.cql_temp
                )
                cql_qf2_ood = (
                    jax.scipy.special.logsumexp(
                        cql_cat_q2 / self.config.cql_temp, axis=1
                    )
                    * self.config.cql_temp
                )

                """Subtract the log likelihood of data"""
                cql_qf1_diff = jnp.clip(
                    cql_qf1_ood - q1_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()
                cql_qf2_diff = jnp.clip(
                    cql_qf2_ood - q2_pred,
                    self.config.cql_clip_diff_min,
                    self.config.cql_clip_diff_max,
                ).mean()

                if self.config.cql_lagrange:
                    alpha_prime = jnp.clip(
                        jnp.exp(
                            self.log_alpha_prime.apply(train_params["log_alpha_prime"])
                        ),
                        a_min=0.0,
                        a_max=1000000.0,
                    )
                    cql_min_qf1_loss = (
                        alpha_prime
                        * self.config.cql_min_q_weight
                        * (cql_qf1_diff - self.config.cql_target_action_gap)
                    )
                    cql_min_qf2_loss = (
                        alpha_prime
                        * self.config.cql_min_q_weight
                        * (cql_qf2_diff - self.config.cql_target_action_gap)
                    )

                    alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5

                    loss_collection["log_alpha_prime"] = alpha_prime_loss

                else:
                    cql_min_qf1_loss = cql_qf1_diff * self.config.cql_min_q_weight
                    cql_min_qf2_loss = cql_qf2_diff * self.config.cql_min_q_weight
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
            self.config.soft_target_update_rate,
        )
        new_target_qf_params["qf2"] = update_target_network(
            new_train_states["qf2"].params,
            target_qf_params["qf2"],
            self.config.soft_target_update_rate,
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

        if self.config.use_cql:
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

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def total_steps(self):
        return self._total_steps


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    env="halfcheetah-medium-expert-v2",
    max_traj_length=1000,
    seed=42,
    save_model=False,
    batch_size=256,
    reward_scale=1.0,
    reward_bias=0.0,
    clip_action=0.999,
    policy_arch="256-256",
    qf_arch="256-256",
    orthogonal_init=False,
    policy_log_std_multiplier=1.0,
    policy_log_std_offset=-1.0,
    n_epochs=2000,
    bc_epochs=0,
    n_train_step_per_epoch=1000,
    eval_period=10,
    eval_n_trajs=5,
    cql=ConservativeSAC.get_default_config(),
    logging=mlxu.WandBLogger.get_default_config(),
)


def main(argv):
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    wandb_logger = mlxu.WandBLogger(config=FLAGS.logging, variant=variant)
    setup_logger(
        variant=variant,
        exp_id=wandb_logger.experiment_id,
        seed=FLAGS.seed,
        base_log_dir=FLAGS.logging.output_dir,
        include_exp_prefix_sub_dir=False,
    )

    mlxu.jax_utils.set_random_seed(FLAGS.seed)

    eval_sampler = TrajSampler(gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length)
    dataset = get_d4rl_dataset(eval_sampler.env)
    dataset["rewards"] = dataset["rewards"] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset["actions"] = np.clip(
        dataset["actions"], -FLAGS.clip_action, FLAGS.clip_action
    )

    observation_dim = eval_sampler.env.observation_space.shape[0]
    action_dim = eval_sampler.env.action_space.shape[0]

    policy = TanhGaussianPolicy(
        observation_dim,
        action_dim,
        FLAGS.policy_arch,
        FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier,
        FLAGS.policy_log_std_offset,
    )
    qf = FullyConnectedQFunction(
        observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init
    )

    if FLAGS.cql.target_entropy >= 0.0:
        FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape).item()

    sac = ConservativeSAC(FLAGS.cql, policy, qf)
    sampler_policy = SamplerPolicy(sac.policy, sac.train_params["policy"])

    viskit_metrics = {}
    for epoch in range(FLAGS.n_epochs):
        metrics = {"epoch": epoch}

        with mlxu.Timer() as train_timer:
            for batch_idx in range(FLAGS.n_train_step_per_epoch):
                batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
                metrics.update(
                    mlxu.prefix_metrics(
                        sac.train(batch, bc=epoch < FLAGS.bc_epochs), "sac"
                    )
                )

        with mlxu.Timer() as eval_timer:
            if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
                trajs = eval_sampler.sample(
                    sampler_policy.update_params(sac.train_params["policy"]),
                    FLAGS.eval_n_trajs,
                    deterministic=True,
                )

                metrics["average_return"] = np.mean(
                    [np.sum(t["rewards"]) for t in trajs]
                )
                metrics["average_traj_length"] = np.mean(
                    [len(t["rewards"]) for t in trajs]
                )
                metrics["average_normalizd_return"] = np.mean(
                    [
                        eval_sampler.env.get_normalized_score(np.sum(t["rewards"]))
                        for t in trajs
                    ]
                )
                if FLAGS.save_model:
                    save_data = {"sac": sac, "variant": variant, "epoch": epoch}
                    wandb_logger.save_pickle(save_data, "model.pkl")

        metrics["train_time"] = train_timer()
        metrics["eval_time"] = eval_timer()
        metrics["epoch_time"] = train_timer() + eval_timer()
        wandb_logger.log(metrics)
        viskit_metrics.update(metrics)
        logger.record_dict(viskit_metrics)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    if FLAGS.save_model:
        save_data = {"sac": sac, "variant": variant, "epoch": epoch}
        wandb_logger.save_pickle(save_data, "model.pkl")


if __name__ == "__main__":
    mlxu.run(main)
