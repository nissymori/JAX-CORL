import time
from collections import defaultdict
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import d4rl
import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax import struct
from flax.training import update_state
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors


class TD3BCConfig:
    # general config
    env_name: str = "Hopper"
    data_quality: str = "medium-expert"
    train_steps: int = 1000000
    evaluate_every_epochs: int = 100000
    num_test_rollouts: int = 5
    batch_size: int = 256
    buffer_size: int = 1000000
    seed: int = 0
    # network config
    num_hidden_layers: int = 2
    num_hidden_units: int = 256
    gamma: float = 0.99
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    # TD3-BC specific
    policy_freq: int = 2
    polyak: float = 0.995
    alpha: float = 2.5
    policy_noise_std: float = 0.2
    policy_noise_clip: float = 0.5


config = TD3BCConfig(**OmegaConf.to_object(OmegaConf.from_cli()))

env = gym.make(f"{config.env_name.lower()}-{config.data_quality}-v2")
dataset = d4rl.qlearning_dataset(env)
obs_dim = dataset["observations"].shape[-1]
action_space = env.action_space
act_dim = action_space.shape[0]
max_action = action_space.high


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class ReplayBuffer(NamedTuple):
    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


class DoubleCritic(nn.Module):
    @nn.compact
    def __call__(self, state, action, rng):
        sa = jnp.concatenate([state, action], axis=-1)
        x_q = nn.Dense(config.num_hidden_units, kernel_init=default_init())(sa)
        x_q = nn.LayerNorm()(x_q)
        x_q = nn.relu(x_q)
        for i in range(1, self.num_hidden_layers):
            x_q = nn.Dense(config.num_hidden_units, kernel_init=default_init())(x_q)
            x_q = nn.LayerNorm()(x_q)
            x_q = nn.relu(x_q)
        q1 = nn.Dense(1, kernel_init=default_init())(x_q)

        x_q = nn.Dense(self.num_hidden_units, kernel_init=default_init())(sa)
        x_q = nn.LayerNorm()(x_q)
        x_q = nn.relu(x_q)
        for i in range(1, self.num_hidden_layers):
            x_q = nn.Dense(self.num_hidden_units, kernel_init=default_init())(x_q)
            x_q = nn.LayerNorm()(x_q)
            x_q = nn.relu(x_q)
        q2 = nn.Dense(
            1,
        )(x_q)
        return q1, q2


class TD3Actor(nn.Module):
    action_dim: int
    max_action: float

    @nn.compact
    def __call__(self, state, rng):
        x_a = nn.relu(
            nn.Dense(config.num_hidden_units, kernel_init=default_init())(state)
        )
        for i in range(1, config.num_hidden_layers):
            x_a = nn.Dense(self.num_hidden_units, kernel_init=default_init())(x_a)
            x_a = nn.relu(x_a)
        action = nn.Dense(config.action_dim, kernel_init=default_init())(x_a)
        action = self.max_action * jnp.tanh(action)
        return action


class TD3BCUpdateState(NamedTuple):
    critic: TrainState
    actor: TrainState
    critic_params_target: flax.core.FrozenDict
    actor_params_target: flax.core.FrozenDict
    update_idx: jnp.int32


def get_models(
    rng: jax.random.PRNGKey,
) -> TD3BCUpdateState:
    critic_model = DoubleCritic(
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_units=config.num_hidden_units,
    )
    actor_model = TD3Actor(
        act_dim,
        num_hidden_layers=config.num_hidden_layers,
        num_hidden_units=config.num_hidden_units,
        max_action=max_action,
    )
    # Initialize the network based on the observation shape
    rng, rng1, rng2 = jax.random.split(rng, 3)
    critic_params = critic_model.init(
        rng1, state=jnp.zeros(obs_dim), action=jnp.zeros(act_dim), rng=rng1
    )
    critic_params_target = critic_model.init(
        rng1, jnp.zeros(obs_dim), jnp.zeros(act_dim), rng=rng1
    )
    actor_params = actor_model.init(rng2, jnp.zeros(obs_dim), rng=rng2)
    actor_params_target = actor_model.init(rng2, jnp.zeros(obs_dim), rng=rng2)

    critic_train_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(config.critic_lr),
    )
    actor_train_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )
    return TD3BCUpdateState(
        critic=critic_train_state,
        actor=actor_train_state,
        critic_params_target=critic_params_target,
        actor_params_target=actor_params_target,
        update_idx=0,
    )


def update_actor(
    update_state: TD3BCUpdateState, batch: ReplayBuffer, rng_key: jax.random.PRNGKey
):
    actor, critic = update_state.actor, update_state.critic
    obs, action, reward, next_obs, done = batch

    def get_actor_loss(
        actor_params: flax.core.frozen_dict.FrozenDict,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        predicted_action = actor.apply_fn(actor_params, obs, rng=None)
        critic_params = jax.lax.stop_gradient(train_state.critic.params)
        q_value, _ = critic.apply_fn(
            critic_params, obs, predicted_action, rng=None
        )  # todo this will also affect the critic update :/

        mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
        loss_lambda = config.alpha / mean_abs_q

        bc_loss = jnp.square(predicted_action - action).mean()
        loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
        return loss_actor, (
            bc_loss,
            -1.0 * q_value.mean() * loss_lambda,
        )

    actor_grad_fn = jax.value_and_grad(get_actor_loss, has_aux=True)
    actor_loss, actor_grads = actor_grad_fn(actor.params)
    actor = actor.apply_gradients(grads=actor_grads)
    return update_state._replace(actor=actor)


def update_critic(
    update_state: TD3BCUpdateState, batch: ReplayBuffer, rng_key: jax.random.PRNGKey
):
    actor, critic = update_state.actor, update_state.critic
    obs, action, reward, next_obs, done = batch

    def critic_loss(
        critic_params: flax.core.frozen_dict.FrozenDict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q_pred_1, q_pred_2 = critic.apply_fn(critic_params, obs, action, rng=None)

        target_next_action = actor.apply_fn(
            update_state.actor_params_target, next_obs, rng=None
        )
        policy_noise = (
            config.policy_noise_std
            * max_action
            * jax.random.normal(rng_key, action.shape)
        )
        target_next_action = target_next_action + policy_noise.clip(
            -config.policy_noise_clip, config.policy_noise_clip
        )
        target_next_action = target_next_action.clip(-max_action, max_action)
        q_next_1, q_next_2 = critic.apply_fn(
            update_state.critic_params_target,
            next_obs,
            target_next_action,
            rng=None,
        )
        target = reward[..., None] + config.gamma * jnp.minimum(q_next_1, q_next_2) * (
            1 - done[..., None]
        )
        target = jax.lax.stop_gradient(target)
        value_loss_1 = jnp.square(q_pred_1 - target)
        value_loss_2 = jnp.square(q_pred_2 - target)
        value_loss = (value_loss_1 + value_loss_2).mean()
        return value_loss, (
            value_loss_1.mean(),
            value_loss_2.mean(),
            target.mean(),
        )

    critic_grad_fn = jax.value_and_grad(critic_loss, has_aux=True)
    critic_loss, critic_grads = critic_grad_fn(critic.params)
    critic = critic.apply_gradients(grads=critic_grads)
    return update_state._replace(critic=critic)


def make_update_steps_fn(
    buffer: ReplayBuffer,
    num_existing_samples: int,
    update_steps: int,
    max_action: float,
    action_dim: int,
    config: TD3BCConfig,
) -> Tuple[dict, TD3BCUpdateState]:
    def update_steps_fn(
        update_state: TD3BCUpdateState,
        rng: jax.random.PRNGKey,
    ):
        def update_step_fn(
            update_state: TD3BCUpdateState,
            rng: jax.random.PRNGKey,
        ):
            rng, subkey = jax.random.split(rng)
            batch = sample_batch(buffer, num_existing_samples, config, subkey)
            rng, subkey = jax.random.split(rng)
            update_state = update_critic(train_state, batch, subkey)

            new_train_state = update_actor(train_state, batch, subkey)
            update_state = jax.lax.cond(
                update_state.update_idx % config.policy_freq == 0,
                lambda: new_train_state,
                lambda: update_state,
            )
            # update target network
            critic_params_target = jax.tree_map(
                lambda target, live: config.polyak * target
                + (1.0 - config.polyak) * live,
                update_state.critic_params_target,
                critic.params,
            )
            actor_params_target = jax.tree_map(
                lambda target, live: config.polyak * target
                + (1.0 - config.polyak) * live,
                update_state.actor_params_target,
                actor.params,
            )
            update_state = update_state._replace(
                critic_params_target=critic_params_target,
                actor_params_target=actor_params_target,
                update_idx=train_state.update_idx + 1,
            )
            return update_state, None

        rng_keys = jax.random.split(rng, update_steps)
        update_state, _ = jax.lax.scan(update_step_fn, update_state, rng_keys)
        return update_state

    return update_steps_fn


@partial(jax.jit)
def get_action(
    actor_train_state: TrainState,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    action = actor_train_state.apply_fn(actor_train_state.params, obs, rng=None)
    action = action.clip(action_space.low, action_space.high)
    return action


@partial(jax.jit, static_argnames=("config"))
def sample_batch(buffer, num_existing_samples, config, rng):
    idxes = jax.random.randint(rng, (config.batch_size,), 0, num_existing_samples)
    obs = buffer.states[idxes]
    action = buffer.actions[idxes]
    reward = buffer.rewards[idxes]
    next_obs = buffer.next_states[idxes]
    done = buffer.dones[idxes]
    return (obs, action, reward, next_obs, done)


def eval_d4rl(
    subkey: jax.random.PRNGKey,
    actor_trainstate: TrainState,
    env: gym.Env,
    obs_mean,
    obs_std,
) -> float:
    episode_rews = []
    for _ in range(config.num_test_rollouts):
        obs = env.reset()
        done = False
        episode_rew = 0.0
        while not done:
            obs = jnp.array((obs - obs_mean) / obs_std)
            action = get_action(actor_train_state=actor_trainstate, obs=obs)
            action = action.reshape(-1)
            obs, rew, done, info = env.step(action)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return np.mean(episode_rews)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(config.seed)
    wandb.init(project="train-TD3-BC", config=config)
    buffer = ReplayBuffer(
        states=jnp.asarray(dataset["observations"]),
        actions=jnp.asarray(dataset["actions"]),
        next_states=jnp.asarray(dataset["next_observations"]),
        rewards=jnp.asarray(dataset["rewards"]),
        dones=jnp.asarray(dataset["terminals"]),
    )
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(buffer.states))
    buffer = jax.tree_map(lambda x: x[perm], buffer)

    buffer_idx = jax.random.randint(
        rng_select, (config.buffer_size,), 0, len(buffer.states)
    )
    buffer = jax.tree_map(lambda x: x[buffer_idx], buffer)
    obs_mean = np.mean(buffer.states, axis=0)
    obs_std = np.std(buffer.states, axis=0)
    buffer = buffer._replace(
        states=(buffer.states - obs_mean) / obs_std,
        next_states=(buffer.next_states - obs_mean) / obs_std,
    )
    update_state = get_models(rng)

    total_steps = 0
    num_total_its = int(config.train_steps) // config.evaluate_every_epochs

    update_steps_fn = make_update_steps_fn(
        buffer,
        len(buffer.states),
        config.evaluate_every_epochs,
        max_action,
        act_dim,
        config,
    )
    jit_update_steps_fn = jax.jit(update_steps_fn)

    for it in tqdm(range(num_total_its)):
        total_steps += 1
        rng, rng_eval, rng_update = jax.random.split(rng, 3)
        update_state = jit_update_steps_fn(train_state, rng_update)
        eval_dict = {}
        eval_reward = eval_d4rl(
            rng_eval,
            update_state.actor,
            env,
            obs_mean,
            obs_std,
        )
        eval_rew_normed = env.get_normalized_score(eval_reward) * 100
        eval_dict[f"offline/eval_rew_normed_{config.env_name}"] = eval_rew_normed
        eval_dict[f"offline/step"] = total_steps
        print(eval_dict)
        wandb.log(eval_dict)
    wandb.finish()
