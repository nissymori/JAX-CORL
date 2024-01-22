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
from flax.training import train_state
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
    td3_alpha: float = 2.5
    td3_policy_noise_std: float = 0.2
    td3_policy_noise_clip: float = 0.5


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
    num_hidden_units: int
    num_hidden_layers: int

    @nn.compact
    def __call__(self, state, action, rng):
        sa = jnp.concatenate([state, action], axis=-1)
        x_q = nn.Dense(
            self.num_hidden_units,
            kernel_init=default_init(),
        )(sa)
        x_q = nn.LayerNorm()(x_q)
        x_q = nn.relu(x_q)
        for i in range(1, self.num_hidden_layers):
            x_q = nn.Dense(
                self.num_hidden_units,
                kernel_init=default_init(),
            )(x_q)
            x_q = nn.LayerNorm()(x_q)
            x_q = nn.relu(x_q)
        q1 = nn.Dense(
            1,
            kernel_init=default_init(),
        )(x_q)

        x_q = nn.Dense(
            self.num_hidden_units,
            kernel_init=default_init(),
        )(sa)
        x_q = nn.LayerNorm()(x_q)
        x_q = nn.relu(x_q)
        for i in range(1, self.num_hidden_layers):
            x_q = nn.Dense(
                self.num_hidden_units,
                kernel_init=default_init(),
            )(x_q)
            x_q = nn.LayerNorm()(x_q)
            x_q = nn.relu(x_q)
        q2 = nn.Dense(
            1,
        )(x_q)
        return q1, q2


class TD3Actor(nn.Module):
    action_dim: int
    num_hidden_units: int
    num_hidden_layers: int
    max_action: float

    @nn.compact
    def __call__(self, state, rng):
        x_a = nn.relu(
            nn.Dense(
                self.num_hidden_units,
                kernel_init=default_init(),
            )(state)
        )
        for i in range(1, self.num_hidden_layers):
            x_a = nn.relu(
                nn.Dense(
                    self.num_hidden_units,
                    kernel_init=default_init(),
                )(x_a)
            )
        action = nn.Dense(
            self.action_dim,
            kernel_init=default_init(),
        )(x_a)
        action = self.max_action * jnp.tanh(action)
        return action


@struct.dataclass
class TD3BCTrainState:
    critic: TrainState
    actor: TrainState
    critic_params_target: flax.core.FrozenDict
    actor_params_target: flax.core.FrozenDict
    update_idx: jnp.int32


def get_models(
    rng: jax.random.PRNGKey,
) -> TD3BCTrainState:
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
    return TD3BCTrainState(
        critic=critic_train_state,
        actor=actor_train_state,
        critic_params_target=critic_params_target,
        actor_params_target=actor_params_target,
        update_idx=0,
    )


def make_update_steps_fn(
    buffer: ReplayBuffer,
    num_existing_samples: int,
    update_steps: int,
    max_action: float,
    action_dim: int,
    config: TD3BCConfig,
) -> Tuple[dict, TD3BCTrainState]:
    def update_steps_fn(
        offline_train_state: TD3BCTrainState,
        rng: jax.random.PRNGKey,
    ):
        def update_step_fn(
            offline_train_state: TD3BCTrainState,
            rng: jax.random.PRNGKey,
        ):
            train_state_critic = offline_train_state.critic
            train_state_actor = offline_train_state.actor
            critic_params_target = offline_train_state.critic_params_target
            actor_params_target = offline_train_state.actor_params_target

            rng, subkey = jax.random.split(rng)
            obs, action, reward, next_obs, done = sample_batch(
                buffer, num_existing_samples, config, subkey
            )
            rng, subkey = jax.random.split(rng)
            critic_grad_fn = jax.value_and_grad(get_critic_loss, has_aux=True)
            critic_loss, critic_grads = critic_grad_fn(
                train_state_critic.params,
                critic_params_target,
                actor_params_target,
                train_state_critic.apply_fn,
                train_state_actor.apply_fn,
                obs,
                action,
                reward,
                done,
                next_obs,
                config.gamma,
                config.td3_policy_noise_std,
                config.td3_policy_noise_std,
                max_action,
                subkey,
            )
            train_state_critic = train_state_critic.apply_gradients(grads=critic_grads)
            actor_grad_fn = jax.value_and_grad(get_actor_loss, has_aux=True)
            actor_loss, actor_grads = actor_grad_fn(
                train_state_actor.params,
                train_state_critic.params,
                train_state_actor.apply_fn,
                train_state_critic.apply_fn,
                obs,
                action,
                config.td3_alpha,
            )
            new_train_state_actor = train_state_actor.apply_gradients(grads=actor_grads)
            train_state_actor = jax.lax.cond(
                offline_train_state.update_idx % config.policy_freq == 0,
                lambda: new_train_state_actor,
                lambda: train_state_actor,
            )
            # update target network
            critic_params_target = jax.tree_map(
                lambda target, live: config.polyak * target
                + (1.0 - config.polyak) * live,
                critic_params_target,
                train_state_critic.params,
            )
            actor_params_target = jax.tree_map(
                lambda target, live: config.polyak * target
                + (1.0 - config.polyak) * live,
                actor_params_target,
                train_state_actor.params,
            )
            offline_train_state = TD3BCTrainState(
                critic=train_state_critic,
                actor=train_state_actor,
                critic_params_target=critic_params_target,
                actor_params_target=actor_params_target,
                update_idx=offline_train_state.update_idx + 1,
            )
            return offline_train_state, None

        rng_keys = jax.random.split(rng, update_steps)
        offline_train_state, _ = jax.lax.scan(
            update_step_fn, offline_train_state, rng_keys
        )
        return offline_train_state

    return update_steps_fn


@partial(jax.jit)
def get_action(
    actor_train_state: TrainState,
    obs: jnp.ndarray,
) -> jnp.ndarray:
    action = actor_train_state.apply_fn(actor_train_state.params, obs)
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
    return obs, action, reward, next_obs, done


def get_actor_loss(
    actor_params: flax.core.frozen_dict.FrozenDict,
    critic_params: flax.core.frozen_dict.FrozenDict,
    actor_apply_fn: Callable[..., Any],
    critic_apply_fn: Callable[..., Any],
    obs: jnp.ndarray,
    action: jnp.ndarray,
    td3_alpha: Optional[float] = None,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    predicted_action = actor_apply_fn(actor_params, obs, rng=None)
    critic_params = jax.lax.stop_gradient(critic_params)
    q_value, _ = critic_apply_fn(
        critic_params, obs, predicted_action, rng=None
    )  # todo this will also affect the critic update :/

    if td3_alpha is None:
        loss_actor = -1.0 * q_value.mean()
        bc_loss = 0.0
        loss_lambda = 1.0
    else:
        mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
        loss_lambda = td3_alpha / mean_abs_q

        bc_loss = jnp.square(predicted_action - action).mean()
        loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
    return loss_actor, (
        bc_loss,
        -1.0 * q_value.mean() * loss_lambda,
    )


def get_critic_loss(
    critic_params: flax.core.frozen_dict.FrozenDict,
    critic_target_params: flax.core.frozen_dict.FrozenDict,
    actor_target_params: flax.core.frozen_dict.FrozenDict,
    critic_apply_fn: Callable[..., Any],
    actor_apply_fn: Callable[..., Any],
    obs: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    done: jnp.ndarray,
    next_obs: jnp.ndarray,
    decay: float,
    policy_noise_std: float,
    policy_noise_clip: float,
    max_action: float,
    rng_key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    q_pred_1, q_pred_2 = critic_apply_fn(critic_params, obs, action, rng=None)

    target_next_action = actor_apply_fn(actor_target_params, next_obs, rng=None)
    policy_noise = (
        policy_noise_std * max_action * jax.random.normal(rng_key, action.shape)
    )
    target_next_action = target_next_action + policy_noise.clip(
        -policy_noise_clip, policy_noise_clip
    )
    target_next_action = target_next_action.clip(-max_action, max_action)

    q_next_1, q_next_2 = critic_apply_fn(
        critic_target_params, next_obs, target_next_action, rng=None
    )

    target = reward[..., None] + decay * jnp.minimum(q_next_1, q_next_2) * (
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
            action = get_action(
                actor_train_state=actor_trainstate,
                obs=obs
            )
            action = action.reshape(-1)
            obs, rew, done, info = env.step(action)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return np.mean(episode_rews)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(config.seed)
    wandb.init(
        project="train-" + "TD3-BC",
        config=config,
    )
    buffer = ReplayBuffer(
        states=jnp.asarray(dataset["observations"]),
        actions=jnp.asarray(dataset["actions"]),
        next_states=jnp.asarray(dataset["next_observations"]),
        rewards=jnp.asarray(dataset["rewards"]),
        dones=jnp.asarray(dataset["terminals"]),
    )
    rng, rng_permute = jax.random.split(rng)
    perm = jax.random.permutation(rng_permute, len(buffer.states))
    buffer = jax.tree_map(lambda x: x[perm], buffer)

    rng, rng_buffer = jax.random.split(rng)
    buffer_idx = jax.random.randint(
        rng_buffer, (config.buffer_size,), 0, len(buffer.states)
    )
    buffer = jax.tree_map(lambda x: x[buffer_idx], buffer)

    obs_mean = np.mean(buffer.states, axis=0)
    obs_std = np.std(buffer.states, axis=0)
    buffer = buffer._replace(
        states=(buffer.states - obs_mean) / obs_std,
        next_states=(buffer.next_states - obs_mean) / obs_std,
    )
    offline_train_state = get_models(rng)

    total_steps = 0
    log_steps, log_return = [], []
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
        offline_train_state = jit_update_steps_fn(
            offline_train_state,
            rng_update,
        )
        eval_dict = {}
        eval_reward = eval_d4rl(
            rng_eval,
            offline_train_state.actor,
            env,
            obs_mean,
            obs_std,
        )
        eval_rew_normed = env.get_normalized_score(eval_reward) * 100
        eval_dict[f"offline/eval_reward_{config.env_name}"] = eval_reward
        eval_dict[f"offline/eval_rew_normed_{config.env_name}"] = eval_rew_normed
        eval_dict[f"offline/step"] = total_steps
        print(eval_dict)
        wandb.log(eval_dict)
        log_steps.append(total_steps)
        log_return.append(eval_dict)
    wandb.finish()
