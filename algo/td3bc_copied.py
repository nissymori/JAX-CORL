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

tfd = tfp.distributions
tfb = tfp.bijectors


class TD3BCConfig:
    # general config
    env_name: str = "Hopper"
    data_quality: str = "medium-expert"
    train_steps: int = 1000000
    evaluate_every_epochs: int = 10000
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
    n_updates_jit: int = 8


config = TD3BCConfig(**OmegaConf.to_object(OmegaConf.from_cli()))


class SegmentTimer(object):
    def __init__(self, first_segment_name: str = "InitialSegment") -> None:
        self.start_time = time.time()
        self.last_summary_time = self.start_time
        self.last_segment_start = self.start_time
        self.segment_name = first_segment_name

        self.segment_averages = {}
        self.segment_counts = {}

    def new_segment(self, new_segment_name: str, quiet=True) -> None:
        segment_duration = time.time() - self.last_segment_start

        if self.segment_name in self.segment_averages:
            prev_avg = self.segment_averages[self.segment_name]
            prev_count = self.segment_counts[self.segment_name]
            new_avg = (prev_avg * prev_count + segment_duration) / (prev_count + 1)
            self.segment_averages[self.segment_name] = new_avg
            self.segment_counts[self.segment_name] = prev_count + 1
        else:
            new_avg = segment_duration
            prev_count = 0
            self.segment_averages[self.segment_name] = new_avg
            self.segment_counts[self.segment_name] = 1

        if not quiet:
            print(
                f"{self.segment_name} took {segment_duration:.3f}s; avg {new_avg:.3f}s after {prev_count + 1} runs"
            )

        self.segment_name = new_segment_name
        self.last_segment_start = time.time()

    def summary(self) -> str:
        self.last_summary_time = time.time()
        return_str = ""
        for segment_name in self.segment_averages.keys():
            segment_duration = self.segment_averages[segment_name]
            segment_count = self.segment_counts[segment_name]
            if segment_name == self.segment_name:
                segment_duration = (
                    segment_duration * segment_count
                    + time.time()
                    - self.last_segment_start
                ) / (segment_count + 1)
                segment_count += 1
            return_str += f"{segment_name}: avg {segment_duration:.3f}s; tot {segment_count * segment_duration:.3f}s \t"

        self.segment_averages = {}
        self.segment_counts = {}
        return return_str[:-1]


segment_timer = SegmentTimer("ImportsEtc.")


class BufferManager:
    def __init__(
        self,
    ):
        pass

    @partial(jax.jit, static_argnums=(0, 2))
    def get(self, buffer, batch_size: int, rng):
        idxes = jax.random.randint(rng, (batch_size,), 0, buffer["_p"])
        batch = (
            buffer["states"][idxes],
            buffer["actions"][idxes],
            buffer["rewards"][idxes],
            buffer["next_states"][idxes],
            buffer["dones"][idxes],
        )
        return batch

    def from_dataset(
        self, ds: dict[np.ndarray], clip_to_eps: bool = True, eps: float = 1e-3
    ):
        buffer = {
            "states": ds["observations"],
            "actions": ds["actions"],
            "rewards": ds["rewards"],
            "next_states": ds["next_observations"],
            "dones": ds["terminals"],
        }
        buffer["_p"] = buffer["states"].shape[0]
        return buffer


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class DoubleCritic(nn.Module):
    num_hidden_units: int
    num_hidden_layers: int
    prefix: str = "critic"
    model_name: str = "double_critic"

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
    prefix: str = "actor"
    model_name: str = "actor"

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


class RLTrainer(object):
    def get_models(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        config: TD3BCConfig,
        rng: jax.random.PRNGKey,
    ):
        raise NotImplementedError()

    def sample_buff_and_update_n_times(
        self,
        train_state,
        buffer,
        num_existing_samples: int,
        max_action: float,
        action_dim: int,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple[TrainState, dict[str, float]]:
        raise NotImplementedError()

    def get_action(
        self,
        actor_train_state: TrainState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
        exploration_noise: bool = True,
    ) -> jnp.ndarray:
        raise NotImplementedError()


@struct.dataclass
class TD3BCTrainState:
    critic: TrainState
    actor: TrainState
    critic_params_target: flax.core.FrozenDict
    actor_params_target: flax.core.FrozenDict


class TD3BCTrainer(RLTrainer):
    def __init__(self, config: TD3BCConfig, action_space) -> None:
        super().__init__()
        self.config = config
        self.action_space = action_space

    def get_models(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        config: TD3BCConfig,
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
        )

    @partial(jax.jit, static_argnames=("self", "config", "num_existing_samples"))
    def sample_buff_and_update_n_times(
        self,
        offline_train_state: TD3BCTrainState,
        buffer,
        num_existing_samples: int,
        max_action: float,
        action_dim: int,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
    ) -> Tuple[dict, TD3BCTrainState]:
        avg_metrics_dict = defaultdict(int)
        train_state_critic = offline_train_state.critic
        train_state_actor = offline_train_state.actor
        critic_params_target = offline_train_state.critic_params_target
        actor_params_target = offline_train_state.actor_params_target

        for update_idx in range(config.n_updates_jit):
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
            avg_metrics_dict["offline/critic_grad_norm"] += jnp.mean(
                jnp.array(
                    jax.tree_util.tree_flatten(
                        jax.tree_map(jnp.linalg.norm, critic_grads)
                    )[0]
                )
            )
            avg_metrics_dict["offline/value_loss_1"] += critic_loss[1][0]
            avg_metrics_dict["offline/value_loss_2"] += critic_loss[1][1]
            avg_metrics_dict["offline/target"] += critic_loss[1][2]

            if update_idx % config.policy_freq == 0:
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
                train_state_actor = train_state_actor.apply_gradients(grads=actor_grads)
                avg_metrics_dict["offline/actor_loss"] += actor_loss[0]
                avg_metrics_dict["offline/actor_loss_bc"] += actor_loss[1][0]
                avg_metrics_dict["offline/actor_loss_td3_xlambda"] += actor_loss[1][1]
                avg_metrics_dict["offline/actor_grad_norm"] += jnp.mean(
                    jnp.array(
                        jax.tree_util.tree_flatten(
                            jax.tree_map(jnp.linalg.norm, actor_grads)
                        )[0]
                    )
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

        for k, v in avg_metrics_dict.items():
            if "offline/actor" in k:
                avg_metrics_dict[k] = v / (config.n_updates_jit / config.policy_freq)
            else:
                avg_metrics_dict[k] = v / config.n_updates_jit

        offline_train_state = TD3BCTrainState(
            critic=train_state_critic,
            actor=train_state_actor,
            critic_params_target=critic_params_target,
            actor_params_target=actor_params_target,
        )

        return avg_metrics_dict, offline_train_state

    @partial(jax.jit, static_argnames=("self", "config", "exploration_noise"))
    def get_action(
        self,
        actor_train_state: TrainState,
        obs: jnp.ndarray,
        rng: jax.random.PRNGKey,
        config: TD3BCConfig,
        exploration_noise: bool = True,
    ) -> jnp.ndarray:
        action = actor_train_state.apply_fn(actor_train_state.params, obs, rng=None)

        if exploration_noise:
            warnings.warn("TD3BC with exploration noise is probably not useful.")
            noise = (
                config.online.exploration_std
                * self.action_space.high
                * jax.random.normal(rng, action.shape)
            )
            action = action + noise.clip(
                -self.config.online.exploration_clip,
                self.config.online.exploration_clip,
            )

        action = action.clip(self.action_space.low, self.action_space.high)
        return action


@partial(jax.jit, static_argnames=("config"))
def sample_batch(buffer, num_existing_samples, config, rng):
    idxes = jax.random.randint(rng, (config.batch_size,), 0, num_existing_samples)
    obs = buffer["states"][idxes]
    action = buffer["actions"][idxes]
    reward = buffer["rewards"][idxes]
    next_obs = buffer["next_states"][idxes]
    done = buffer["dones"][idxes]
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
    actor_trainer: RLTrainer,
    actor_trainstate: TrainState,
    env: gym.Env,
    n_episodes: int,
    obs_mean,
    obs_std,
    config: TD3BCConfig,
    vae=None,
) -> float:
    episode_rews = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_rew = 0.0
        i = 0
        while not done:
            rng, subkey = jax.random.split(subkey)
            obs = (obs - obs_mean) / obs_std
            obs = jnp.array(obs)
            action = actor_trainer.get_action(
                actor_train_state=actor_trainstate,
                obs=obs,
                rng=subkey,
                config=config,
                exploration_noise=False,
            )
            action = action.reshape(-1)
            last_obs = obs
            obs, rew, done, info = env.step(action)
            episode_rew += rew
            i += 1
        episode_rews.append(episode_rew)
    return np.mean(episode_rews)


def normalize_dataset(
    buffer: dict,
    config: TD3BCConfig,
):
    """
    Normalize the dataset.
    If we have predictor, we normalize based on the labels (pu, pvu, ground_true)
    If we don't have predictor, we normalize based on the whole dataset (vanilla).
    """
    print("normalizing")
    obs_mean = np.mean(buffer["states"], axis=0)
    obs_std = np.std(buffer["states"], axis=0)
    buffer["states"] = (buffer["states"] - obs_mean) / obs_std
    buffer["next_states"] = (buffer["next_states"] - obs_mean) / obs_std
    return buffer, obs_mean, obs_std


def train_offline_d4rl():
    """
    Offline Training Loop.
    """
    print("loading dataset")
    rng = jax.random.PRNGKey(config.seed)
    wandb.init(
        project="train-" + "TD3-BC",
        config=config,
    )
    env = gym.make(f"{config.env_name.lower()}-{config.data_quality}-v2")
    dataset = d4rl.qlearning_dataset(env)
    env_obs_dim = dataset["observations"].shape[-1]
    action_space = env.action_space
    act_dim = action_space.shape[0]
    max_action = action_space.high

    buffer_manager = BufferManager()
    buffer = buffer_manager.from_dataset(dataset)

    offline_trainer = TD3BCTrainer(config=config, action_space=action_space)

    offline_train_state = offline_trainer.get_models(
        env_obs_dim, act_dim, max_action, config, rng
    )

    # normalize dataset
    (
        buffer,
        obs_mean,
        obs_std,
    ) = normalize_dataset(buffer, config)

    buffer = jax.tree_map(jnp.array, buffer)

    total_steps = 0
    log_steps, log_return = [], []
    num_total_its = int(config.train_steps // config.n_updates_jit + 1)
    t = tqdm.trange(
        1,
        config.train_steps,
        desc=f"TD3-BC",
        leave=True,
    )
    start_time = time.time()
    for it in range(num_total_its):
        segment_timer.new_segment("updating")
        total_steps += config.n_updates_jit
        t.update(config.n_updates_jit)
        rng, rng_eval, rng_update = jax.random.split(rng, 3)
        (
            metric_dict,
            offline_train_state,
        ) = offline_trainer.sample_buff_and_update_n_times(
            offline_train_state,
            buffer=buffer,
            num_existing_samples=(
                buffer["_p"] if isinstance(buffer["_p"], int) else buffer["_p"].item()
            ),
            rng=rng_update,
            action_dim=act_dim,
            max_action=max_action,
            config=config,
        )
        if it % 200 == 0:
            metric_dict["offline/step"] = total_steps
            wandb.log(metric_dict)
        if time.time() - segment_timer.last_summary_time > 60:
            print(segment_timer.summary())
        if (it + 1) % config.evaluate_every_epochs == 0:
            segment_timer.new_segment("evaluation")
            rng, rng_eval = jax.random.split(rng)
            eval_dict = {}
            eval_reward = eval_d4rl(
                rng_eval,
                offline_trainer,
                offline_train_state.actor,
                env,
                config.num_test_rollouts,
                obs_mean,
                obs_std,
                config,
            )
            eval_rew_normed = env.get_normalized_score(eval_reward) * 100
            info = {}
            eval_dict[f"offline/eval_reward_{config.env_name}"] = eval_reward
            eval_dict[f"offline/eval_rew_normed_{config.env_name}"] = eval_rew_normed
            eval_dict[f"offline/step"] = total_steps
            t.set_description(
                f"TD3-BC/{config.env_name} R_te: {eval_reward:.2f}, {eval_rew_normed:.2f}"
            )
            t.refresh()

            wandb.log(eval_dict)
            log_steps.append(total_steps)
            log_return.append(eval_dict)
    print(f"Total time: {time.time() - start_time:.2f}s")
    wandb.finish()
    return (
        log_steps,
        log_return,
        offline_train_state.actor.params,
    )


train_offline_d4rl()