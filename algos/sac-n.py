import copy
import os
import gym
import d4rl
import pyrallis
import numpy as np
import wandb
import uuid

import jax
import chex
import optax
import distrax
import jax.numpy as jnp

import flax
import flax.linen as nn

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Any
from tqdm.auto import trange

from flax.training.train_state import TrainState
from typing import NamedTuple
from functools import partial
from omegaconf import OmegaConf
from pydantic import BaseModel

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


class SACNConfig(BaseModel):
    # wandb params
    algo: str = "SAC-N"
    project: str = "train-SAC-N"    
    env_name: str = "halfcheetah-medium-expert-v2"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    # training params
    batch_size: int = 256
    max_steps: int = 1000000
    n_jitted_updates: int = 8
    target_entropy: float = -1.0
    # evaluation params
    eval_episodes: int = 10
    eval_interval: int = 50000
    # general params
    seed: int = 10
    eval_seed: int = 42
    
    def __hash__(self):
        return hash(self.__repr__())

conf_dict = OmegaConf.from_cli()
config = SACNConfig(**conf_dict)


@chex.dataclass(frozen=True)
class ReplayBuffer:
    data: Dict[str, jax.Array]

    @staticmethod
    def create_from_d4rl(dataset_name: str) -> "ReplayBuffer":
        d4rl_data = d4rl.qlearning_dataset(gym.make(dataset_name))
        buffer = {
            "obs": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_obs": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32)
        }
        return ReplayBuffer(data=buffer)

    @property
    def size(self):
        # WARN: do not use __len__ here! It will use len of the dataclass, i.e. number of fields.
        return self.data["obs"].shape[0]

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict

    def soft_update(self, tau):
        new_target_params = optax.incremental_update(self.params, self.target_params, tau)
        return self.replace(target_params=new_target_params)


# SAC-N networks
class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


# WARN: only for [-1, 1] action bounds, scaling/unscaling is left as an exercise for the reader :D
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state):
        net = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
        ])
        log_sigma_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))
        mu_net = nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3))

        trunk = net(state)
        mu, log_sigma = mu_net(trunk), log_sigma_net(trunk)
        log_sigma = jnp.clip(log_sigma, -5, 2)

        dist = TanhNormal(mu, jnp.exp(log_sigma))
        return dist


class Critic(nn.Module):
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state, action):
        network = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_uniform(), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ])
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics
        )
        q_values = ensemble(self.hidden_dim)(state, action)
        return q_values


class Alpha(nn.Module):
    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_alpha = self.param("log_alpha", lambda key: jnp.array([jnp.log(self.init_value)]))
        return jnp.exp(log_alpha)


class SACNTrainState(NamedTuple):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState


class SACN(object):
    # SAC-N losses
    def update_actor(
            self,
            train_state: SACNTrainState,
            batch: Dict[str, jax.Array],
            rng: jax.random.PRNGKey,
            config: SACNConfig
    ) -> Tuple[SACNTrainState, Dict[str, Any]]:
        def actor_loss_fn(actor_params):
            actions_dist = train_state.actor.apply_fn(actor_params, batch["obs"])
            actions, actions_logp = actions_dist.sample_and_log_prob(seed=rng)

            q_values = train_state.critic.apply_fn(train_state.critic.params, batch["obs"], actions).min(0)
            loss = (train_state.alpha.apply_fn(train_state.alpha.params) * actions_logp.sum(-1) - q_values).mean()

            batch_entropy = -actions_logp.sum(-1).mean()
            return loss, batch_entropy

        (loss, batch_entropy), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(train_state.actor.params)
        new_actor = train_state.actor.apply_gradients(grads=grads)
        info = {
            "batch_entropy": batch_entropy,
            "actor_loss": loss
        }
        return train_state._replace(actor=new_actor), info


    def update_alpha(
            self,
            train_state: SACNTrainState,
            entropy: float,
            config: SACNConfig
    ) -> Tuple[SACNTrainState, Dict[str, Any]]:
        def alpha_loss_fn(alpha_params):
            alpha_value = train_state.alpha.apply_fn(alpha_params)
            loss = (alpha_value * (entropy - config.target_entropy)).mean()
            return loss

        loss, grads = jax.value_and_grad(alpha_loss_fn)(train_state.alpha.params)
        new_alpha = train_state.alpha.apply_gradients(grads=grads)
        info = {
            "alpha": train_state.alpha.apply_fn(train_state.alpha.params),
            "alpha_loss": loss
        }
        return train_state._replace(alpha=new_alpha), info


    def update_critic(
            self,
            train_state: SACNTrainState,
            batch: Dict[str, jax.Array],
            rng: jax.random.PRNGKey,
            config: SACNConfig
    ) -> Tuple[SACNTrainState, Dict[str, Any]]:
        next_actions_dist = train_state.actor.apply_fn(train_state.actor.params, batch["next_obs"])
        next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=rng)

        next_q = train_state.critic.apply_fn(train_state.critic.target_params, batch["next_obs"], next_actions).min(0)
        next_q = next_q - train_state.alpha.apply_fn(train_state.alpha.params) * next_actions_logp.sum(-1)
        target_q = batch["rewards"] + (1 - batch["dones"]) * config.gamma * next_q

        def critic_loss_fn(critic_params):
            # [N, batch_size] - [1, batch_size]
            q = train_state.critic.apply_fn(critic_params, batch["obs"], batch["actions"])
            loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
            return loss

        loss, grads = jax.value_and_grad(critic_loss_fn)(train_state.critic.params)
        new_critic = train_state.critic.apply_gradients(grads=grads).soft_update(tau=config.tau)
        info = {
            "critic_loss": loss
        }
        return train_state._replace(critic=new_critic), info
    

    @partial(jax.jit, static_argnums=(0, 4))
    def update_n_times(
            self,
            train_state: SACNTrainState,
            buffer: ReplayBuffer,
            rng: jax.random.PRNGKey,
            config: SACNConfig
    ):
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, actor_rng, critic_rng = jax.random.split(rng, 4)
            batch = buffer.sample_batch(batch_rng, config.batch_size)

            train_state, actor_info = self.update_actor(train_state, batch, actor_rng, config)
            train_state, alpha_info = self.update_alpha(train_state, actor_info["batch_entropy"], config)
            train_state, critic_info = self.update_critic(train_state, batch, critic_rng, config)
        return train_state, {
            "critic_loss": critic_info["critic_loss"],
            "actor_loss": actor_info["actor_loss"],
            "alpha_loss": alpha_info["alpha_loss"],
            "alpha": alpha_info["alpha"],
            "batch_entropy": actor_info["batch_entropy"]
        }


# evaluation
@jax.jit
def eval_actions_jit(actor: TrainState, obs: jax.Array) -> jax.Array:
    dist = actor.apply_fn(actor.params, obs)
    action = dist.mean()
    return action


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def evaluate(env: gym.Env, actor: TrainState, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)

    returns = []
    for _ in trange(num_episodes, leave=False):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = eval_actions_jit(actor, obs)
            obs, reward, done, _ = env.step(np.asarray(jax.device_get(action)))
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def create_train_state(
    observations: jax.Array,
    actions: jax.Array,
    config: SACNConfig
) -> SACNTrainState:
    key = jax.random.PRNGKey(seed=config.seed)
    key, actor_key, critic_key, alpha_key = jax.random.split(key, 4)
    init_state = jnp.zeros_like(observations)
    init_action = jnp.zeros_like(actions)

    actor_module = Actor(action_dim=init_action.shape[-1], hidden_dim=config.hidden_dim)
    actor = TrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, init_state),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(hidden_dim=config.hidden_dim, num_critics=config.num_critics)
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, init_state, init_action),
        target_params=critic_module.init(critic_key, init_state, init_action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    alpha_module = Alpha()
    alpha = TrainState.create(
        apply_fn=alpha_module.apply,
        params=alpha_module.init(alpha_key),
        tx=optax.adam(learning_rate=config.alpha_learning_rate)
    )
    train_state = SACNTrainState(actor=actor, critic=critic, alpha=alpha)
    return train_state


if __name__ == "__main__":
    wandb.init(config=config, project=config.project)
    rng = jax.random.PRNGKey(config.seed)
    buffer = ReplayBuffer.create_from_d4rl(config.env_name)
    eval_env = make_env(config.env_name, seed=config.eval_seed)
    target_entropy = -np.prod(eval_env.action_space.shape)
    config.target_entropy = target_entropy

    example_obs = buffer.data["obs"][0]
    example_act = buffer.data["actions"][0]
    train_state = create_train_state(example_obs, example_act, config)

    algo = SACN()
    num_steps = int(config.max_steps / config.n_jitted_updates)
    eval_interval = int(config.eval_interval / config.n_jitted_updates)
    for _ in trange(num_steps):
        rng, update_rng = jax.random.split(rng)
        train_state, update_info = algo.update_n_times(train_state, buffer, update_rng, config)
        wandb.log({"step": _, **update_info})

        if _ % eval_interval == 0:
            eval_returns = evaluate(eval_env, train_state.actor, config.eval_episodes, seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0

            wandb.log({
                "step": _,
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })
            print(f"Epoch {_}, eval/normalized_score_mean: {np.mean(normalized_score)}, eval/normalized_score_std: {np.std(normalized_score)}")
    wandb.finish()