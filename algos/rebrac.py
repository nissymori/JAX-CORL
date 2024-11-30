import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import wandb
import uuid
import pyrallis

import jax
import numpy as np
import optax

from functools import partial
from dataclasses import dataclass, asdict
from flax.core import FrozenDict
from typing import Dict, Tuple, Any, Callable
from tqdm.auto import trange

from flax.training.train_state import TrainState
import math
import distrax
import flax.linen as nn
import chex
import gym
import d4rl
import jax.numpy as jnp
from typing import NamedTuple
from functools import partial
from omegaconf import OmegaConf
from pydantic import BaseModel


class ReBRACConfig(BaseModel):
    # wandb params
    project: str = "ReBRAC"
    group: str = "rebrac"
    name: str = "rebrac"
    # model params
    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    gamma: float = 0.99
    tau: float = 5e-3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True
    # training params
    dataset_name: str = "halfcheetah-medium-v2"
    batch_size: int = 256
    max_steps: int = 1000000
    normalize_reward: bool = False
    normalize_states: bool = False
    n_jitted_updates: int = 8
    # evaluation params
    eval_episodes: int = 10
    eval_interval: int = 100000
    # general params
    train_seed: int = 0
    eval_seed: int = 42
    
    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = ReBRACConfig(**conf_dict)

from copy import deepcopy
from tqdm.auto import trange
from typing import Sequence, Dict, Callable, Tuple, Union


@chex.dataclass(frozen=True)
class Metrics:
    accumulators: Dict[str, Tuple[jax.Array, jax.Array]]

    @staticmethod
    def create(metrics: Sequence[str]) -> "Metrics":
        init_metrics = {key: (jnp.array([0.0]), jnp.array([0.0])) for key in metrics}
        return Metrics(accumulators=init_metrics)

    def update(self, updates: Dict[str, jax.Array]) -> "Metrics":
        new_accumulators = deepcopy(self.accumulators)
        for key, value in updates.items():
            acc, steps = new_accumulators[key]
            new_accumulators[key] = (acc + value, steps + 1)

        return self.replace(accumulators=new_accumulators)

    def compute(self) -> Dict[str, np.ndarray]:
        # cumulative_value / total_steps
        return {k: np.array(v[0] / v[1]) for k, v in self.accumulators.items()}


def normalize(arr: jax.Array, mean: jax.Array, std: jax.Array, eps: float = 1e-8) -> jax.Array:
    return (arr - mean) / (std + eps)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def evaluate(env: gym.Env, params, action_fn: Callable, num_episodes: int, seed: int) -> np.ndarray:
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    returns = []
    for _ in trange(num_episodes, desc="Eval", leave=False):
    # for _ in range(num_episodes):
        obs, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = np.asarray(jax.device_get(action_fn(params, obs)))
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, next_actins, rewards,
     and a terminal flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            next_actions: An N x dim_action array of next actions.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        new_action = dataset['actions'][i + 1].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        next_action_.append(new_action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'next_actions': np.array(next_action_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def compute_mean_std(states: jax.Array, eps: float) -> Tuple[jax.Array, jax.Array]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: jax.Array, mean: jax.Array, std: jax.Array):
    return (states - mean) / std


@chex.dataclass
class ReplayBuffer:
    data: Dict[str, jax.Array] = None
    mean: float = 0
    std: float = 1

    def create_from_d4rl(self, dataset_name: str, normalize_reward: bool = False,
                         normalize: bool = False):
        d4rl_data = qlearning_dataset(gym.make(dataset_name))
        buffer = {
            "states": jnp.asarray(d4rl_data["observations"], dtype=jnp.float32),
            "actions": jnp.asarray(d4rl_data["actions"], dtype=jnp.float32),
            "rewards": jnp.asarray(d4rl_data["rewards"], dtype=jnp.float32),
            "next_states": jnp.asarray(d4rl_data["next_observations"], dtype=jnp.float32),
            "next_actions": jnp.asarray(d4rl_data["next_actions"], dtype=jnp.float32),
            "dones": jnp.asarray(d4rl_data["terminals"], dtype=jnp.float32)
        }
        if normalize:
            self.mean, self.std = compute_mean_std(buffer["states"], eps=1e-3)
            buffer["states"] = normalize_states(
                buffer["states"], self.mean, self.std
            )
            buffer["next_states"] = normalize_states(
                buffer["next_states"], self.mean, self.std
            )
        if normalize_reward:
            buffer["rewards"] = ReplayBuffer.normalize_reward(dataset_name, buffer["rewards"])
        self.data = buffer

    @property
    def size(self):
        # WARN: do not use __len__ here! It will use len of the dataclass, i.e. number of fields.
        return self.data["states"].shape[0]

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int) -> Dict[str, jax.Array]:
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree_map(lambda arr: arr[indices], self.data)
        return batch

    def get_moments(self, modality: str) -> Tuple[jax.Array, jax.Array]:
        mean = self.data[modality].mean(0)
        std = self.data[modality].std(0)
        return mean, std

    @staticmethod
    def normalize_reward(dataset_name: str, rewards: jax.Array) -> jax.Array:
        if "antmaze" in dataset_name:
            return rewards * 100.0  # like in LAPO
        else:
            raise NotImplementedError("Reward normalization is implemented only for AntMaze yet!")


default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


def pytorch_init(fan_in: float):
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(key, shape=shape, minval=-bound, maxval=bound, dtype=dtype)
    return _init


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mean(self):
        return self.bijector.forward(self.distribution.mean())


def identity(x):
    return x


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state):
        s_d, h_d = state.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(self.action_dim, kernel_init=uniform_init(1e-3), bias_init=uniform_init(1e-3)),
            nn.tanh,
        ]
        net = nn.Sequential(layers)
        actions = net(state)
        return actions


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = state.shape[-1], action.shape[-1], self.hidden_dim
        # Initialization as in the EDAC paper
        layers = [
            nn.Dense(self.hidden_dim, kernel_init=pytorch_init(s_d + a_d), bias_init=nn.initializers.constant(0.1)),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(self.hidden_dim, kernel_init=pytorch_init(h_d), bias_init=nn.initializers.constant(0.1)),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(1, kernel_init=uniform_init(3e-3), bias_init=uniform_init(3e-3))
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(state, action)
        return q_values


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


class SACNState(NamedTuple):
    actor: TrainState
    critic: CriticTrainState


class ReBRAC(object):
    def update_actor(
        self,
        train_state: SACNState,
        batch: Dict[str, jax.Array],
        rng: jax.random.PRNGKey,
        config: ReBRACConfig,
    ) -> Tuple[SACNState, jax.Array]:
        key, random_action_key = jax.random.split(rng, 2)

        def actor_loss_fn(params):
            actions = train_state.actor.apply_fn(params, batch["states"])

            bc_penalty = ((actions - batch["actions"]) ** 2).sum(-1)
            q_values = train_state.critic.apply_fn(train_state.critic.params, batch["states"], actions).min(0)
            lmbda = 1
            if config.normalize_q:
                lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

            loss = (config.actor_bc_coef * bc_penalty - lmbda * q_values).mean()

            # logging stuff
            random_actions = jax.random.uniform(random_action_key, shape=batch["actions"].shape, minval=-1.0, maxval=1.0)
            return loss

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(train_state.actor.params)
        new_actor = train_state.actor.apply_gradients(grads=grads)

        new_actor = new_actor.replace(
            target_params=optax.incremental_update(train_state.actor.params, train_state.actor.target_params, config.tau)
        )
        new_critic = train_state.critic.replace(
            target_params=optax.incremental_update(train_state.critic.params, train_state.critic.target_params, config.tau)
        )

        return train_state._replace(actor=new_actor, critic=new_critic), actor_loss


    def update_critic(
            self,
            train_state: SACNState,
            batch: Dict[str, jax.Array],
            rng: jax.random.PRNGKey,
            config: ReBRACConfig,
    ) -> Tuple[SACNState, jax.Array]:
        key, actions_key = jax.random.split(rng)

        next_actions = train_state.actor.apply_fn(train_state.actor.target_params, batch["next_states"])
        noise = jax.numpy.clip(
            (jax.random.normal(actions_key, next_actions.shape) * config.policy_noise),
            -config.noise_clip,
            config.noise_clip,
        )
        next_actions = jax.numpy.clip(next_actions + noise, -1, 1)
        bc_penalty = ((next_actions - batch["next_actions"]) ** 2).sum(-1)
        next_q = train_state.critic.apply_fn(train_state.critic.target_params, batch["next_states"], next_actions).min(0)
        next_q = next_q - config.critic_bc_coef * bc_penalty

        target_q = batch["rewards"] + (1 - batch["dones"]) * config.gamma * next_q

        def critic_loss_fn(critic_params):
            # [N, batch_size] - [1, batch_size]
            q = train_state.critic.apply_fn(critic_params, batch["states"], batch["actions"])
            loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
            return loss

        critic_loss, grads = jax.value_and_grad(critic_loss_fn)(train_state.critic.params)
        new_critic = train_state.critic.apply_gradients(grads=grads)
        return train_state._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(0, 4))
    def update_n_times(
        self,
        train_state: SACNState,
        buffer: ReplayBuffer,
        rng: jax.random.PRNGKey,
        config: ReBRACConfig,
    ):
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch = buffer.sample_batch(batch_rng, config.batch_size)

            train_state, critic_loss = self.update_critic(train_state, batch, critic_rng, config)
            if _ % config.policy_freq == 0:
                train_state, actor_loss = self.update_actor(train_state, batch, actor_rng, config)

        return train_state, {"critic_loss": critic_loss, "actor_loss": actor_loss}


def action_fn(actor: TrainState) -> Callable:
    @jax.jit
    def _action_fn(obs: jax.Array) -> jax.Array:
        action = actor.apply_fn(actor.params, obs)
        return action

    return _action_fn


def create_train_state(observation, action, config):
    key = jax.random.PRNGKey(seed=config.train_seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    action_dim = action.shape[-1]

    actor_module = DetActor(action_dim=action_dim, hidden_dim=config.hidden_dim, layernorm=config.actor_ln,
                            n_hiddens=config.actor_n_hiddens)
    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, observation),
        target_params=actor_module.init(actor_key, observation),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(hidden_dim=config.hidden_dim, num_critics=2, layernorm=config.critic_ln,
                                   n_hiddens=config.critic_n_hiddens)
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, observation, action),
        target_params=critic_module.init(critic_key, observation, action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    train_state = SACNState(actor=actor, critic=critic)
    return train_state


if __name__ == "__main__":
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    buffer = ReplayBuffer()
    buffer.create_from_d4rl(config.dataset_name, config.normalize_reward, config.normalize_states)
    eval_env = make_env(config.dataset_name, seed=config.eval_seed)
    eval_env = wrap_env(eval_env, buffer.mean, buffer.std)

    key = jax.random.PRNGKey(seed=config.train_seed)
    train_state = create_train_state(buffer.data["states"][0], buffer.data["actions"][0], config)

    rebrac = ReBRAC()
    

    @jax.jit
    def actor_action_fn(params, obs):
        return train_state.actor.apply_fn(params, obs)

    num_steps = int(config.max_steps / config.n_jitted_updates)
    eval_interval = int(config.eval_interval / config.n_jitted_updates)
    for step in trange(num_steps, desc="ReBRAC Steps"):
        key, subkey = jax.random.split(key)
        train_state, info = rebrac.update_n_times(train_state, buffer, subkey, config)
        wandb.log({"epoch": step, **{f"ReBRAC/{k}": v for k, v in info.items()}})

        if step % eval_interval == 0 or step == num_steps - 1:
            eval_returns = evaluate(eval_env, train_state.actor.params, actor_action_fn, config.eval_episodes,
                                    seed=config.eval_seed)
            normalized_score = eval_env.get_normalized_score(eval_returns) * 100.0
            wandb.log({
                "epoch": step,
                "eval/return_mean": np.mean(eval_returns),
                "eval/return_std": np.std(eval_returns),
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })
            print(f"Step {step} | Eval Return Mean: {np.mean(eval_returns)} | Eval Normalized Score Mean: {np.mean(normalized_score)}")
    wandb.finish()