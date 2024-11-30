# Source: https://github.com/tinkoff-ai/ReBRAC/tree/public-release
# Paper: https://arxiv.org/abs/2305.09836

import os

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import math
import uuid
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Callable, Dict, NamedTuple, Tuple

import chex
import d4rl
import distrax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyrallis
import wandb
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm.auto import trange


class ReBRACConfig(BaseModel):
    # wandb params
    algo: str = "ReBRAC"
    project: str = "train-ReBRAC"
    env_name: str = "halfcheetah-medium-expert-v2"
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
    batch_size: int = 256
    max_steps: int = 1000000
    data_size: int = 1000000
    normalize_reward: bool = False
    normalize_state: bool = False
    n_jitted_updates: int = 8
    # evaluation params
    eval_episodes: int = 10
    eval_interval: int = 100000
    # general params

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = ReBRACConfig(**conf_dict)

from copy import deepcopy
from typing import Callable, Dict, Sequence, Tuple, Union

from tqdm.auto import trange

default_kernel_init = nn.initializers.lecun_normal()
default_bias_init = nn.initializers.zeros


def pytorch_init(fan_in: float):
    """
    Default init for PyTorch Linear layer weights and biases:
    https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
    """
    bound = math.sqrt(1 / fan_in)

    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def uniform_init(bound: float):
    def _init(key, shape, dtype):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

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
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else identity,
            ]
        layers += [
            nn.Dense(
                self.action_dim,
                kernel_init=uniform_init(1e-3),
                bias_init=uniform_init(1e-3),
            ),
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
            nn.Dense(
                self.hidden_dim,
                kernel_init=pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
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
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(
            state, action
        )
        return q_values


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    next_actions: jnp.ndarray
    dones: jnp.ndarray


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

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    next_action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = "timeouts" in dataset

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        new_action = dataset["actions"][i + 1].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
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
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "next_actions": np.array(next_action_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
    }


def get_dataset(
    env: gym.Env, config: ReBRACConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    dataset = qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    imputed_next_observations = np.roll(dataset["observations"], -1, axis=0)
    same_obs = np.all(
        np.isclose(imputed_next_observations, dataset["next_observations"], atol=1e-5),
        axis=-1,
    )
    dones = 1.0 - same_obs.astype(np.float32)
    dones[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        next_actions=jnp.array(dataset["next_actions"], dtype=jnp.float32),
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


class CriticTrainState(TrainState):
    target_params: FrozenDict


class ActorTrainState(TrainState):
    target_params: FrozenDict


class ReBRACTrainState(NamedTuple):
    actor: TrainState
    critic: CriticTrainState


class ReBRAC(object):
    def update_actor(
        self,
        train_state: ReBRACTrainState,
        batch: Dict[str, jax.Array],
        rng: jax.random.PRNGKey,
        config: ReBRACConfig,
    ) -> Tuple[ReBRACTrainState, jax.Array]:
        key, random_action_key = jax.random.split(rng, 2)

        def actor_loss_fn(params):
            actions = train_state.actor.apply_fn(params, batch.observations)

            bc_penalty = ((actions - batch.actions) ** 2).sum(-1)
            q_values = train_state.critic.apply_fn(
                train_state.critic.params, batch.observations, actions
            ).min(0)
            lmbda = 1
            if config.normalize_q:
                lmbda = jax.lax.stop_gradient(1 / jax.numpy.abs(q_values).mean())

            loss = (config.actor_bc_coef * bc_penalty - lmbda * q_values).mean()

            # logging stuff
            random_actions = jax.random.uniform(
                random_action_key, shape=batch.actions.shape, minval=-1.0, maxval=1.0
            )
            return loss

        actor_loss, grads = jax.value_and_grad(actor_loss_fn)(train_state.actor.params)
        new_actor = train_state.actor.apply_gradients(grads=grads)

        new_actor = new_actor.replace(
            target_params=optax.incremental_update(
                train_state.actor.params, train_state.actor.target_params, config.tau
            )
        )
        new_critic = train_state.critic.replace(
            target_params=optax.incremental_update(
                train_state.critic.params, train_state.critic.target_params, config.tau
            )
        )

        return train_state._replace(actor=new_actor, critic=new_critic), actor_loss

    def update_critic(
        self,
        train_state: ReBRACTrainState,
        batch: Dict[str, jax.Array],
        rng: jax.random.PRNGKey,
        config: ReBRACConfig,
    ) -> Tuple[ReBRACTrainState, jax.Array]:
        key, actions_key = jax.random.split(rng)

        next_actions = train_state.actor.apply_fn(
            train_state.actor.target_params, batch.next_observations
        )
        noise = jax.numpy.clip(
            (jax.random.normal(actions_key, next_actions.shape) * config.policy_noise),
            -config.noise_clip,
            config.noise_clip,
        )
        next_actions = jax.numpy.clip(next_actions + noise, -1, 1)
        bc_penalty = ((next_actions - batch.next_actions) ** 2).sum(-1)
        next_q = train_state.critic.apply_fn(
            train_state.critic.target_params, batch.next_observations, next_actions
        ).min(0)
        next_q = next_q - config.critic_bc_coef * bc_penalty

        target_q = batch.rewards + (1 - batch.dones) * config.gamma * next_q

        def critic_loss_fn(critic_params):
            # [N, batch_size] - [1, batch_size]
            q = train_state.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
            return loss

        critic_loss, grads = jax.value_and_grad(critic_loss_fn)(
            train_state.critic.params
        )
        new_critic = train_state.critic.apply_gradients(grads=grads)
        return train_state._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(0, 4))
    def update_n_times(
        self,
        train_state: ReBRACTrainState,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: ReBRACConfig,
    ):
        for _ in range(config.n_jitted_updates):
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            indices = jax.random.randint(
                batch_rng,
                shape=(config.batch_size,),
                minval=0,
                maxval=len(dataset.observations),
            )
            batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)

            train_state, critic_loss = self.update_critic(
                train_state, batch, critic_rng, config
            )
            if _ % config.policy_freq == 0:
                train_state, actor_loss = self.update_actor(
                    train_state, batch, actor_rng, config
                )

        return train_state, {"critic_loss": critic_loss, "actor_loss": actor_loss}

    @partial(jax.jit, static_argnums=(0,))
    def get_action(self, train_state: ReBRACTrainState, obs: jax.Array) -> jax.Array:
        return train_state.actor.apply_fn(train_state.actor.params, obs)


def create_train_state(observation, action, config):
    key = jax.random.PRNGKey(seed=config.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)

    action_dim = action.shape[-1]

    actor_module = DetActor(
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
        layernorm=config.actor_ln,
        n_hiddens=config.actor_n_hiddens,
    )
    actor = ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=actor_module.init(actor_key, observation),
        target_params=actor_module.init(actor_key, observation),
        tx=optax.adam(learning_rate=config.actor_learning_rate),
    )

    critic_module = EnsembleCritic(
        hidden_dim=config.hidden_dim,
        num_critics=2,
        layernorm=config.critic_ln,
        n_hiddens=config.critic_n_hiddens,
    )
    critic = CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=critic_module.init(critic_key, observation, action),
        target_params=critic_module.init(critic_key, observation, action),
        tx=optax.adam(learning_rate=config.critic_learning_rate),
    )

    train_state = ReBRACTrainState(actor=actor, critic=critic)
    return train_state


def evaluate(
    policy_fn: Callable[[jnp.ndarray], jnp.ndarray],
    env: gym.Env,
    num_episodes: int,
    obs_mean,
    obs_std,
) -> float:  # D4RL specific
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, done = env.reset(), False
        while not done:
            observation = (observation - obs_mean) / obs_std
            action = policy_fn(obs=observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    return env.get_normalized_score(np.mean(episode_returns)) * 100


if __name__ == "__main__":
    wandb.init(
        config=config,
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
    )
    wandb.mark_preempting()
    env = gym.make(config.env_name)
    dataset, obs_mean, obs_std = get_dataset(env, config)
    key = jax.random.PRNGKey(seed=config.seed)
    example_batch = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_train_state(
        example_batch.observations, example_batch.actions, config
    )

    algo = ReBRAC()

    num_steps = int(config.max_steps / config.n_jitted_updates)
    eval_interval = int(config.eval_interval / config.n_jitted_updates)
    for step in trange(num_steps, desc="ReBRAC Steps"):
        key, subkey = jax.random.split(key)
        train_state, info = algo.update_n_times(train_state, dataset, subkey, config)
        wandb.log({"epoch": step, **{f"ReBRAC/{k}": v for k, v in info.items()}})

        if step % eval_interval == 0 or step == num_steps - 1:
            policy_fn = partial(algo.get_action, train_state=train_state)
            normalized_score = evaluate(
                policy_fn, env, config.eval_episodes, obs_mean, obs_std
            )
            wandb.log(
                {
                    "epoch": step,
                    "eval/normalized_score_mean": normalized_score,
                }
            )
            print(f"Step {step} | Eval Normalized Score Mean: {normalized_score}")
    wandb.finish()
