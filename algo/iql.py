
import jax
import jax.numpy as jnp
import collections
import os
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import flax
import flax.linen as nn
import optax
import functools
from tensorflow_probability.substrates import jax as tfp

import gym
import numpy as np
import tqdm
from tensorboardX import SummaryWriter
from flax.training.train_state import TrainState
from tqdm import tqdm
from omegaconf import OmegaConf
from pydantic import BaseModel
from functools import partial
from typing import NamedTuple


tfd = tfp.distributions
tfb = tfp.bijectors


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]


class IQLConfig(BaseModel):
    env_name: str = 'hopper-medium-expert-v2'
    save_dir: str = './tmp/'
    seed: int = 42
    eval_episodes: int = 10
    log_interval: int = 1000
    eval_interval: int = 100000
    batch_size: int = 256
    max_steps: int = int(1e6)
    tqdm: bool = True
    tau: float = 0.005
    discount: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    dropout_rate: Optional[float] = None
    hidden_dims: Sequence[int] = [256, 256]


config = IQLConfig(**OmegaConf.to_object(OmegaConf.from_cli()))




class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.dropout_rate is not None:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2


def _sample_actions(rng: PRNGKey,
                    actor_fn: nn.Module,
                    actor_params: Params,
                    observations: jnp.ndarray,
                    temperature: float) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_fn(actor_params, observations=observations, temperature=temperature)
    return rng, dist.sample(seed=rng)


def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: jnp.ndarray,
                   temperature: float) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations, temperature)


def awr_update_actor(key: PRNGKey, actor: TrainState, critic: TrainState, value: TrainState,
           batch: Batch, temperature: float) -> Tuple[TrainState, InfoDict]:
    v = value.apply_fn(value.params, batch.observations)

    q1, q2 = critic.apply_fn(critic.params, batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature)
    exp_a = jnp.minimum(exp_a, 100.0)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply_fn(actor_params,
                           observations=batch.observations,
                           training=True,
                           rngs={'dropout': key})
        log_probs = dist.log_prob(batch.actions)
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    (actor_loss, info), grads = actor_grad_fn(actor.params)
    new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info


def loss(diff, expectile):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(critic: TrainState, value: TrainState, batch: Batch) -> Tuple[TrainState, InfoDict]:
    actions = batch.actions
    q1, q2 = critic.apply_fn(critic.params, batch.observations, actions)
    q = jnp.minimum(q1, q2)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply_fn(value_params, observations=batch.observations)
        value_loss = loss(q - v, config.expectile).mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }
    value_grad_fn = jax.value_and_grad(value_loss_fn, has_aux=True)
    (value_loss, info), grads = value_grad_fn(value.params)
    new_value = value.apply_gradients(grads=grads)
    return new_value, info


def update_q(critic: TrainState, target_value: TrainState, batch: Batch) -> Tuple[TrainState, InfoDict]:
    next_v = target_value.apply_fn(target_value.params, batch.next_observations)
    target_q = batch.rewards + config.discount * batch.masks * next_v

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn(critic_params, observations=batch.observations,
                              actions=batch.actions)
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    (critic_loss, info), grads = critic_grad_fn(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    return new_critic, info


from typing import Dict
import d4rl
import flax.linen as nn
import gym
import numpy as np
import time


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False
        episode_return, episode_length = 0, 0
        while not done:
            action = agent.sample_actions(observation, 0.0)
            observation, reward, done, info = env.step(action)
            episode_return += reward
            episode_length += 1
        stats['return'].append(episode_return)
        stats['length'].append(episode_length)
    for k, v in stats.items():
        stats[k] = np.mean(v)
    #stats["return"] = env.get_normalized_score(stats["return"]) * 100
    return stats


"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

class IQLUpdateState(NamedTuple):
    actor: TrainState
    critic: TrainState
    value: TrainState
    target_critic: TrainState

def target_update(critic: TrainState, target_critic: TrainState) -> TrainState:
    new_target_params = jax.tree_map(
        lambda p, tp: p * config.tau + tp * (1 - config.tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey, update_state: IQLUpdateState, batch: Batch
) -> Tuple[IQLUpdateState, InfoDict]:

    new_value, value_info = update_v(update_state.target_critic, update_state.value, batch)
    new_actor, actor_info = awr_update_actor(rng, update_state.actor, update_state.target_critic,
                                             new_value, batch, config.temperature)

    new_critic, critic_info = update_q(update_state.critic, new_value, batch)

    new_target_critic = target_update(new_critic, update_state.target_critic)

    return IQLUpdateState(
        actor=new_actor,
        critic=new_critic,
        value=new_value,
        target_critic=new_target_critic,
        ), {
        **critic_info,
        **value_info,
        **actor_info
    }


def init_params(model_def: nn.Module, inputs: Sequence[jnp.ndarray]):
    return model_def.init(*inputs)


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 opt_decay_schedule: str = "cosine"):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = NormalTanhPolicy(config.hidden_dims,
                                    action_dim,
                                    log_std_scale=1e-3,
                                    log_std_min=-5.0,
                                    dropout_rate=config.dropout_rate,
                                    state_dependent_std=False,
                                    tanh_squash_distribution=False)

        schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_steps)
        optimiser = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(schedule_fn))

        actor_params = init_params(actor_def, [actor_key, observations])
        actor: TrainState = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optimiser,
        )

        critic_def = DoubleCritic(config.hidden_dims)
        critic_params = init_params(critic_def, [critic_key, observations, actions])
        critic: TrainState = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=config.critic_lr),
        )


        value_def = ValueCritic(config.hidden_dims)
        value_params = init_params(value_def, [value_key, observations])
        value: TrainState = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.adam(learning_rate=config.value_lr),
        )

        target_critic_params = init_params(critic_def, [critic_key, observations, actions])
        target_critic: TrainState = TrainState.create(
            apply_fn=critic_def.apply,
            params=target_critic_params,
            tx=optax.adam(learning_rate=config.value_lr),
        )
        self.update_state = IQLUpdateState(
            actor=actor,
            critic=critic,
            value=value,
            target_critic=target_critic,
        )
        self.rng = rng

    @partial(jax.jit, static_argnums=(0,))
    def sample_actions(self,
                       observations: np.ndarray,
                       temperature) -> jnp.ndarray:
        rng, actions = sample_actions(self.rng, self.update_state.actor.apply_fn,
                                             self.update_state.actor.params, observations,
                                             temperature)
        self.rng = rng
        actions = jnp.asarray(actions)
        return jnp.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        rng, subrng = jax.random.split(self.rng)
        update_state, info = _update_jit(
            subrng, self.update_state,
            batch)

        self.rng = rng
        self.update_state = update_state
        return info


import collections
from typing import Optional

import d4rl
import gym
import numpy as np

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))


TimeStep = Tuple[np.ndarray, float, bool, dict]

from gym.spaces import Box, Dict

import os
from typing import Tuple

import gym
import numpy as np
from tensorboardX import SummaryWriter


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in config.env_name:
        dataset.rewards -= 1.0
        # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
        # but I found no difference between (x - 0.5) * 4 and x - 1.0
    elif ('halfcheetah' in config.env_name or 'walker2d' in config.env_name
          or 'hopper' in config.env_name):
        normalize(dataset)

    return env, dataset


def main():
    summary_writer = SummaryWriter(os.path.join(config.save_dir, 'tb',
                                                str(config.seed)),
                                   write_to_disk=True)
    os.makedirs(config.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(config.env_name, config.seed)

    agent = Learner(config.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                )

    eval_returns = []
    for i in tqdm(range(1, config.max_steps + 1),
                       smoothing=0.1,
                       disable=not config.tqdm):
        batch = dataset.sample(config.batch_size)

        update_info = agent.update(batch)

        if i % config.log_interval == 0:
            for k, v in update_info.items():
                if v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % config.eval_interval == 0:
            eval_stats = evaluate(agent, env, config.eval_episodes)
            print(eval_stats)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            summary_writer.flush()

            eval_returns.append((i, eval_stats['return']))
            np.savetxt(os.path.join(config.save_dir, f'{config.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    main()