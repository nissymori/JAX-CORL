import copy
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


@dataclass
class SACNConfig:
    # wandb params
    project: str = "SAC-N-JAX"
    group: str = "SAC-N"
    name: str = "sac-n-jax-flax"
    # model params
    hidden_dim: int = 256
    num_critics: int = 10
    gamma: float = 0.99
    tau: float = 5e-3
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4
    data_size: int = 1000000
    normalize_state: bool = False
    # training params
    env_name: str = "halfcheetah-medium-expert-v2"
    batch_size: int = 256
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 50
    # general params
    seed: int = 10
    eval_seed: int = 42

    def __post_init__(self):
        self.name = f"{self.name}-{self.env_name}-{str(uuid.uuid4())[:8]}"


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


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def get_dataset(
    env: gym.Env, config: SACNConfig, clip_to_eps: bool = True, eps: float = 1e-5
) -> Transition:
    dataset = d4rl.qlearning_dataset(env)

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
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        dones=jnp.array(dones, dtype=jnp.float32),
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


class SACNTrainState(NamedTuple):
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState

# SAC-N losses
def update_actor(
        train_state: SACNTrainState,
        batch: Dict[str, jax.Array],
        rng: jax.random.PRNGKey,
        config: SACNConfig
) -> Tuple[SACNTrainState, Dict[str, Any]]:
    def actor_loss_fn(actor_params):
        actions_dist = train_state.actor.apply_fn(actor_params, batch.observations)
        actions, actions_logp = actions_dist.sample_and_log_prob(seed=rng)

        q_values = train_state.critic.apply_fn(train_state.critic.params, batch.observations, actions).min(0)
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
        train_state: SACNTrainState,
        batch: Dict[str, jax.Array],
        rng: jax.random.PRNGKey,
        config: SACNConfig
) -> Tuple[SACNTrainState, Dict[str, Any]]:
    next_actions_dist = train_state.actor.apply_fn(train_state.actor.params, batch.next_observations)
    next_actions, next_actions_logp = next_actions_dist.sample_and_log_prob(seed=rng)

    next_q = train_state.critic.apply_fn(train_state.critic.target_params, batch.next_observations, next_actions).min(0)
    next_q = next_q - train_state.alpha.apply_fn(train_state.alpha.params) * next_actions_logp.sum(-1)
    target_q = batch.rewards + (1 - batch.dones) * config.gamma * next_q

    def critic_loss_fn(critic_params):
        # [N, batch_size] - [1, batch_size]
        q = train_state.critic.apply_fn(critic_params, batch.observations, batch.actions)
        loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
        return loss

    loss, grads = jax.value_and_grad(critic_loss_fn)(train_state.critic.params)
    new_critic = train_state.critic.apply_gradients(grads=grads).soft_update(tau=config.tau)
    info = {
        "critic_loss": loss
    }
    return train_state._replace(critic=new_critic), info


# evaluation
@jax.jit
def eval_actions_jit(actor: TrainState, obs: jax.Array) -> jax.Array:
    dist = actor.apply_fn(actor.params, obs)
    action = dist.mean()
    return action


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


@pyrallis.wrap()
def main(config: SACNConfig):
    wandb.init(
        config=asdict(config),
        project=config.project,
        group=config.group,
        name=config.name,
        id=str(uuid.uuid4()),
        save_code=True
    )

    env = gym.make(config.env_name)
    dataset, obs_mean, obs_std = get_dataset(env, config)
    target_entropy = -np.prod(env.action_space.shape)
    config.target_entropy = target_entropy

    example_batch: Transition = jax.tree_util.tree_map(lambda x: x[0], dataset)
    train_state = create_train_state(example_batch.observations, example_batch.actions, config)

    def update_networks(key, train_state, batch, config):
        actor_key, critic_key = jax.random.split(key)

        train_state, actor_info = update_actor(train_state, batch, actor_key, config)
        train_state, alpha_info = update_alpha(train_state, actor_info["batch_entropy"], config)
        train_state, critic_info = update_critic(train_state, batch, critic_key, config)

        return train_state, {**actor_info, **critic_info, **alpha_info}

    @jax.jit
    def update_step(_, carry):
        key, update_key, batch_key = jax.random.split(carry["key"], 3)
        dataset = carry["dataset"]
        indices = jax.random.randint(batch_key, shape=(config.batch_size,), minval=0, maxval=len(dataset.observations))
        batch = jax.tree_util.tree_map(lambda x: x[indices], dataset)

        train_state, update_info = update_networks(
            key=update_key,
            train_state=carry["train_state"],
            batch=batch,
            config=config
        )
        update_info = jax.tree_map(lambda c, u: c + u, carry["update_info"], update_info)
        carry.update(key=key, train_state=train_state, update_info=update_info)

        return carry

    update_carry = {
        "key": jax.random.PRNGKey(config.seed),
        "train_state": train_state,
        "dataset": dataset,
    }
    for epoch in trange(config.num_epochs):
        # metrics for accumulation during epoch and logging to wandb, we need to reset them every epoch
        update_carry["update_info"] = {
            "critic_loss": jnp.array([0.0]),
            "actor_loss": jnp.array([0.0]),
            "alpha_loss": jnp.array([0.0]),
            "alpha": jnp.array([0.0]),
            "batch_entropy": jnp.array([0.0])
        }
        update_carry = jax.lax.fori_loop(
            lower=0,
            upper=config.num_updates_on_epoch,
            body_fun=update_step,
            init_val=update_carry
        )
        # log mean over epoch for each metric
        update_info = jax.tree_map(lambda v: v.item() / config.num_updates_on_epoch, update_carry["update_info"])
        wandb.log({"epoch": epoch, **update_info})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs - 1:
            eval_returns = evaluate(env, train_state.actor, config.eval_episodes, seed=config.eval_seed)
            normalized_score = env.get_normalized_score(eval_returns) * 100.0

            wandb.log({
                "epoch": epoch,
                "eval/normalized_score_mean": np.mean(normalized_score),
                "eval/normalized_score_std": np.std(normalized_score)
            })
            print(f"Epoch {epoch}, eval/normalized_score_mean: {np.mean(normalized_score)}, eval/normalized_score_std: {np.std(normalized_score)}")


if __name__ == "__main__":
    main()