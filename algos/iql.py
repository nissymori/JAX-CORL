# source https://github.com/ikostrikov/implicit_q_learning
# https://arxiv.org/abs/2110.06169
import os
import time
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

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "


class IQLConfig(BaseModel):
    # GENERAL
    algo: str = "IQL"
    project: str = "train-IQL"
    env_name: str = "halfcheetah-medium-expert-v2"
    seed: int = 42
    eval_episodes: int = 5
    log_interval: int = 100000
    eval_interval: int = 10000
    batch_size: int = 256
    max_steps: int = int(1e6)
    n_jitted_updates: int = 8
    # DATASET
    data_size: int = int(1e6)
    normalize_state: bool = False
    # TRAINING
    hidden_dims: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    # IQL SPECIFIC
    expectile: float = 0.7  # FYI: for Hopper-me, 0.5 produce better result from CORL
    temperature: float = 3.0  # FYI: for Hopper-me, 6.0 produce better result from CORL
    tau: float = 0.005
    discount: float = 0.99

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = IQLConfig(**conf_dict)


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
        return x


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    split_rngs = kwargs.pop("split_rngs", {})
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={**split_rngs, "params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class GaussianPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    final_fc_init_scale: float = 1e-3

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
        )(observations)

        means = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )(outputs)
        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        return distribution


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray


def get_dataset(
    env: gym.Env, config: IQLConfig, clip_to_eps: bool = True, eps: float = 1e-5
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
    dataset = jax.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= data_size
    dataset = jax.tree_map(lambda x: x[:data_size], dataset)
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


def expectile_loss(diff, expectile=0.8) -> jnp.ndarray:
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def target_update(
    model: TrainState, target_model: TrainState, tau: float
) -> TrainState:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class IQLTrainer(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState

    def update_critic(
        agent, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainer", Dict]:
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            next_v = agent.value.apply_fn(agent.value.params, batch.next_observations)
            target_q = batch.rewards + config.discount * (1 - batch.dones) * next_v
            q1, q2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, critic_loss_fn)
        return agent._replace(critic=new_critic), critic_loss

    def update_value(
        agent, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainer", Dict]:
        def value_loss_fn(value_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            q1, q2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.observations, batch.actions
            )
            q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
            v = agent.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, config.expectile).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(agent.value, value_loss_fn)
        return agent._replace(value=new_value), value_loss

    def update_actor(
        agent, batch: Transition, config: IQLConfig
    ) -> Tuple["IQLTrainer", Dict]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            v = agent.value.apply_fn(agent.value.params, batch.observations)
            q1, q2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, batch.actions
            )
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * config.temperature)
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = agent.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions)
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    @partial(jax.jit, static_argnums=(3,))
    def update_n_times(
        agent,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        config: IQLConfig,
    ) -> Tuple["IQLTrainer", Dict]:
        for _ in range(config.n_jitted_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (config.batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_map(lambda x: x[batch_indices], dataset)

            agent, value_loss = agent.update_value(batch, config)
            agent, actor_loss = agent.update_actor(batch, config)
            agent, critic_loss = agent.update_critic(batch, config)
            new_target_critic = target_update(
                agent.critic, agent.target_critic, config.tau
            )
        return agent._replace(target_critic=new_target_critic), {
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
        }

    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
        max_action: float = 1.0,  # In D4RL, the action space is [-1, 1]
    ) -> jnp.ndarray:
        actions = agent.actor.apply_fn(
            agent.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -max_action, max_action)
        return actions


def create_trainer(
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    config: IQLConfig,
) -> IQLTrainer:
    rng = jax.random.PRNGKey(config.seed)
    rng, actor_rng, critic_rng, value_rng = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = GaussianPolicy(
        config.hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
    )
    schedule_fn = optax.cosine_decay_schedule(-config.actor_lr, config.max_steps)
    actor_tx = optax.chain(optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn))
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations),
        tx=actor_tx,
    )
    # initialize critic
    critic_model = ensemblize(Critic, num_qs=2)(config.hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    # initialize value
    value_model = ValueCritic(config.hidden_dims)
    value = TrainState.create(
        apply_fn=value_model.apply,
        params=value_model.init(value_rng, observations),
        tx=optax.adam(learning_rate=config.value_lr),
    )
    return IQLTrainer(
        rng,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
    )


def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, obs_mean: float, obs_std: float
) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, done = env.reset(), False
        while not done:
            observation = (observation - obs_mean) / (obs_std + 1e-5)
            action = policy_fn(observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    return env.get_normalized_score(np.mean(episode_returns)) * 100


def get_normalization(dataset: Transition) -> float:
    # into numpy.ndarray
    dataset = jax.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


if __name__ == "__main__":
    wandb.init(config=config, project=config.project)
    rng = jax.random.PRNGKey(config.seed)
    env = gym.make(config.env_name)
    dataset, obs_mean, obs_std = get_dataset(env, config)

    normalizing_factor = get_normalization(dataset)
    dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)
    # create agent
    example_batch: Transition = jax.tree_map(lambda x: x[0], dataset)
    agent: IQLTrainer = create_trainer(
        example_batch.observations,
        example_batch.actions,
        config,
    )

    num_steps = config.max_steps // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        agent, update_info = agent.update_n_times(dataset, subkey, config)
        if i % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % config.eval_interval == 0:
            policy_fn = partial(
                agent.sample_actions, temperature=0.0, seed=jax.random.PRNGKey(0)
            )
            normalized_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.eval_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )
            print(i, normalized_score)
            eval_metrics = {f"{config.env_name}/normalized_score": normalized_score}
            wandb.log(eval_metrics, step=i)
    wandb.finish()
