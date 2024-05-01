from functools import partial
from typing import (Any, Callable, Dict, NamedTuple, Optional, Sequence, Tuple,
                    Union)

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

Params = flax.core.FrozenDict[str, Any]


class IQLConfig(BaseModel):
    # GENERAL
    env_name: str = "hopper-medium-expert-v2"
    seed: int = 42
    data_size: int = int(1e6)
    eval_episodes: int = 5
    log_interval: int = 100000
    eval_interval: int = 10000
    batch_size: int = 256
    max_steps: int = int(1e6)
    n_updates: int = 8
    # TRAINING
    hidden_dims: Tuple[int, int] = (256, 256)
    actor_lr: float = 3e-4
    value_lr: float = 3e-4
    critic_lr: float = 3e-4
    # IQL SPECIFIC
    expectile: float = 0.7  # for Hopper 0.5
    temperature: float = 3.0  # for Hopper 6.0
    tau: float = 0.005
    discount: float = 0.99
    disable_wandb: bool = True


conf_dict = OmegaConf.from_cli()
config = IQLConfig(**conf_dict)


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
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
    """
    Ensemblize a module by creating `num_qs` instances of the module
    """
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
    masks: jnp.ndarray
    dones_float: jnp.ndarray
    next_observations: jnp.ndarray


def get_dataset(
    env: gym.Env, config, clip_to_eps: bool = True, eps: float = 1e-5
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
    dones_float = 1.0 - same_obs.astype(np.float32)
    dones_float[-1] = 1

    dataset = Transition(
        observations=jnp.array(dataset["observations"], dtype=jnp.float32),
        actions=jnp.array(dataset["actions"], dtype=jnp.float32),
        rewards=jnp.array(dataset["rewards"], dtype=jnp.float32),
        masks=jnp.array(1.0 - dones_float, dtype=jnp.float32),
        dones_float=jnp.array(dones_float, dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
    )

    # shuffle data and select the first data_size samples
    rng = jax.random.PRNGKey(config.seed)
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(dataset.observations))
    dataset = jax.tree_map(lambda x: x[perm], dataset)
    assert len(dataset.observations) >= config.data_size
    dataset = jax.tree_map(lambda x: x[: config.data_size], dataset)
    return dataset


def expectile_loss(diff, expectile=0.8):
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
) -> Tuple[float, Params]:
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
    config: flax.core.FrozenDict

    def update_critic(agent, batch: Transition) -> Tuple["IQLTrainer", Dict]:
        def critic_loss_fn(critic_params):
            next_v = agent.value.apply_fn(agent.value.params, batch.next_observations)
            target_q = batch.rewards + agent.config["discount"] * batch.masks * next_v
            q1, q2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, critic_loss_fn)
        return agent._replace(critic=new_critic), critic_loss

    def update_value(agent, batch: Transition) -> Tuple["IQLTrainer", Dict]:
        def value_loss_fn(value_params):
            q1, q2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.observations, batch.actions
            )
            q = jax.lax.stop_gradient(jnp.minimum(q1, q2))
            v = agent.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, agent.config["expectile"]).mean()
            return value_loss

        new_value, value_loss = update_by_loss_grad(agent.value, value_loss_fn)
        return agent._replace(value=new_value), value_loss

    def update_actor(agent, batch: Transition) -> Tuple["IQLTrainer", Dict]:
        def actor_loss_fn(actor_params):
            v = agent.value.apply_fn(agent.value.params, batch.observations)
            q1, q2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, batch.actions
            )
            q = jnp.minimum(q1, q2)
            exp_a = jnp.exp((q - v) * agent.config["temperature"])
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = agent.actor.apply_fn(actor_params, batch.observations)
            log_probs = dist.log_prob(batch.actions)
            actor_loss = -(exp_a * log_probs).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    @partial(jax.jit, static_argnums=(3, 4))
    def update_n_times(
        agent,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        batch_size: int,
        n_updates: int,
    ):
        for _ in range(n_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_map(lambda x: x[batch_indices], dataset)

            agent, value_loss = agent.update_value(batch)
            agent, actor_loss = agent.update_actor(batch)
            agent, critic_loss = agent.update_critic(batch)
            new_target_critic = target_update(
                agent.critic, agent.target_critic, agent.config["target_update_rate"]
            )
        return agent._replace(target_critic=new_target_critic), {} 

    @jax.jit
    def sample_actions(
        agent,
        observations: np.ndarray,
        *,
        seed: jax.random.PRNGKey,
        temperature: float = 1.0,
    ) -> jnp.ndarray:
        actions = agent.actor.apply_fn(
            agent.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
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
    # create immutable config for IQL.
    config = flax.core.FrozenDict(
        dict(
            discount=config.discount,
            temperature=config.temperature,
            expectile=config.expectile,
            target_update_rate=config.tau,
        )
    )  # make sure config is immutable
    return IQLTrainer(
        rng,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
        config=config,
    )


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> float:
    episode_returns = []
    for _ in range(num_episodes):
        episode_return = 0
        observation, done = env.reset(), False
        while not done:
            action = policy_fn(observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    return env.get_normalized_score(np.mean(episode_returns)) * 100


def get_normalization(dataset: Transition):
    # into numpy.ndarray
    dataset = jax.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones_float):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


if __name__ == "__main__":
    if not config.disable_wandb:
        wandb.init(config=config, project="iql")
    rng = jax.random.PRNGKey(config.seed)
    env = gym.make(config.env_name)
    dataset: Transition = get_dataset(env, config)

    normalizing_factor = get_normalization(dataset)
    dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)

    example_batch: Transition = jax.tree_map(lambda x: x[0], dataset)
    agent: IQLTrainer = create_trainer(
        example_batch.observations,
        example_batch.actions,
        config,
    )

    num_steps = config.max_steps // config.n_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        agent, update_info = agent.update_n_times(
            dataset, subkey, config.batch_size, config.n_updates
        )
        if i % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if not config.disable_wandb:
                wandb.log(train_metrics, step=i)

        if i % config.eval_interval == 0:
            policy_fn = partial(
                agent.sample_actions, temperature=0.0, seed=jax.random.PRNGKey(0)
            )
            normalized_score = evaluate(
                policy_fn, env, num_episodes=config.eval_episodes
            )
            print(i, normalized_score)
            eval_metrics = {"normalized_score": normalized_score}
            if not config.disable_wandb:
                wandb.log(eval_metrics, step=i)
