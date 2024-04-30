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


class AWACConfig(BaseModel):
    # GENERAL
    env_name: str = "halfcheetah-medium-expert-v2"
    seed: int = np.random.choice(1000000)
    data_size: int = int(1e6)
    eval_episodes: int = 10
    log_interval: int = 100000
    eval_interval: int = 10000
    save_interval: int = 25000
    batch_size: int = 256
    max_steps: int = int(1e6)
    n_updates: int = 8
    # TRAINING
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    actor_hidden_dims: Tuple[int, int] = (256, 256, 256, 256)
    critic_hidden_dims: Tuple[int, int] = (256, 256)
    discount: float = 0.99
    tau: float = 0.005
    beta: float = 1.0
    target_update_freq: int = 1
    exp_adv_max: float = 100.0
    disable_wandb: bool = True


conf_dict = OmegaConf.from_cli()
config = AWACConfig(**conf_dict)


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
            if (
                i + 1 < len(self.hidden_dims) or self.activate_final
            ):  # Add activation after layer norm
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

class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20.0
    log_std_max: Optional[float] = 2.0
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


class AWACTrainer(NamedTuple):
    rng: jax.random.PRNGKey
    critic: TrainState
    target_critic: TrainState
    actor: TrainState
    config: flax.core.FrozenDict

    def update_actor(agent, batch: Transition, rng: jax.random.PRNGKey):
        def get_actor_loss(actor_params):
            dist = agent.actor.apply_fn(actor_params, batch.observations)
            pi_actions = dist.sample(seed=rng)
            q_1, q_2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, pi_actions
            )
            v = jnp.minimum(q_1, q_2)

            lim = 1 - 1e-5
            actions = jnp.clip(batch.actions, -lim, lim)
            q_1, q_2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, actions
            )
            q = jnp.minimum(q_1, q_2)
            adv = q - v
            weights = jnp.clip(
                jnp.exp(adv / agent.config["beta"]), 0, agent.config["exp_adv_max"]
            )
            weights = jax.lax.stop_gradient(weights)

            log_prob = dist.log_prob(batch.actions)
            loss = -jnp.mean(log_prob * weights).mean()
            return loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, get_actor_loss)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(agent, batch: Transition, rng: jax.random.PRNGKey):
        def get_critic_loss(critic_params):
            dist = agent.actor.apply_fn(agent.actor.params, batch.observations)
            next_actions = dist.sample(seed=rng)
            n_q_1, n_q_2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.next_observations, next_actions
            )
            next_q = jnp.minimum(n_q_1, n_q_2)
            q_target = (
                batch.rewards
                + agent.config["discount"] * (1.0 - batch.dones_float) * next_q
            )
            q_target = jax.lax.stop_gradient(q_target)

            q_1, q_2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )

            loss = jnp.mean((q_1 - q_target) ** 2 + (q_2 - q_target) ** 2)
            return loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, get_critic_loss)
        return agent._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(3, 4, 5))
    def update_n_times(
        agent,
        dataset: Transition,
        rng: jax.random.PRNGKey,
        batch_size: int,
        target_update_freq: int,
        n_updates: int,
    ):
        for _ in range(n_updates):
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            batch_indices = jax.random.randint(
                batch_rng, (batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_map(lambda x: x[batch_indices], dataset)

            agent, critic_loss = agent.update_critic(batch, critic_rng)
            new_target_critic = target_update(
                agent.critic,
                agent.target_critic,
                agent.config["target_update_rate"],
            )
            agent, actor_loss = agent.update_actor(batch, actor_rng)
        return agent._replace(target_critic=new_target_critic), {}  # TODO return losses

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
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions


def create_trainer(
    observations: jnp.ndarray, actions: jnp.ndarray, config: AWACConfig
) -> AWACTrainer:
    rng = jax.random.PRNGKey(config.seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)
    # initialize actor
    action_dim = actions.shape[-1]
    actor_model = NormalTanhPolicy(
        config.actor_hidden_dims,
        action_dim=action_dim,
    )

    actor_params = actor_model.init(actor_key, observations)
    actor = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=config.actor_lr),
    )
    # initialize critic
    critic_model = ensemblize(Critic, num_qs=2)(hidden_dims=config.critic_hidden_dims)
    critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_key, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_key, observations, actions),
        tx=optax.adam(learning_rate=config.critic_lr),
    )
    # create immutable config for AWAC.
    config = flax.core.FrozenDict(
        dict(
            discount=config.discount,
            beta=config.beta,
            target_update_rate=config.tau,
            exp_adv_max=config.exp_adv_max,
        )
    )  # make sure config is immutable
    return AWACTrainer(
        rng,
        critic=critic,
        target_critic=target_critic,
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
            observation, rew, done, info = env.step(action)
            episode_return += rew
        episode_returns.append(episode_return)
    return env.get_normalized_score(np.mean(episode_returns)) * 100


if __name__ == "__main__":
    if not config.disable_wandb:
        wandb.init(config=config, project="AWAC")
    rng = jax.random.PRNGKey(config.seed)
    env = gym.make(config.env_name)
    dataset = get_dataset(env, config)

    example_batch: Transition = jax.tree_map(lambda x: x[0], dataset)
    agent: AWACTrainer = create_trainer(
        example_batch.observations,
        example_batch.actions,
        config,
    )

    num_steps = config.max_steps // config.n_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        agent, update_info = agent.update_n_times(
            dataset,
            subkey,
            config.batch_size,
            config.target_update_freq,
            config.n_updates,
        )
        if i % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if not config.disable_wandb:
                wandb.log(train_metrics, step=i)

        if i % config.eval_interval == 0:
            policy_fn = partial(
                agent.sample_actions, temperature=0.0, seed=jax.random.PRNGKey(0)
            )
            normalized_score = evaluate(policy_fn, env, config.eval_episodes)
            print(i, normalized_score)
            eval_metrics = {"normalized_score": normalized_score}
            if not config.disable_wandb:
                wandb.log(eval_metrics, step=i)
