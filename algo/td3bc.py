import time
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
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

Params = flax.core.FrozenDict[str, Any]


class TD3BCConfig(BaseModel):
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
    # NETWORK
    hidden_dims: Sequence[int] = (256, 256)
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    # TD3-BC SPECIFIC
    policy_freq: int = 2  # update actor every policy_freq updates
    alpha: float = 2.5  # BC loss weight
    policy_noise_std: float = 0.2  # std of policy noise
    policy_noise_clip: float = 0.5  # clip policy noise
    tau: float = 0.005  # target network update rate
    discount: float = 0.99  # discount factor
    disable_wandb: bool = True


conf_dict = OmegaConf.from_cli()
config = TD3BCConfig(**conf_dict)


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    add_layer_norm: bool = False
    layer_norm_final: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dims in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dims, kernel_init=self.kernel_init)(x)
            if self.add_layer_norm:  # Add layer norm after activation
                if self.layer_norm_final or i + 1 < len(self.hidden_dims):
                    x = nn.LayerNorm()(x)
            if (
                i + 1 < len(self.hidden_dims) or self.activate_final
            ):  # Add activation after layer norm
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        q2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(x)
        return q1, q2


class TD3Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    max_action: float = 1.0  # In D4RL, action is scaled to [-1, 1]

    @nn.compact
    def __call__(self, observation: jnp.ndarray) -> jnp.ndarray:
        action = MLP((*self.hidden_dims, self.action_dim))(observation)
        action = self.max_action * jnp.tanh(
            action
        )  # scale to [-max_action, max_action]
        return action


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    rewards: jnp.ndarray
    dones_float: jnp.ndarray
    masks: jnp.ndarray


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

    # normalize states
    obs_mean = dataset.observations.mean(0)
    obs_std = dataset.observations.std(0) + 1e-6
    dataset = dataset._replace(
        observations=(dataset.observations - obs_mean) / obs_std,
        next_observations=(dataset.next_observations - obs_mean) / obs_std,
    )
    return dataset, obs_mean, obs_std


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


class TD3BCTrainer(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_actor: TrainState
    target_critic: TrainState
    update_idx: jnp.int32
    config: flax.core.FrozenDict
    max_action: float = 1.0

    def update_actor(
        agent, batch: Transition, rng: jax.random.PRNGKey
    ) -> Tuple["TD3BCTrainer", jnp.ndarray]:
        def actor_loss_fn(actor_params: Params) -> jnp.ndarray:
            predicted_action = agent.actor.apply_fn(actor_params, batch.observations)
            critic_params = jax.lax.stop_gradient(agent.critic.params)
            q_value, _ = agent.critic.apply_fn(
                critic_params, batch.observations, predicted_action
            )

            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
            loss_lambda = agent.config["alpha"] / mean_abs_q

            bc_loss = jnp.square(predicted_action - batch.actions).mean()
            loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
            return loss_actor

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(
        agent, batch: Transition, rng: jax.random.PRNGKey
    ) -> Tuple["TD3BCTrainer", jnp.ndarray]:
        def critic_loss_fn(critic_params: Params) -> jnp.ndarray:
            q_pred_1, q_pred_2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            target_next_action = agent.target_actor.apply_fn(
                agent.target_actor.params, batch.next_observations
            )
            policy_noise = (
                agent.config["policy_noise_std"]
                * agent.max_action
                * jax.random.normal(rng, batch.actions.shape)
            )
            target_next_action = target_next_action + policy_noise.clip(
                -agent.config["policy_noise_clip"], agent.config["policy_noise_clip"]
            )
            target_next_action = target_next_action.clip(
                -agent.max_action, agent.max_action
            )
            q_next_1, q_next_2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.next_observations, target_next_action
            )
            target = (
                batch.rewards[..., None]
                + agent.config["discount"]
                * jnp.minimum(q_next_1, q_next_2)
                * batch.masks[..., None]
            )
            target = jax.lax.stop_gradient(target)  # stop gradient for target
            value_loss_1 = jnp.square(q_pred_1 - target)
            value_loss_2 = jnp.square(q_pred_2 - target)
            value_loss = (value_loss_1 + value_loss_2).mean()
            return value_loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, critic_loss_fn)
        return agent._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(3, 4, 5))
    def update_n_times(
        agent,
        data: Transition,
        rng: jax.random.PRNGKey,
        policy_freq: int,
        batch_size: int,
        n: int,
    ) -> Tuple["TD3BCTrainer", Dict]:
        for _ in range(n):
            rng, batch_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, len(data.observations)
            )
            batch: Transition = jax.tree_map(lambda x: x[batch_idx], data)
            rng, critic_rng, actor_rng = jax.random.split(rng, 3)
            agent, critic_loss = agent.update_critic(batch, critic_rng)
            if _ % policy_freq == 0:
                agent, actor_loss = agent.update_actor(batch, actor_rng)
                new_target_critic = target_update(
                    agent.critic, agent.target_critic, agent.config["tau"]
                )
                new_target_actor = target_update(
                    agent.actor, agent.target_actor, agent.config["tau"]
                )
                agent = agent._replace(
                    target_critic=new_target_critic,
                    target_actor=new_target_actor,
                )
        return agent._replace(update_idx=agent.update_idx + 1), {}

    @jax.jit
    def get_actions(
        agent,
        obs: jnp.ndarray,
    ) -> jnp.ndarray:
        action = agent.actor.apply_fn(agent.actor.params, obs)
        action = action.clip(-1.0, 1.0)
        return action


def create_trainer(
    observations: jnp.ndarray, actions: jnp.ndarray, config
) -> TD3BCTrainer:
    rng = jax.random.PRNGKey(config.seed)
    critic_model = DoubleCritic(
        hidden_dims=config.hidden_dims,
    )
    action_dim = actions.shape[-1]
    actor_model = TD3Actor(
        action_dim=action_dim,
        hidden_dims=config.hidden_dims,
    )
    rng, critic_rng, actor_rng = jax.random.split(rng, 3)
    # initialize critic and actor parameters
    critic_params = critic_model.init(critic_rng, observations, actions)
    critic_params_target = critic_model.init(critic_rng, observations, actions)

    actor_params = actor_model.init(actor_rng, observations)
    actor_params_target = actor_model.init(actor_rng, observations)

    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(config.critic_lr),
    )
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params_target,
        tx=optax.adam(config.critic_lr),
    )
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )
    target_actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params_target,
        tx=optax.adam(config.actor_lr),
    )

    config = flax.core.FrozenDict(
        dict(
            alpha=config.alpha,
            policy_noise_std=config.policy_noise_std,
            policy_noise_clip=config.policy_noise_clip,
            discount=config.discount,
            tau=config.tau,
            batch_size=config.batch_size,
            policy_freq=config.policy_freq,
        )
    )
    return TD3BCTrainer(
        actor=actor_train_state,
        critic=critic_train_state,
        target_actor=target_actor_train_state,
        target_critic=target_critic_train_state,
        update_idx=0,
        config=config,
    )


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
            action = policy_fn(observation)
            observation, reward, done, info = env.step(action)
            episode_return += reward
        episode_returns.append(episode_return)
    return env.get_normalized_score(np.mean(episode_returns)) * 100


if __name__ == "__main__":
    env = gym.make(config.env_name)

    rng = jax.random.PRNGKey(config.seed)
    # initialize data and update state
    dataset, obs_mean, obs_std = get_dataset(env, config)
    example_batch: Transition = jax.tree_map(lambda x: x[0], dataset)
    agent = create_trainer(example_batch.observations, example_batch.actions, config)

    wandb.init(project="train-TD3-BC", config=config)
    epochs = int(
        config.max_steps // config.n_updates
    )  # we update multiple times per epoch
    steps = 0
    for _ in tqdm(range(epochs)):
        steps += 1
        rng, update_rng = jax.random.split(rng)
        agent, update_info = agent.update_n_times(
            dataset,
            update_rng,
            config.policy_freq,
            config.batch_size,
            config.n_updates,
        )  # update parameters
        if _ % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            if not config.disable_wandb:
                wandb.log(train_metrics, step=i)

        if _ % config.eval_interval == 0:
            policy_fn = agent.get_actions
            normalized_score = evaluate(
                policy_fn,
                env,
                num_episodes=config.eval_episodes,
                obs_mean=obs_mean,
                obs_std=obs_std,
            )
            print(_, normalized_score)
            eval_metrics = {"normalized_score": normalized_score}
            if not config.disable_wandb:
                wandb.log(eval_metrics, step=i)
