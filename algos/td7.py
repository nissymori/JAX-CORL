# source https://github.com/sfujim/TD7
# https://arxiv.org/abs/2306.02451
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


class TD7Config(BaseModel):
    # GENERAL
    algo: str = "TD3-BC"
    project: str = "train-TD3-BC"
    env_name: str = "halfcheetah-medium-expert-v2"
    seed: int = 42
    eval_episodes: int = 5
    log_interval: int = 100000
    eval_interval: int = 10000
    batch_size: int = 256
    max_steps: int = int(1e6)
    n_jitted_updates: int = 10
    discount: float = 0.99  # discount factor
    target_update_rate: int = 250
    exploration_noise: float = 0.1  # std of exploration noise
    # DATASET
    data_size: int = int(1e6)
    normalize_state: bool = False
    prioritized: bool = True
    # NETWORK
    hidden_dim: int = 256
    zs_dim: int = 256
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    encoder_lr: float = 3e-4
    # TD3
    policy_freq: int = 2  # update actor every policy_freq updates
    policy_noise_std: float = 0.2  # std of policy noise
    policy_noise_clip: float = 0.5  # clip policy noise
    # LAP
    alpha: float = 0.4
    min_priority: float = 1
    # TD3-BC
    lmbda: float = 0.1

    def __hash__(
        self,
    ):  # make config hashable to be specified as static_argnums in jax.jit.
        return hash(self.__repr__())


conf_dict = OmegaConf.from_cli()
config = TD7Config(**conf_dict)


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


def AvgL1Norm(x, eps=1e-8):
    return x / jnp.abs(x).mean(axis=-1, keepdims=True).clip(min=eps)


def LAP_huber(x, min_priority=1):
    return jnp.where(x < min_priority, 0.5 * x**2, min_priority * x).sum(1).mean()


class TD7Actor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, state: jnp.ndarray, zs: jnp.ndarray) -> jnp.ndarray:
        action = AvgL1Norm(nn.Dense(self.hidden_dim)(state))
        action = jnp.concatenate([action, zs], axis=-1)
        action = self.activation(nn.Dense(self.hidden_dim)(action))
        action = self.activation(nn.Dense(self.hidden_dim)(action))
        return nn.tanh(nn.Dense(self.action_dim)(action))


class SEncoder(nn.Module):
    zs_dim: int = 256
    hidden_dim: int = 256
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        zs = self.activation(nn.Dense(self.hidden_dim)(state))
        zs = self.activation(nn.Dense(self.hidden_dim)(zs))
        zs = AvgL1Norm(nn.Dense(self.zs_dim)(zs))
        return zs


class AEncoder(nn.Module):
    zs_dim: int = 256
    hidden_dim: int = 256
    activation: Callable = nn.elu

    @nn.compact
    def __call__(self, zs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        zsa = jnp.concatenate([zs, action], axis=-1)
        zsa = self.activation(nn.Dense(self.hidden_dim)(zsa))
        zsa = self.activation(nn.Dense(self.hidden_dim)(zsa))
        zsa = nn.Dense(self.zs_dim)(zsa)  # no AvgL1Norm for zsa
        return zsa


class Encoder(nn.Module):
    action_dim: int
    zs_dim: int = 256
    hidden_dim: int = 256
    activation: Callable = nn.elu

    def setup(self):
        self.enc = SEncoder(
            zs_dim=self.zs_dim, hidden_dim=self.hidden_dim, activation=self.activation
        )
        self.a_enc = AEncoder(
            zs_dim=self.zs_dim, hidden_dim=self.hidden_dim, activation=self.activation
        )

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        zs = self.encoder(state)
        zsa = self.action_encoder(zs, action)
        return zs, zsa

    def encoder(self, state: jnp.ndarray):
        zs = self.enc(state)
        return zs

    def action_encoder(self, zs: jnp.ndarray, action: jnp.ndarray):
        zsa = self.a_enc(zs, action)
        return zsa


class Critic(nn.Module):
    action_dim: int
    zs_dim: int = 256
    hidden_dim: int = 256
    activation: Callable = nn.elu

    @nn.compact
    def __call__(
        self, state: jnp.ndarray, action: jnp.ndarray, zs: jnp.ndarray, zsa: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([state, action], axis=-1)
        embeddings = jnp.concatenate([zs, zsa], axis=-1)

        q1 = AvgL1Norm(nn.Dense(self.hidden_dim)(sa))
        q1 = jnp.concatenate([q1, embeddings], axis=-1)
        q1 = self.activation(nn.Dense(self.hidden_dim)(q1))
        q1 = self.activation(nn.Dense(self.hidden_dim)(q1))
        q1 = nn.Dense(1)(q1)

        q2 = AvgL1Norm(nn.Dense(self.hidden_dim)(sa))
        q2 = jnp.concatenate([q2, embeddings], axis=-1)
        q2 = self.activation(nn.Dense(self.hidden_dim)(q2))
        q2 = self.activation(nn.Dense(self.hidden_dim)(q2))
        q2 = nn.Dense(1)(q2)
        return jnp.concatenate([q1, q2], axis=-1)


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    priorities: jnp.ndarray


def get_dataset(
    env: gym.Env, config: TD7Config, clip_to_eps: bool = True, eps: float = 1e-5
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
        dones=jnp.array(dones, dtype=jnp.float32),
        next_observations=jnp.array(dataset["next_observations"], dtype=jnp.float32),
        priorities=jnp.ones(len(dataset["observations"]), dtype=jnp.float32),
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


def sample_batch(
    rng: jnp.ndarray, dataset: Transition, batch_size: int, prioritized, data_size: int
):
    if prioritized:
        csum = jnp.cumsum(dataset.priorities, axis=0)
        val = jax.random.uniform(rng, (batch_size,), minval=0, maxval=1) * csum[-1] 
        indices = jnp.searchsorted(csum, val)
        batch = jax.tree_map(lambda x: x[indices], dataset)
    else:
        indices = jax.random.randint(rng, (batch_size,), 0, data_size)
        batch = jax.tree_map(lambda x: x[indices], dataset)
    return batch, indices


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable, has_aux: bool = False
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)
    if has_aux:
        (loss, aux), grad = grad_fn(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grad)
        return new_train_state, loss, aux
    else:
        loss, grad = grad_fn(train_state.params)
        new_train_state = train_state.apply_gradients(grads=grad)
        return new_train_state, loss


class TD7Trainer(NamedTuple):
    actor: TrainState
    critic: TrainState
    encoder: TrainState
    target_actor: TrainState
    target_critic: TrainState
    fixed_encoder: TrainState
    fixed_target_encoder: TrainState
    max_action: float = 1.0
    max_target: float = 0.0
    min_target: float = 0.0
    max_: float = -1e8
    min_: float = 1e8
    max_priority: float = 1.0

    def update_encoder(
        agent, batch: Transition, config: TD7Config
    ) -> Tuple["TD7Trainer", jnp.ndarray]:
        def encoder_loss_fn(
            encoder_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            next_zs = agent.encoder.apply_fn(
                encoder_params, batch.next_observations, method=Encoder.encoder
            )
            next_zs = jax.lax.stop_gradient(next_zs)

            zs = agent.encoder.apply_fn(
                encoder_params, batch.observations, method=Encoder.encoder
            )
            pred_zs = agent.encoder.apply_fn(
                encoder_params,
                zs,
                batch.actions,
                method=Encoder.action_encoder,
            )
            encoder_loss = jnp.square(next_zs - pred_zs).mean()
            return encoder_loss

        new_encoder, encoder_loss = update_by_loss_grad(agent.encoder, encoder_loss_fn)
        return agent._replace(encoder=new_encoder), encoder_loss

    def update_actor(
        agent, batch: Transition, rng: jax.random.PRNGKey, config: TD7Config
    ) -> Tuple["TD7Trainer", jnp.ndarray]:
        def actor_loss_fn(actor_params: flax.core.FrozenDict[str, Any]) -> jnp.ndarray:
            fixed_zs = agent.fixed_encoder.apply_fn(
                agent.fixed_encoder.params,
                batch.observations,
                method=Encoder.encoder,
            )
            action = agent.actor.apply_fn(actor_params, batch.observations, fixed_zs)
            fixed_zsa = agent.fixed_target_encoder.apply_fn(
                agent.fixed_target_encoder.params,
                fixed_zs,
                action,
                method=Encoder.action_encoder,
            )
            q = agent.critic.apply_fn(
                agent.critic.params, batch.observations, action, fixed_zs, fixed_zsa
            )

            actor_loss = (
                -q.mean()
                + config.lmbda
                * jax.lax.stop_gradient(jnp.abs(q).mean())
                * jnp.square(action - batch.actions).mean()
            )
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(
        agent, batch: Transition, rng: jax.random.PRNGKey, config: TD7Config
    ) -> Tuple["TD7Trainer", jnp.ndarray]:
        def critic_loss_fn(
            critic_params: flax.core.FrozenDict[str, Any]
        ) -> jnp.ndarray:
            #######################
            # No grad start
            #######################
            fixed_target_zs = agent.fixed_target_encoder.apply_fn(
                agent.fixed_target_encoder.params,
                batch.next_observations,
                method=Encoder.encoder,
            )

            noise = (
                jax.random.normal(rng, batch.actions.shape) * config.policy_noise_std
            ).clip(-config.policy_noise_clip, config.policy_noise_clip)
            next_action = (
                agent.target_actor.apply_fn(
                    agent.target_actor.params, batch.next_observations, fixed_target_zs
                )
                + noise
            ).clip(-agent.max_action, agent.max_action)
            fixed_target_zsa = agent.fixed_target_encoder.apply_fn(
                agent.fixed_target_encoder.params,
                fixed_target_zs,
                next_action,
                method=Encoder.action_encoder,
            )

            target_q = agent.target_critic.apply_fn(
                agent.target_critic.params,
                batch.next_observations,
                next_action,
                fixed_target_zs,
                fixed_target_zsa,
            )
            target_q = target_q.min(axis=1, keepdims=True)[0]
            target_q = batch.rewards + config.discount * (1.0 - batch.dones) * target_q.clip(min=agent.min_target, max=agent.max_target)
            target_q = jax.lax.stop_gradient(target_q)[..., None]  # (batch_size, 1)

            fixed_zs, fixed_zsa = agent.fixed_encoder.apply_fn(
                agent.fixed_encoder.params, batch.observations, batch.actions
            )
            fized_zs = jax.lax.stop_gradient(fixed_zs)
            fixed_zsa = jax.lax.stop_gradient(fixed_zsa)
            #######################
            # No grad end
            #######################

            pred_q = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions, fixed_zs, fixed_zsa
            )
            td_loss = jnp.abs(target_q - pred_q)  # (batch_size, 2)
            critic_loss = LAP_huber(td_loss)
            return critic_loss, (td_loss, target_q)

        new_critic, critic_loss, (td_loss, target_q) = update_by_loss_grad(
            agent.critic, critic_loss_fn, has_aux=True
        )
        return agent._replace(critic=new_critic), (critic_loss, td_loss, target_q)

    def update_lap(
        agent,
        dataset: Transition,
        indices: jnp.ndarray,
        td_loss: jnp.ndarray,
        config: TD7Config,
    ):
        """
        Update the priorities of the samples corresponding to the given batch indices.
        """
        priority = jnp.power(
            td_loss.max(axis=1)[0].clip(config.min_priority), config.alpha
        )
        new_priority = dataset.priorities.at[indices].set(priority)
        max_priority = jnp.maximum(jnp.max(new_priority), agent.max_priority)
        return agent._replace(max_priority=max_priority), dataset._replace(
            priorities=new_priority
        )

    @partial(jax.jit, static_argnums=(3,))
    def update_n_times(
        agent,
        data: Transition,
        rng: jax.random.PRNGKey,
        config: TD7Config,
    ) -> Tuple["TD7Trainer", Dict]:
        for _ in range(
            config.n_jitted_updates
        ):  # we can jit for roop for static unroll
            # sample batch
            rng, batch_rng = jax.random.split(rng, 2)
            batch, indices = sample_batch(
                batch_rng, data, config.batch_size, config.prioritized, config.data_size
            )
            # update networks
            rng, critic_rng, actor_rng = jax.random.split(rng, 3)
            agent, encoder_loss = agent.update_encoder(batch, config)  # update encoder 
            agent, (critic_loss, td_loss, target_q) = agent.update_critic(
                batch, critic_rng, config
            )  # update critic
            agent = agent._replace(
                max_=jnp.maximum(agent.max_, target_q.max()),
                min_=jnp.minimum(agent.min_, target_q.min()),
            )  # update max and min target
            agent, data = agent.update_lap(data, indices, td_loss, config)  # update LAP
            if _ % config.policy_freq == 0:  # update actor
                agent, actor_loss = agent.update_actor(batch, actor_rng, config)
        return (
            agent,
            data,
            {
                "encoder_loss": encoder_loss,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
            },
        )

    @jax.jit
    def get_actions(
        agent,
        obs: jnp.ndarray,
        max_action: float = 1.0,  # In D4RL, action is scaled to [-1, 1]
    ) -> jnp.ndarray:
        zs = agent.fixed_encoder.apply_fn(
            agent.fixed_encoder.params, obs, method=Encoder.encoder
        )
        action = agent.actor.apply_fn(agent.actor.params, obs, zs)
        action = action.clip(-max_action, max_action)
        return action


def create_trainer(
    observations: jnp.ndarray, actions: jnp.ndarray, config: TD7Config
) -> TD7Trainer:
    rng = jax.random.PRNGKey(config.seed)
    action_dim = actions.shape[-1]
    critic_model = Critic(
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    )
    actor_model = TD7Actor(
        action_dim=action_dim,
        hidden_dim=config.hidden_dim,
    )
    encoder_model = Encoder(
        action_dim=action_dim,
        zs_dim=config.zs_dim,
        hidden_dim=config.hidden_dim,
    )
    rng, critic_rng, actor_rng, encoder_rng = jax.random.split(rng, 4)
    zs = jnp.zeros((config.zs_dim,), dtype=jnp.float32)
    zsa = jnp.zeros((config.zs_dim,), dtype=jnp.float32)
    # initialize critic
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions, zs, zsa),
        tx=optax.adam(config.critic_lr),
    )
    target_critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_model.init(critic_rng, observations, actions, zs, zsa),
        tx=optax.adam(config.critic_lr),
    )
    # initialize actor
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations, zs),
        tx=optax.adam(config.actor_lr),
    )
    target_actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(actor_rng, observations, zs),
        tx=optax.adam(config.actor_lr),
    )
    # initialize encoder
    encoder_train_state: TrainState = TrainState.create(
        apply_fn=encoder_model.apply,
        params=encoder_model.init(encoder_rng, observations, actions),
        tx=optax.adam(config.encoder_lr),
    )
    fixed_encoder_train_state: TrainState = TrainState.create(
        apply_fn=encoder_model.apply,
        params=encoder_model.init(encoder_rng, observations, actions),
        tx=optax.adam(config.encoder_lr),
    )
    fixed_target_encoder_train_state: TrainState = TrainState.create(
        apply_fn=encoder_model.apply,
        params=encoder_model.init(encoder_rng, observations, actions),
        tx=optax.adam(config.encoder_lr),
    )
    return TD7Trainer(
        actor=actor_train_state,
        critic=critic_train_state,
        target_actor=target_actor_train_state,
        target_critic=target_critic_train_state,
        encoder=encoder_train_state,
        fixed_encoder=fixed_encoder_train_state,
        fixed_target_encoder=fixed_target_encoder_train_state,
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
    wandb.init(project=config.project, config=config)
    env = gym.make(config.env_name)
    rng = jax.random.PRNGKey(config.seed)
    dataset, obs_mean, obs_std = get_dataset(env, config)
    # create agent
    example_batch: Transition = jax.tree_map(lambda x: x[0], dataset)
    agent = create_trainer(example_batch.observations, example_batch.actions, config)

    num_steps = config.max_steps // config.n_jitted_updates
    assert config.target_update_rate % config.n_jitted_updates == 0
    target_update_rate = config.target_update_rate // config.n_jitted_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, update_rng = jax.random.split(rng)
        agent, dataset, update_info = agent.update_n_times(
            dataset,
            update_rng,
            config,
        )  # update parameters
        if i % target_update_rate == 0:  # update target networks
            agent = agent._replace(
                target_critic=agent.critic,
                target_actor=agent.actor,
                fixed_encoder=agent.encoder,
                fixed_target_encoder=agent.fixed_encoder,
                max_priority=dataset.priorities.max(),
                max_target=agent.max_,
                min_target=agent.min_,
            )
        if i % config.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % config.eval_interval == 0:
            policy_fn = agent.get_actions
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
    # final evaluation
    policy_fn = agent.get_actions
    normalized_score = evaluate(
        policy_fn,
        env,
        num_episodes=config.eval_episodes,
        obs_mean=obs_mean,
        obs_std=obs_std,
    )
    print("Final Evaluation Score:", normalized_score)
    wandb.log({f"{config.env_name}/final_normalized_score": normalized_score})
    wandb.finish()
