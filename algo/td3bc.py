import time
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import d4rl
import flax
import flax.linen as nn
import gym
import numpy as np
import jax
import jax.numpy as jnp
import optax
import tqdm
import wandb
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm


class TD3BCConfig:
    # general config
    env_name: str = "hopper"
    data_quality: str = "medium-expert"
    total_updates: int = 1000000
    updates_per_epoch: int = 100000  # how many updates per epoch. it is equivalent to how frequent we evaluate the policy
    num_test_rollouts: int = 5
    batch_size: int = 256
    buffer_size: int = 1000000
    seed: int = 0
    # network config
    num_hidden_layers: int = 2
    num_hidden_units: int = 256
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    # TD3-BC specific
    policy_freq: int = 2  # update actor every policy_freq updates
    polyak: float = 0.995  # target network update rate
    alpha: float = 2.5  # BC loss weight
    policy_noise_std: float = 0.2  # std of policy noise
    policy_noise_clip: float = 0.5  # clip policy noise
    gamma: float = 0.99  # discount factor


config = TD3BCConfig(**OmegaConf.to_object(OmegaConf.from_cli()))

def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class DoubleCritic(nn.Module):
    """
    For twin Q networks
    """
    @nn.compact
    def __call__(self, state, action, rng):
        sa = jnp.concatenate([state, action], axis=-1)
        x_q = nn.Dense(config.num_hidden_units, kernel_init=default_init())(sa)
        x_q = nn.LayerNorm()(x_q)
        x_q = nn.relu(x_q)
        for i in range(1, config.num_hidden_layers):
            x_q = nn.Dense(config.num_hidden_units, kernel_init=default_init())(x_q)
            x_q = nn.LayerNorm()(x_q)
            x_q = nn.relu(x_q)
        q1 = nn.Dense(1, kernel_init=default_init())(x_q)

        x_q = nn.Dense(config.num_hidden_units, kernel_init=default_init())(sa)
        x_q = nn.LayerNorm()(x_q)
        x_q = nn.relu(x_q)
        for i in range(1, config.num_hidden_layers):
            x_q = nn.Dense(config.num_hidden_units, kernel_init=default_init())(x_q)
            x_q = nn.LayerNorm()(x_q)
            x_q = nn.relu(x_q)
        q2 = nn.Dense(1)(x_q)
        return q1, q2


class TD3Actor(nn.Module):
    action_dim: int
    max_action: float

    @nn.compact
    def __call__(self, state, rng):
        x_a = nn.relu(
            nn.Dense(config.num_hidden_units, kernel_init=default_init())(state)
        )
        for i in range(1, config.num_hidden_layers):
            x_a = nn.Dense(config.num_hidden_units, kernel_init=default_init())(x_a)
            x_a = nn.relu(x_a)
        action = nn.Dense(action_dim, kernel_init=default_init())(x_a)
        action = self.max_action * jnp.tanh(action) # scale to [-max_action, max_action]
        return action


class TD3BCUpdateState(NamedTuple):
    critic: TrainState
    actor: TrainState
    critic_params_target: flax.core.FrozenDict
    actor_params_target: flax.core.FrozenDict
    update_idx: jnp.int32


def initialize_update_state(observation_dim, action_dim, max_action, rng) -> TD3BCUpdateState:
    critic_model = DoubleCritic()
    actor_model = TD3Actor(action_dim=action_dim, max_action=max_action)
    rng, rng1, rng2 = jax.random.split(rng, 3)
    # initialize critic and actor parameters
    critic_params = critic_model.init(
        rng1, state=jnp.zeros(observation_dim), action=jnp.zeros(action_dim), rng=rng1
    )
    critic_params_target = critic_model.init(
        rng1, jnp.zeros(observation_dim), jnp.zeros(action_dim), rng=rng1
    )
    actor_params = actor_model.init(rng2, jnp.zeros(observation_dim), rng=rng2)
    actor_params_target = actor_model.init(rng2, jnp.zeros(observation_dim), rng=rng2)

    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(config.critic_lr),
    )
    actor_train_state:TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )
    return TD3BCUpdateState(
        critic=critic_train_state,
        actor=actor_train_state,
        critic_params_target=critic_params_target,
        actor_params_target=actor_params_target,
        update_idx=0,
    )


class ReplayBuffer(NamedTuple):
    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


def initilize_buffer(dataset: dict, rng: jax.random.PRNGKey) -> Tuple[ReplayBuffer, np.ndarray, np.ndarray]:
    rng, subkey = jax.random.split(rng)
    buffer = ReplayBuffer(
        states=jnp.asarray(dataset["observations"]),
        actions=jnp.asarray(dataset["actions"]),
        next_states=jnp.asarray(dataset["next_observations"]),
        rewards=jnp.asarray(dataset["rewards"]),
        dones=jnp.asarray(dataset["terminals"]),
    )
    # shuffle buffer and select the first buffer_size samples
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(buffer.states))
    buffer = jax.tree_map(lambda x: x[perm], buffer)
    assert len(buffer.states) >= config.buffer_size
    buffer = jax.tree_map(lambda x: x[:config.buffer_size], buffer)
    # normalize states and next_states
    obs_mean = jnp.mean(buffer.states, axis=0)
    obs_std = jnp.std(buffer.states, axis=0)
    buffer = buffer._replace(
        states=(buffer.states - obs_mean) / obs_std,
        next_states=(buffer.next_states - obs_mean) / obs_std,
    )
    return buffer, obs_mean, obs_std

    
def update_actor(
    update_state: TD3BCUpdateState, batch: ReplayBuffer, rng: jax.random.PRNGKey
) -> TD3BCUpdateState:
    """
    Update actor using the following loss:
    L = - Q(s, a) * lambda + BC(s, a)
    """
    actor, critic = update_state.actor, update_state.critic

    def get_actor_loss(
        actor_params: flax.core.frozen_dict.FrozenDict,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        predicted_action = actor.apply_fn(actor_params, batch.states, rng=None)
        critic_params = jax.lax.stop_gradient(update_state.critic.params)
        q_value, _ = critic.apply_fn(
            critic_params, batch.states, predicted_action, rng=None
        )  # todo this will also affect the critic update :/

        mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
        loss_lambda = config.alpha / mean_abs_q

        bc_loss = jnp.square(predicted_action - batch.actions).mean()
        loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
        return loss_actor

    actor_grad_fn = jax.value_and_grad(get_actor_loss)
    actor_loss, actor_grads = actor_grad_fn(actor.params)
    actor = actor.apply_gradients(grads=actor_grads)
    return update_state._replace(actor=actor)


def update_critic(
    update_state: TD3BCUpdateState, batch: ReplayBuffer, max_action, rng: jax.random.PRNGKey
) -> TD3BCUpdateState:
    """
    Update critic using the following loss:
    L = (Q(s, a) - (r + gamma * min(Q_1'(s', a'), Q_2(s', a'))))^2
    """
    actor, critic = update_state.actor, update_state.critic

    def critic_loss(
        critic_params: flax.core.frozen_dict.FrozenDict,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        q_pred_1, q_pred_2 = critic.apply_fn(critic_params, batch.states, batch.actions, rng=None)

        target_next_action = actor.apply_fn(
            update_state.actor_params_target, batch.next_states, rng=None
        )
        policy_noise = (
            config.policy_noise_std
            * max_action
            * jax.random.normal(rng, batch.actions.shape)
        )
        target_next_action = target_next_action + policy_noise.clip(
            -config.policy_noise_clip, config.policy_noise_clip
        )
        target_next_action = target_next_action.clip(-max_action, max_action)
        q_next_1, q_next_2 = critic.apply_fn(
            update_state.critic_params_target,
            batch.next_states,
            target_next_action,
            rng=None,
        )  # twin Q networks
        target = batch.rewards[..., None] + config.gamma * jnp.minimum(q_next_1, q_next_2) * (
            1 - batch.dones[..., None]
        )
        target = jax.lax.stop_gradient(target)
        value_loss_1 = jnp.square(q_pred_1 - target)
        value_loss_2 = jnp.square(q_pred_2 - target)
        value_loss = (value_loss_1 + value_loss_2).mean()
        return value_loss

    critic_grad_fn = jax.value_and_grad(critic_loss)
    critic_loss, critic_grads = critic_grad_fn(critic.params)
    critic = critic.apply_gradients(grads=critic_grads)
    return update_state._replace(critic=critic)


def make_update_steps_fn(
    batches: ReplayBuffer,
    max_action: float,
) -> Callable:
    def update_steps_fn(
        update_state: TD3BCUpdateState,
        rng: jax.random.PRNGKey,
    ):
        def update_step_fn(
            update_state: TD3BCUpdateState,
            rng: jax.random.PRNGKey,
        ):  
            rng, batch_rng, critic_rng, actor_rng = jax.random.split(rng, 4)
            # sample batch
            batch_idx = jax.random.randint(batch_rng, (config.batch_size,), 0, len(batches.states))
            batch = jax.tree_map(lambda x: x[batch_idx], batches)
            # update critic
            update_state = update_critic(update_state, batch, max_action, critic_rng)
            # update actor if policy_freq is met
            new_update_state = update_actor(update_state, batch, actor_rng)
            update_state = jax.lax.cond(
                update_state.update_idx % config.policy_freq == 0,
                lambda: new_update_state,
                lambda: update_state,
            )
            # update target parameters
            critic_params_target = jax.tree_map(
                lambda target, live: config.polyak * target
                + (1.0 - config.polyak) * live,
                update_state.critic_params_target,
                update_state.critic.params,
            )
            actor_params_target = jax.tree_map(
                lambda target, live: config.polyak * target
                + (1.0 - config.polyak) * live,
                update_state.actor_params_target,
                update_state.actor.params,
            )
            return update_state._replace(
                critic_params_target=critic_params_target,
                actor_params_target=actor_params_target,
                update_idx=update_state.update_idx + 1,
            ), None
        rngs = jax.random.split(rng, config.updates_per_epoch)
        update_state, _ = jax.lax.scan(update_step_fn, update_state, rngs)
        return update_state
    return update_steps_fn


@partial(jax.jit)
def get_action(
    actor: TrainState,
    obs: jnp.ndarray,
    low: float,
    high: float,
) -> jnp.ndarray:
    action = actor.apply_fn(actor.params, obs, rng=None)
    action = action.clip(low, high)
    return action


def eval_d4rl(
    subkey: jax.random.PRNGKey,
    actor: TrainState,
    env: gym.Env,
    obs_mean,
    obs_std,
) -> float:
    episode_rews = []
    for _ in range(config.num_test_rollouts):
        obs = env.reset()
        done = False
        episode_rew = 0.0
        while not done:
            obs = jnp.array((obs - obs_mean) / obs_std)
            action = get_action(actor=actor, obs=obs, low=env.action_space.low, high=env.action_space.high)
            action = action.reshape(-1)
            obs, rew, done, info = env.step(action)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return env.get_normalized_score(np.mean(episode_rews)) * 100  # average normalized score


if __name__ == "__main__":
    # setup environemnt, inthis case, D4RL. Please change to your own environment
    env = gym.make(f"{config.env_name}-{config.data_quality}-v2")
    dataset = d4rl.qlearning_dataset(env)
    observation_dim = dataset["observations"].shape[-1]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high

    rng = jax.random.PRNGKey(config.seed)
    rng, buffer_rng, model_rng = jax.random.split(rng, 3)
    # initialize buffer and update state
    buffer, obs_mean, obs_std = initilize_buffer(dataset, buffer_rng)
    update_state = initialize_update_state(observation_dim, action_dim, max_action, model_rng)
    # initialize update steps function
    update_steps_fn = make_update_steps_fn(buffer, max_action)
    jit_update_steps_fn = jax.jit(update_steps_fn)
    
    wandb.init(project="train-TD3-BC", config=config)
    total_steps = 0
    num_total_its = int(config.total_updates) // config.updates_per_epoch
    start_time = time.time()
    for it in tqdm(range(num_total_its)):
        total_steps += 1
        rng, batch_rng, update_rng, eval_rng = jax.random.split(rng, 4)
        # update parameters
        update_state = jit_update_steps_fn(update_state, update_rng) 
        eval_dict = {}
        eval_rew_normed = eval_d4rl(
            eval_rng,
            update_state.actor,
            env,
            obs_mean,
            obs_std,
        )  # evaluate actor
        eval_dict[f"eval_rew_normed_{config.env_name}"] = eval_rew_normed
        eval_dict[f"step"] = total_steps
        print(eval_dict)
        wandb.log(eval_dict)
    print(f"training time: {time.time() - start_time}")
    wandb.finish()
