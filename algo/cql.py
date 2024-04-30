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


class CQLConfig(BaseModel):
    # general config
    env_name: str = "hopper-medium-expert-v2"
    max_steps: int = 1000000
    eval_interval: int = 10000
    updates_per_epoch: int = (
        16  # how many updates per epoch. it is equivalent to how frequent we evaluate the policy
    )
    num_test_rollouts: int = 5
    batch_size: int = 256
    data_size: int = 1000000
    seed: int = 0
    # network config
    hidden_dims: Sequence[int] = (256, 256)  # from jaxcql
    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    # TD3-BC specific
    policy_freq: int = 2  # update actor every policy_freq updates
    tau: float = 0.005  # target network update rate
    alpha: float = 2.5  # BC loss weight
    policy_noise_std: float = 0.2  # std of policy noise
    policy_noise_clip: float = 0.5  # clip policy noise
    gamma: float = 0.99  # discount factor


conf_dict = OmegaConf.from_cli()
config = CQLConfig(**conf_dict)


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: bool = False
    kernel_init: Callable[[Any, Sequence[int], Any], jnp.ndarray] = default_init()
    add_layer_norm: bool = False  # TODO add layer norm
    layer_norm_final: bool = False

    def setup(self):
        self.layers = []
        for i, hidden_dim in enumerate(self.hidden_dims):
            self.layers.append(nn.Dense(hidden_dim, kernel_init=self.kernel_init))
            if self.add_layer_norm:
                if i + 1 < len(self.hidden_dims) or self.layer_norm_final:
                    self.layers.append(nn.LayerNorm())

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x


class DoubleCritic(nn.Module):  # TODO use MLP class ?
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, state, action, rng):
        sa = jnp.concatenate([state, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(sa)

        q_2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(sa)
        return q1, q2


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -10
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
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
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
        else:
            log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash_distribution:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


# from https://github.com/young-geng/JaxCQL
class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param('value', lambda x: self.init_value)

    def __call__(self):
        return self.value


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


def get_dataset(
    dataset: dict, rng: jax.random.PRNGKey
) -> Tuple[Transition, np.ndarray, np.ndarray]:
    """
    This part is D4RL specific. Please change to your own dataset.
    As long as your can convert your dataset in the form of Transition, it should work.
    """
    rng, subkey = jax.random.split(rng)
    data = Transition(
        observations=jnp.asarray(dataset["observations"]),
        actions=jnp.asarray(dataset["actions"]),
        next_observations=jnp.asarray(dataset["next_observations"]),
        rewards=jnp.asarray(dataset["rewards"]),
        dones=jnp.asarray(dataset["terminals"]),
    )
    # shuffle data and select the first data_size samples
    rng, rng_permute, rng_select = jax.random.split(rng, 3)
    perm = jax.random.permutation(rng_permute, len(data.observations))
    data = jax.tree_map(lambda x: x[perm], data)
    assert len(data.observations) >= config.data_size
    data = jax.tree_map(lambda x: x[: config.data_size], data)
    # normalize observations and next_observations
    obs_mean = jnp.mean(data.observations, axis=0)
    obs_std = jnp.std(data.observations, axis=0)
    data = data._replace(
        observations=(data.observations - obs_mean) / obs_std,
        next_observations=(data.next_observations - obs_mean) / obs_std,
    )
    return data, obs_mean, obs_std


def update_by_loss_grad(
    train_state: TrainState, loss_fn: Callable
) -> Tuple[TrainState, jnp.ndarray]:
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(train_state.params)
    new_train_state = train_state.apply_gradients(grads=grad)
    return new_train_state, loss


class CQLTrainer(NamedTuple):
    actor: TrainState
    critic: TrainState
    actor_params_target: flax.core.frozen_dict.FrozenDict
    critic_params_target: flax.core.frozen_dict.FrozenDict
    update_idx: jnp.int32
    max_action: float
    config: flax.core.FrozenDict

    def update_actor(agent, batch: Transition, rng: jax.random.PRNGKey):
        def actor_loss_fn(actor_params):
            predicted_action = agent.actor.apply_fn(
                actor_params, batch.observations, rng=None
            )
            critic_params = jax.lax.stop_gradient(agent.critic.params)
            q_value, _ = agent.critic.apply_fn(
                critic_params, batch.observations, predicted_action, rng=None
            )

            mean_abs_q = jax.lax.stop_gradient(jnp.abs(q_value).mean())
            loss_lambda = agent.config["alpha"] / mean_abs_q

            bc_loss = jnp.square(predicted_action - batch.actions).mean()
            loss_actor = -1.0 * q_value.mean() * loss_lambda + bc_loss
            return loss_actor

        new_actor, _ = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor)

    def update_critic(agent, batch: Transition, rng: jax.random.PRNGKey):
        def critic_loss_fn(critic_params):
            q_pred_1, q_pred_2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions, rng=None
            )
            target_next_action = agent.actor.apply_fn(
                agent.actor_params_target, batch.next_observations, rng=None
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
            q_next_1, q_next_2 = agent.critic.apply_fn(
                agent.critic_params_target,
                batch.next_observations,
                target_next_action,
                rng=None,
            )
            target = batch.rewards[..., None] + agent.config["gamma"] * jnp.minimum(
                q_next_1, q_next_2
            ) * (1 - batch.dones[..., None])
            target = jax.lax.stop_gradient(target)
            value_loss_1 = jnp.square(q_pred_1 - target)
            value_loss_2 = jnp.square(q_pred_2 - target)
            value_loss = (value_loss_1 + value_loss_2).mean()
            return value_loss

        new_critic, _ = update_by_loss_grad(agent.critic, critic_loss_fn)
        return agent._replace(critic=new_critic)

    @partial(
        jax.jit,
        static_argnames=(
            "policy_freq",
            "batch_size",
            "n",
        ),
    )
    def update_n_times(
        agent,
        data: Transition,
        rng: jax.random.PRNGKey,
        policy_freq: int,
        batch_size: int,
        n: int,
    ):  # TODO reduce arguments??
        for _ in range(n):
            rng, batch_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, len(data.observations)
            )
            batch: Transition = jax.tree_map(lambda x: x[batch_idx], data)
            rng, critic_rng, actor_rng = jax.random.split(rng, 3)
            agent = agent.update_critic(batch, critic_rng)
            if _ % policy_freq == 0:
                agent = agent.update_actor(batch, actor_rng)
                critic_params_target = jax.tree_map(
                    lambda target, live: (1 - agent.config["tau"]) * target
                    + agent.config["tau"] * live,
                    agent.critic_params_target,
                    agent.critic.params,
                )
                actor_params_target = jax.tree_map(
                    lambda target, live: (1 - agent.config["tau"]) * target
                    + agent.config["tau"] * live,
                    agent.actor_params_target,
                    agent.actor.params,
                )
                agent = agent._replace(
                    critic_params_target=critic_params_target,
                    actor_params_target=actor_params_target,
                )
        return agent._replace(update_idx=agent.update_idx + 1)

    @jax.jit
    def get_action(
        agent,
        obs: jnp.ndarray,
        low: float,
        high: float,
    ) -> jnp.ndarray:
        action = agent.actor.apply_fn(agent.actor.params, obs, rng=None)
        action = action.clip(low, high)
        return action


def init_params(model_def: nn.Module, inputs: Sequence[jnp.ndarray]):
    return model_def.init(*inputs)


def create_trainer(
    observation_dim, action_dim, max_action, rng, config
) -> CQLTrainer:
    critic_model = DoubleCritic(hidden_dims=config.hidden_dims)
    actor_model = NormalTanhPolicy(
        hidden_dims=config.hidden_dims,
        action_dim=action_dim,
        log_std_min=config.log_std_min,
        log_std_max=config.log_std_max,
        state_dependent_std=True,
        tanh_squash_distribution=True,
    )
    rng, rng1, rng2 = jax.random.split(rng, 3)
    # initialize critic and actor parameters
    critic_params = init_params(
        critic_model, [rng1, jnp.zeros(observation_dim), jnp.zeros(action_dim), rng1]
    )
    critic_params_target = init_params(
        critic_model, [rng1, jnp.zeros(observation_dim), jnp.zeros(action_dim), rng1]
    )

    actor_params = init_params(actor_model, [rng2, jnp.zeros(observation_dim), rng2])
    actor_params_target = init_params(
        actor_model, [rng2, jnp.zeros(observation_dim), rng2]
    )

    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(config.critic_lr),
    )
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )

    config = flax.core.FrozenDict(
        dict(
            alpha=config.alpha,
            policy_noise_std=config.policy_noise_std,
            policy_noise_clip=config.policy_noise_clip,
            gamma=config.gamma,
            tau=config.tau,
            batch_size=config.batch_size,
            policy_freq=config.policy_freq,
        )
    )
    return CQLTrainer(
        actor=actor_train_state,
        critic=critic_train_state,
        actor_params_target=actor_params_target,
        critic_params_target=critic_params_target,
        update_idx=0,
        max_action=max_action,
        config=config,
    )


def evaluate(
    subkey: jax.random.PRNGKey,
    agent: CQLTrainer,
    env: gym.Env,
    obs_mean,
    obs_std,
) -> float:  # D4RL specific
    episode_rews = []
    for _ in range(config.num_test_rollouts):
        obs = env.reset()
        done = False
        episode_rew = 0.0
        while not done:
            obs = jnp.array((obs - obs_mean) / obs_std)
            action = agent.get_action(
                obs=obs,
                low=env.action_space.low,
                high=env.action_space.high,
            )
            action = action.reshape(-1)
            obs, rew, done, info = env.step(action)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return (
        env.get_normalized_score(np.mean(episode_rews)) * 100
    )  # average normalized score


if __name__ == "__main__":
    # setup environemnt, inthis case, D4RL. Please change to your own environment
    env = gym.make(config.env_name)
    dataset = d4rl.qlearning_dataset(env)
    observation_dim = dataset["observations"].shape[-1]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high

    rng = jax.random.PRNGKey(config.seed)
    rng, data_rng, model_rng = jax.random.split(rng, 3)
    # initialize data and update state
    data, obs_mean, obs_std = get_dataset(dataset, data_rng)
    agent = create_trainer(observation_dim, action_dim, max_action, model_rng, config)

    wandb.init(project="train-TD3-BC", config=config)
    epochs = int(
        config.max_steps // config.updates_per_epoch
    )  # we update multiple times per epoch
    steps = 0
    for _ in tqdm(range(epochs)):
        steps += 1
        rng, update_rng, eval_rng = jax.random.split(rng, 3)
        # update parameters
        agent = agent.update_n_times(
            data,
            update_rng,
            config.policy_freq,
            config.batch_size,
            config.updates_per_epoch,
        )
        if steps % config.eval_interval == 0:
            eval_dict = {}
            normalized_score = evaluate(
                eval_rng,
                agent,
                env,
                obs_mean,
                obs_std,
            )  # evaluate actor
            eval_dict[f"normalized_score_{config.env_name}"] = normalized_score
            eval_dict[f"step"] = steps
            print(eval_dict)
            wandb.log(eval_dict)
    wandb.finish()
