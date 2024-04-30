import time
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple

import d4rl
import flax
import flax.linen as nn
from flax.training.train_state import TrainState
import distrax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm


class CQLConfig(BaseModel):
    # general config
    env_name: str = "hopper-medium-expert-v2"
    max_steps: int = 1000000
    eval_interval: int = 10000
    updates_per_epoch: int = 4  # how many updates per epoch.

    num_test_rollouts: int = 5
    batch_size: int = 256
    data_size: int = 1000000
    seed: int = 0
    # network config
    hidden_dims: Sequence[int] = (256, 256)  # from jaxcql
    critic_lr: float = 3e-4
    actor_lr: float = 3e-4
    # CQL config
    discount: float = 0.99
    alpha_multiplier: float = 1.0
    use_automatic_entropy_tuning: bool = True
    backup_entropy: bool = False
    target_entropy: float = 0.0
    tau: float = 0.005
    cql_n_actions: int = 10
    cql_clip_diff_min: float = -np.inf
    cql_clip_diff_max: float = np.inf
    cql_temperature: float = 1.0
    alpha_lagrangian: bool = True
    cql_target_budget: Optional[float] = None
    cql_alpha: float = 1.0


conf_dict = OmegaConf.from_cli()
config = CQLConfig(**conf_dict)


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


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


class DoubleCritic(nn.Module):  # TODO use MLP class ?
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, state, action):
        sa = jnp.concatenate([state, action], axis=-1)
        q1 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(sa)
        q2 = MLP((*self.hidden_dims, 1), add_layer_norm=True)(sa)
        return q1, q2


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -10.0
    log_std_max: Optional[float] = 2.0
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


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())


# from https://github.com/young-geng/JaxCQL
class Scalar(nn.Module):
    init_value: float

    def setup(self):
        self.value = self.param("value", lambda x: self.init_value)

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


class CQLTrainer(NamedTuple):
    actor: TrainState
    critic: TrainState
    target_critic: TrainState
    log_entropy_alpha: TrainState
    log_conservative_alpha: TrainState
    max_action: float
    action_dim: int
    config: flax.core.FrozenDict

    def update_alpha(agent, batch: Transition, rng: jax.random.PRNGKey):
        dist = agent.actor.apply_fn(agent.actor.params, batch.observations)
        actions = dist.sample(seed=rng)

        def get_alpha_loss(alpha_params):
            log_alpha = agent.log_entropy_alpha.apply_fn(alpha_params)
            alpha_loss = log_alpha * (
                -dist.log_prob(actions).mean() - agent.config["target_entropy"]
            )
            return alpha_loss

        new_alpha, alpha_loss = update_by_loss_grad(
            agent.log_entropy_alpha, get_alpha_loss
        )
        return agent._replace(log_entropy_alpha=new_alpha), alpha_loss

    def update_actor(agent, batch: Transition, rng: jax.random.PRNGKey):
        alpha = jnp.exp(
            agent.log_entropy_alpha.apply_fn(agent.log_entropy_alpha.params)
        )

        def actor_loss_fn(actor_params):
            dist = agent.actor.apply_fn(actor_params, batch.observations)
            actions = dist.sample(seed=rng)
            log_probs = dist.log_prob(actions)

            q1, q2 = agent.critic.apply_fn(
                agent.critic.params, batch.observations, actions
            )
            q = jnp.minimum(q1, q2)
            actor_loss = (alpha * log_probs - q).mean()
            return actor_loss

        new_actor, actor_loss = update_by_loss_grad(agent.actor, actor_loss_fn)
        return agent._replace(actor=new_actor), actor_loss

    def update_critic(
        agent,
        batch: Transition,
        _rng: jax.random.PRNGKey,
        batch_size: int,
        action_dim: int,
        cql_n_actions: int,
    ):
        def get_cql_critic_loss(critic_params):
            q1, q2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )

            ########## Normal double critic TD loss ##########
            rng, actor_rng = jax.random.split(_rng)
            next_dist = agent.actor.apply_fn(
                agent.actor.params, batch.next_observations
            )
            next_actions = next_dist.sample(seed=actor_rng)
            n_q1, n_q2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.next_observations, next_actions
            )
            n_q = jnp.minimum(n_q1, n_q2)
            target_q = (
                batch.rewards + agent.config["discount"] * (1 - batch.dones) * n_q
            )

            q_mse1 = jnp.mean((q1 - target_q) ** 2)
            q_mse2 = jnp.mean((q2 - target_q) ** 2)

            ######### CQL loss #########
            # radom actions
            rng, rand_rng, current_rng, next_rng = jax.random.split(rng, 4)
            rand_actions = jax.random.uniform(
                rand_rng,
                shape=(
                    batch_size,
                    cql_n_actions,
                    action_dim,
                ),
                minval=-1.0,
                maxval=1.0,
            )  # (batch, n_a, a)
            # current actions
            current_rngs = jax.random.split(current_rng, cql_n_actions)
            current_dist = agent.actor.apply_fn(agent.actor.params, batch.observations)
            current_actions, current_log_probs = jax.vmap(
                current_dist.sample_and_log_prob
            )(
                seed=current_rngs
            )  # (n_a, batch, a)
            current_actions = jnp.swapaxes(current_actions, 0, 1)  # (batch, n_a, a)
            current_log_probs = jnp.swapaxes(current_log_probs, 0, 1)  # (batch, n_a)
            # next actions
            next_rngs = jax.random.split(next_rng, cql_n_actions)
            next_dist = agent.actor.apply_fn(
                agent.actor.params, batch.next_observations
            )
            next_actions, next_log_probs = jax.vmap(next_dist.sample_and_log_prob)(
                seed=next_rngs
            )  # (n_a, batch, a)
            next_actions = jnp.swapaxes(next_actions, 0, 1)  # (batch, n_a, a)
            next_log_probs = jnp.swapaxes(next_log_probs, 0, 1)  # (batch, n_a)

            # Q values
            repeated_obs = jnp.tile(
                batch.observations[:, None, :], (1, cql_n_actions, 1)
            )  # (batch, n_a, o)
            cql_q1_rand, cql_q2_rand = agent.critic.apply_fn(
                critic_params, repeated_obs, rand_actions
            )  # (batch, n_a, 1)
            cql_q1_current, cql_q2_current = agent.critic.apply_fn(
                critic_params, repeated_obs, current_actions
            )  # (batch, n_a, 1)
            cql_q1_next, cql_q2_next = agent.critic.apply_fn(
                critic_params, repeated_obs, next_actions
            )  # (batch, n_a, 1)

            with jax.ensure_compile_time_eval():
                random_density = jnp.log(0.5**action_dim)
            # concatenate q values
            cql_cat_q1 = jnp.concatenate(
                [
                    cql_q1_rand[:, :, 0] - random_density,
                    cql_q1_current[:, :, 0] - current_log_probs,
                    cql_q1_next[:, :, 0] - next_log_probs,
                ],
                axis=1,
            )  # (batch, 3 * n_a)
            cql_cat_q2 = jnp.concatenate(
                [
                    cql_q2_rand[:, :, 0] - random_density,
                    cql_q2_current[:, :, 0] - current_log_probs,
                    cql_q2_next[:, :, 0] - next_log_probs,
                ],
                axis=1,
            )  # (batch, 3 * n_a)

            # logsumexp cql
            cql_ood_q1 = (
                jax.scipy.special.logsumexp(
                    cql_cat_q1 / agent.config["cql_temperature"], axis=1
                )
                * agent.config["cql_temperature"]
            )  # (b, )
            cql_ood_q2 = (
                jax.scipy.special.logsumexp(
                    cql_cat_q2 / agent.config["cql_temperature"], axis=1
                )
                * agent.config["cql_temperature"]
            )  # (b, )

            # cql loss
            cql_diff1 = jnp.clip(
                cql_ood_q1 - q1,
                agent.config["cql_clip_diff_min"],
                agent.config["cql_clip_diff_max"],
            )
            cql_diff2 = jnp.clip(
                cql_ood_q2 - q2,
                agent.config["cql_clip_diff_min"],
                agent.config["cql_clip_diff_max"],
            )
            cql_alpha = jnp.exp(
                agent.log_conservative_alpha.apply_fn(
                    agent.log_conservative_alpha.params
                )
            )
            q1_cql_loss = jnp.mean(cql_diff1) * cql_alpha
            q2_cql_loss = jnp.mean(cql_diff2) * cql_alpha
            # mse loss + cql loss
            total_loss = q_mse1 + q_mse2 + q1_cql_loss + q2_cql_loss
            return total_loss

        new_critic, critic_loss = update_by_loss_grad(agent.critic, get_cql_critic_loss)
        return agent._replace(critic=new_critic), critic_loss

    @partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
    def update_n_times(
        agent,
        data: Transition,
        rng: jax.random.PRNGKey,
        alpha_lagrangian: bool,
        batch_size: int,
        action_dim: int,
        cql_n_actions: int,
        n: int,
    ):  # TODO reduce arguments??
        for _ in range(n):
            rng, batch_rng = jax.random.split(rng, 2)
            batch_idx = jax.random.randint(
                batch_rng, (batch_size,), 0, len(data.observations)
            )
            batch: Transition = jax.tree_map(lambda x: x[batch_idx], data)
            rng, alpha_rng, critic_rng, actor_rng = jax.random.split(rng, 4)

            agent, critic_loss = agent.update_critic(
                batch, critic_rng, batch_size, action_dim, cql_n_actions
            )
            agent, actor_loss = agent.update_actor(batch, actor_rng)
            if alpha_lagrangian:
                agent, alpha_loss = agent.update_alpha(batch, alpha_rng)
            else:
                alpha_loss = 0.0
            new_target_critic = target_update(
                agent.critic, agent.target_critic, agent.config["tau"]
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
        actions = jnp.clip(actions, -1.0, 1.0)
        return actions


def create_trainer(observation_dim, action_dim, max_action, rng, config) -> CQLTrainer:
    critic_model = DoubleCritic(hidden_dims=config.hidden_dims)
    actor_model = NormalTanhPolicy(
        hidden_dims=config.hidden_dims,
        action_dim=action_dim,
        state_dependent_std=True,
        tanh_squash_distribution=True,
    )
    rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
    # initialize critic
    critic_params = critic_model.init(
        rng1, jnp.zeros(observation_dim), jnp.zeros(action_dim)
    )
    critic_train_state: TrainState = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(config.critic_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=config.critic_lr),
    )

    # initialize actor
    actor_params = actor_model.init(rng2, jnp.zeros(observation_dim))
    actor_train_state: TrainState = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=optax.adam(config.actor_lr),
    )

    log_entropy_alpha = Scalar(
        np.log(0.1) if config.alpha_lagrangian is not None else np.log(config.sac_alpha)
    )
    log_entropy_alpha_train_state = TrainState.create(
        apply_fn=log_entropy_alpha.apply,
        params=log_entropy_alpha.init(rng3),
        tx=optax.adam(config.actor_lr),
    )
    log_conservative_alpha = Scalar(
        np.log(1.0)
        if config.cql_target_budget is not None
        else np.log(config.cql_alpha)
    )
    log_conservative_alpha_train_state = TrainState.create(
        apply_fn=log_conservative_alpha.apply,
        params=log_conservative_alpha.init(rng3),
        tx=optax.adam(config.critic_lr),
    )

    cql_config = flax.core.FrozenDict(
        dict(
            cql_clip_diff_max=config.cql_clip_diff_max,
            cql_clip_diff_min=config.cql_clip_diff_min,
            cql_temperature=config.cql_temperature,
            tau=config.tau,
            discount=config.discount,
            target_entropy=config.target_entropy,
        )
    )
    return CQLTrainer(
        actor=actor_train_state,
        critic=critic_train_state,
        target_critic=target_critic,
        log_entropy_alpha=log_entropy_alpha_train_state,
        log_conservative_alpha=log_conservative_alpha_train_state,
        max_action=max_action,
        action_dim=action_dim,
        config=cql_config,
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
            action = agent.sample_actions(
                obs, seed=jax.random.PRNGKey(0), temperature=0.0
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

    wandb.init(project="train-CQL", config=config)
    epochs = int(
        config.max_steps // config.updates_per_epoch
    )  # we update multiple times per epoch
    steps = 0
    for _ in tqdm(range(epochs)):
        steps += 1
        rng, update_rng, eval_rng = jax.random.split(rng, 3)
        # update parameters
        agent, _ = agent.update_n_times(
            data,
            update_rng,
            config.alpha_lagrangian,
            config.batch_size,
            action_dim,
            config.cql_n_actions,
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
