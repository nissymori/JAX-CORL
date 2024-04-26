from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, NamedTuple
import numpy as np
import jax.numpy as jnp
import jax
import flax
from functools import partial
import wandb
import flax.linen as nn
import optax
import functools
import gym
from collections import defaultdict
import time
import distrax
import d4rl
import os
from absl import app, flags
from functools import partial
import tqdm
from ml_collections import config_flags
import pickle

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
PRNGKey = Any
Shape = Sequence[int]
Dtype = Any  # this could be a real type?
InfoDict = Dict[str, float]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
ModuleMethod = Union[
    str, Callable, None
]  # A method to be passed into TrainState.__call__


nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def target_update(
    model: "TrainState", target_model: "TrainState", tau: float
) -> "TrainState":
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


class Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    masks: jnp.ndarray
    dones_float: jnp.ndarray
    next_observations: jnp.ndarray


def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
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


def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]

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
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

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


class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

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


def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


from flax.training.train_state import TrainState


class IQLAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch: Transition) -> InfoDict:
        def critic_loss_fn(critic_params):
            next_v = agent.value.apply_fn(agent.value.params, batch.next_observations)
            target_q = batch.rewards + agent.config["discount"] * batch.masks * next_v
            q1, q2 = agent.critic.apply_fn(
                critic_params, batch.observations, batch.actions
            )
            critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
            return critic_loss

        def value_loss_fn(value_params):
            q1, q2 = agent.target_critic.apply_fn(
                agent.target_critic.params, batch.observations, batch.actions
            )
            q = jnp.minimum(q1, q2)
            v = agent.value.apply_fn(value_params, batch.observations)
            value_loss = expectile_loss(q - v, agent.config["expectile"]).mean()
            return value_loss

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

        critic_grad_fn = jax.value_and_grad(critic_loss_fn)
        critic_loss, critic_grad = critic_grad_fn(agent.critic.params)
        new_critic = agent.critic.apply_gradients(grads=critic_grad)

        new_target_critic = target_update(
            agent.critic, agent.target_critic, agent.config["target_update_rate"]
        )

        value_grad_fn = jax.value_and_grad(value_loss_fn)
        value_loss, value_grad = value_grad_fn(agent.value.params)
        new_value = agent.value.apply_gradients(grads=value_grad)

        actor_grad_fn = jax.value_and_grad(actor_loss_fn)
        actor_loss, actor_grad = actor_grad_fn(agent.actor.params)
        new_actor = agent.actor.apply_gradients(grads=actor_grad)

        return (
            agent.replace(
                critic=new_critic,
                target_critic=new_target_critic,
                value=new_value,
                actor=new_actor,
            ),
            {},
        )

    @jax.jit
    def sample_actions(
        agent, observations: np.ndarray, *, seed: PRNGKey, temperature: float = 1.0
    ) -> jnp.ndarray:
        actions = agent.actor.apply_fn(
            agent.actor.params, observations, temperature=temperature
        ).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnums=(3, 4))
    def sample_batch_and_update_n_times(
        agent,
        dataset: Transition,
        rng: PRNGKey,
        batch_size: int,
        n_updates: int,
    ):
        for _ in range(n_updates):
            rng, subkey = jax.random.split(rng)
            batch_indices = jax.random.randint(
                subkey, (batch_size,), 0, len(dataset.observations)
            )
            batch = jax.tree_map(lambda x: x[batch_indices], dataset)
            agent, info = agent.update(batch)
        return agent, info


def create_learner(
    seed: int,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    actor_lr: float = 3e-4,
    value_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    hidden_dims: Sequence[int] = (256, 256),
    discount: float = 0.99,
    tau: float = 0.005,
    expectile: float = 0.8,
    temperature: float = 0.1,
    dropout_rate: Optional[float] = None,
    max_steps: Optional[int] = None,
    opt_decay_schedule: str = "cosine",
    **kwargs,
):

    print("Extra kwargs:", kwargs)

    rng = jax.random.PRNGKey(seed)
    rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

    action_dim = actions.shape[-1]
    actor_def = Policy(
        hidden_dims,
        action_dim=action_dim,
        log_std_min=-5.0,
        state_dependent_std=False,
        tanh_squash_distribution=False,
    )

    if opt_decay_schedule == "cosine":
        schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
        actor_tx = optax.chain(
            optax.scale_by_adam(), optax.scale_by_schedule(schedule_fn)
        )
    else:
        actor_tx = optax.adam(learning_rate=actor_lr)

    actor_params = actor_def.init(actor_key, observations)
    actor = TrainState.create(
        apply_fn=actor_def.apply, params=actor_params, tx=actor_tx
    )

    critic_def = ensemblize(Critic, num_qs=2)(hidden_dims)
    critic_params = critic_def.init(critic_key, observations, actions)
    critic = TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=critic_lr),
    )
    target_critic = TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=optax.adam(learning_rate=critic_lr),
    )

    value_def = ValueCritic(hidden_dims)
    value_params = value_def.init(value_key, observations)
    value = TrainState.create(
        apply_fn=value_def.apply,
        params=value_params,
        tx=optax.adam(learning_rate=value_lr),
    )

    config = flax.core.FrozenDict(
        dict(
            discount=discount,
            temperature=temperature,
            expectile=expectile,
            target_update_rate=tau,
        )
    )

    return IQLAgent(
        rng,
        critic=critic,
        target_critic=target_critic,
        value=value,
        actor=actor,
        config=config,
    )


def get_default_config():
    import ml_collections

    config = ml_collections.ConfigDict(
        {
            "actor_lr": 3e-4,
            "value_lr": 3e-4,
            "critic_lr": 3e-4,
            "hidden_dims": (256, 256),
            "discount": 0.99,
            "expectile": 0.7,
            "temperature": 3.0,
            "dropout_rate": ml_collections.config_dict.placeholder(float),
            "tau": 0.005,
        }
    )
    return config


def make_env(env_name: str):
    env = gym.make(env_name)
    return env


def get_dataset(
    env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5
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
    return dataset


FLAGS = flags.FLAGS
flags.DEFINE_string("env_name", "halfcheetah-medium-expert-v2", "Environment name.")
flags.DEFINE_string("save_dir", None, "Logging dir (if not None, save params).")
flags.DEFINE_integer("seed", np.random.choice(1000000), "Random seed.")
flags.DEFINE_integer("eval_episodes", 10, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 100000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("save_interval", 25000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("n_updates", 8, "Number of updates per step.")

config_flags.DEFINE_config_dict("config", get_default_config(), lock_config=False)


def get_normalization(dataset: Transition):
    # into_numpy
    dataset = jax.tree_map(lambda x: np.array(x), dataset)
    returns = []
    ret = 0
    for r, term in zip(dataset.rewards, dataset.dones_float):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000


def main(_):
    wandb.init(config=FLAGS.config, project="iql")
    rng = jax.random.PRNGKey(FLAGS.seed)
    if FLAGS.save_dir is not None:
        pass
        # TODO save config

    env = make_env(FLAGS.env_name)
    dataset: Transition = get_dataset(env)

    normalizing_factor = get_normalization(dataset)
    dataset = dataset._replace(rewards=dataset.rewards / normalizing_factor)

    example_batch = jax.tree_map(lambda x: x[0], dataset)
    agent = create_learner(
        FLAGS.seed,
        example_batch.observations,
        example_batch.actions,
        max_steps=FLAGS.max_steps,
        **FLAGS.config,
    )
    # into jnp.ndarray
    dataset = jax.tree_map(lambda x: jnp.asarray(x), dataset)
    num_steps = FLAGS.max_steps // FLAGS.n_updates
    for i in tqdm.tqdm(range(1, num_steps + 1), smoothing=0.1, dynamic_ncols=True):
        rng, subkey = jax.random.split(rng)
        agent, update_info = agent.sample_batch_and_update_n_times(
            dataset, subkey, FLAGS.batch_size, FLAGS.n_updates
        )

        if i % FLAGS.log_interval == 0:
            train_metrics = {f"training/{k}": v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(
                agent.sample_actions, temperature=0.0, seed=jax.random.PRNGKey(0)
            )
            normalized_score = evaluate(
                policy_fn, env, num_episodes=FLAGS.eval_episodes
            )
            print(i, normalized_score)
            eval_metrics = {"normalized_score": normalized_score}
            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            pass
            # TODO save agent


if __name__ == "__main__":
    app.run(main)
