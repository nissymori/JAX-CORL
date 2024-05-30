import collections
from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import d4rl
import flax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm


class DTConfig(BaseModel):
    # GENERAL
    project: str = "decision-transformer"
    seed: int = 0
    env_name: str = "walker2d-medium-v2"
    batch_size: int = 64
    num_eval_episodes: int = 5
    max_eval_ep_len: int = 1000
    max_steps: int = 20000
    eval_interval: int = 2000
    # NETWORK
    context_len: int = 20
    n_blocks: int = 3
    embed_dim: int = 128
    n_heads: int = 1
    dropout_p: float = 0.1
    lr: float = 1e-4
    wt_decay: float = 1e-4
    warmup_steps: int = 10000
    # DT SPECIFIC
    rtg_scale: int = 1000
    rtg_target: int = None


conf_dict = OmegaConf.from_cli()
config: DTConfig = DTConfig(**conf_dict)

# RTG target is specific to each environment
if "halfcheetah" in config.env_name:
    rtg_target = 12000
elif "hopper" in config.env_name:
    rtg_target = 3600
elif "walker" in config.env_name:
    rtg_target = 5000
else:
    raise ValueError("We only care about Mujoco envs for now.")
config.rtg_target = rtg_target


class MaskedCausalAttention(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=True) -> jnp.ndarray:
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        # rearrange q, k, v as (B, N, T, D)
        q = nn.Dense(self.h_dim)(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        k = nn.Dense(self.h_dim)(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)
        v = nn.Dense(self.h_dim)(x).reshape(B, T, N, D).transpose(0, 2, 1, 3)

        # causal mask
        ones = jnp.ones((self.max_T, self.max_T))
        mask = jnp.tril(ones).reshape(1, 1, self.max_T, self.max_T)

        # weights (B, N, T, T) jax
        weights = jnp.einsum("bntd,bnfd->bntf", q, k) / jnp.sqrt(D)
        # causal mask applied to weights
        weights = jnp.where(mask[..., :T, :T] == 0, -jnp.inf, weights[..., :T, :T])
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = jax.nn.softmax(weights, axis=-1)

        # attention (B, N, T, D)
        attention = nn.Dropout(self.drop_p, deterministic=not training)(
            jnp.einsum("bntf,bnfd->bntd", normalized_weights, v)
        )
        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(0, 2, 1, 3).reshape(B, T, N * D)
        out = nn.Dropout(self.drop_p, deterministic=not training)(
            nn.Dense(self.h_dim)(attention)
        )
        return out


class Block(nn.Module):
    h_dim: int
    max_T: int
    n_heads: int
    drop_p: float

    @nn.compact
    def __call__(self, x: jnp.ndarray, training=True) -> jnp.ndarray:
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + MaskedCausalAttention(
            self.h_dim, self.max_T, self.n_heads, self.drop_p
        )(
            x, training=training
        )  # residual
        x = nn.LayerNorm()(x)
        # MLP
        out = nn.Dense(4 * self.h_dim)(x)
        out = nn.gelu(out)
        out = nn.Dense(self.h_dim)(out)
        out = nn.Dropout(self.drop_p, deterministic=not training)(out)
        # residual
        x = x + out
        x = nn.LayerNorm()(x)
        return x


class DecisionTransformer(nn.Module):
    state_dim: int
    act_dim: int
    n_blocks: int
    h_dim: int
    context_len: int
    n_heads: int
    drop_p: float
    max_timestep: int = 4096

    def setup(self) -> None:
        self.blocks = [
            Block(self.h_dim, 3 * self.context_len, self.n_heads, self.drop_p)
            for _ in range(self.n_blocks)
        ]

        # projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm()
        self.embed_timestep = nn.Embed(self.max_timestep, self.h_dim)
        self.embed_rtg = nn.Dense(self.h_dim)
        self.embed_state = nn.Dense(self.h_dim)

        # continuous actions
        self.embed_action = nn.Dense(self.h_dim)
        self.use_action_tanh = True

        # prediction heads
        self.predict_rtg = nn.Dense(1)
        self.predict_state = nn.Dense(self.state_dim)
        self.predict_action = nn.Dense(self.act_dim)

    def __call__(
        self,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
        training=True,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        B, T, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_rtg(returns_to_go) + time_embeddings

        # stack rtg, states and actions and reshape sequence as
        # (r1, s1, a1, r2, s2, a2 ...)
        h = (
            jnp.stack((returns_embeddings, state_embeddings, action_embeddings), axis=1)
            .transpose(0, 2, 1, 3)
            .reshape(B, 3 * T, self.h_dim)
        )

        h = self.embed_ln(h)

        # transformer and prediction
        for block in self.blocks:
            h = block(h, training=training)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).transpose(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_rtg(h[:, 2])  # predict next rtg given r, s, a
        state_preds = self.predict_state(h[:, 2])  # predict next state given r, s, a
        action_preds = self.predict_action(h[:, 1])
        if self.use_action_tanh:
            action_preds = jnp.tanh(action_preds)

        return state_preds, action_preds, return_preds


def discount_cumsum(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    disc_cumsum = np.zeros_like(x)
    disc_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
    return disc_cumsum


def get_traj(env_name):
    name = env_name
    print("processing: ", name)
    env = gym.make(name)
    dataset = env.get_dataset()
    N = dataset["rewards"].shape[0]
    data_ = collections.defaultdict(list)
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset["terminals"][i])
        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == 1000 - 1
        for k in [
            "observations",
            "next_observations",
            "actions",
            "rewards",
            "terminals",
        ]:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1
    returns = np.array([np.sum(p["rewards"]) for p in paths])
    num_samples = np.sum([p["rewards"].shape[0] for p in paths])
    print(f"Number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
    )
    obs_mean = dataset["observations"].mean(axis=0)
    obs_std = dataset["observations"].std(axis=0)
    return paths, obs_mean, obs_std


class Trajectory(NamedTuple):
    timesteps: np.ndarray  # num_ep x max_len
    states: np.ndarray  # num_ep x max_len x state_dim
    actions: np.ndarray  # num_ep x max_len x act_dim
    returns_to_go: np.ndarray  # num_ep x max_len x 1
    masks: np.ndarray  # num_ep x max_len


def padd_by_zero(arr: jnp.ndarray, pad_to: int) -> jnp.ndarray:
    return np.pad(arr, ((0, pad_to - arr.shape[0]), (0, 0)), mode="constant")


def make_padded_trajectories(
    config: DTConfig,
) -> Tuple[Trajectory, int, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    trajectories, mean, std = get_traj(config.env_name)

    # Calculate returns to go for all trajectories
    # Normalize states
    # Calculate max len of traj
    max_len = 0
    traj_lengths = []
    for traj in trajectories:
        traj["returns_to_go"] = discount_cumsum(traj["rewards"], 1.0) / config.rtg_scale
        traj["observations"] = (traj["observations"] - mean) / std
        max_len = max(max_len, traj["observations"].shape[0])
        traj_lengths.append(traj["observations"].shape[0])

    # Pad trajectories
    padded_trajectories = {key: [] for key in Trajectory._fields}
    for traj in trajectories:
        timesteps = np.arange(0, len(traj["observations"]))
        padded_trajectories["timesteps"].append(
            padd_by_zero(timesteps.reshape(-1, 1), max_len).reshape(-1)
        )
        padded_trajectories["states"].append(
            padd_by_zero(traj["observations"], max_len)
        )
        padded_trajectories["actions"].append(padd_by_zero(traj["actions"], max_len))
        padded_trajectories["returns_to_go"].append(
            padd_by_zero(traj["returns_to_go"].reshape(-1, 1), max_len)
        )
        padded_trajectories["masks"].append(
            padd_by_zero(
                np.ones((len(traj["observations"]), 1)).reshape(-1, 1), max_len
            ).reshape(-1)
        )

    return (
        Trajectory(
            timesteps=np.stack(padded_trajectories["timesteps"]),
            states=np.stack(padded_trajectories["states"]),
            actions=np.stack(padded_trajectories["actions"]),
            returns_to_go=np.stack(padded_trajectories["returns_to_go"]),
            masks=np.stack(padded_trajectories["masks"]),
        ),
        len(trajectories),
        jnp.array(traj_lengths),
        mean,
        std,
    )


def sample_start_idx(
    rng: jax.random.PRNGKey,
    traj_idx: int,
    padded_traj_length: jnp.ndarray,
    context_len: int,
) -> jnp.ndarray:
    """
    Determine the start_idx for given trajectory, the trajectories are padded to max_len.
    Therefore, naively sample from 0, max_len will produce bunch of all zero data.
    To avoid that, we refer padded_traj_length, the list of actual trajectry length + context_len
    """
    traj_len = padded_traj_length[traj_idx]
    start_idx = jax.random.randint(rng, (1,), 0, traj_len - context_len - 1)
    return start_idx


def extract_traj(
    traj_idx: jnp.ndarray, start_idx: jnp.ndarray, traj: Trajectory, context_len: int
) -> Trajectory:
    return jax.tree_map(
        lambda x: jax.lax.dynamic_slice_in_dim(x[traj_idx], start_idx, context_len),
        traj,
    )


@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_traj_batch(
    rng,
    traj: Trajectory,
    batch_size: int,
    context_len: int,
    episode_num: int,
    padded_traj_lengths: jnp.ndarray,
) -> Trajectory:
    traj_idx = jax.random.randint(rng, (batch_size,), 0, episode_num)  # B
    start_idx = jax.vmap(sample_start_idx, in_axes=(0, 0, None, None))(
        jax.random.split(rng, batch_size), traj_idx, padded_traj_lengths, context_len
    ).reshape(
        -1
    )  # B
    return jax.vmap(extract_traj, in_axes=(0, 0, None, None))(
        traj_idx, start_idx, traj, context_len
    )


class DTTrainer(NamedTuple):
    train_state: TrainState

    @jax.jit
    def update(
        agent, batch: Trajectory, rng: jax.random.PRNGKey
    ) -> Tuple[Any, jnp.ndarray]:
        timesteps, states, actions, returns_to_go, traj_mask = (
            batch.timesteps,
            batch.states,
            batch.actions,
            batch.returns_to_go,
            batch.masks,
        )

        def loss_fn(params):
            state_preds, action_preds, return_preds = agent.train_state.apply_fn(
                params, timesteps, states, actions, returns_to_go, rngs={"dropout": rng}
            )  # B x T x state_dim, B x T x act_dim, B x T x 1
            # mask actions
            actions_masked = actions * traj_mask[:, :, None]
            action_preds_masked = action_preds * traj_mask[:, :, None]
            # Calculate mean squared error loss
            action_loss = jnp.mean(jnp.square(action_preds_masked - actions_masked))

            return action_loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grad = grad_fn(agent.train_state.params)
        # Apply gradient clipping
        grad = jax.tree_util.tree_map(lambda g: jnp.clip(g, -0.25, 0.25), grad)
        train_state = agent.train_state.apply_gradients(grads=grad)
        return agent._replace(train_state=train_state), loss

    @jax.jit
    def get_action(
        agent,
        timesteps: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns_to_go: jnp.ndarray,
    ) -> jnp.ndarray:
        state_preds, action_preds, return_preds = agent.train_state.apply_fn(
            agent.train_state.params,
            timesteps,
            states,
            actions,
            returns_to_go,
            training=False,
        )
        return action_preds


def create_trainer(state_dim: int, act_dim: int, config: DTConfig) -> DTTrainer:
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_blocks=config.n_blocks,
        h_dim=config.embed_dim,
        context_len=config.context_len,
        n_heads=config.n_heads,
        drop_p=config.dropout_p,
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    # initialize params
    params = model.init(
        init_rng,
        timesteps=jnp.zeros((1, config.context_len), jnp.int32),
        states=jnp.zeros((1, config.context_len, state_dim), jnp.float32),
        actions=jnp.zeros((1, config.context_len, act_dim), jnp.float32),
        returns_to_go=jnp.zeros((1, config.context_len, 1), jnp.float32),
        training=False,
    )
    # optimizer
    scheduler = optax.cosine_decay_schedule(
        init_value=config.lr, decay_steps=config.warmup_steps
    )
    tx = optax.chain(
        optax.scale_by_schedule(scheduler),
        optax.adamw(learning_rate=config.lr, weight_decay=config.wt_decay),
    )

    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return DTTrainer(train_state)


def evaluate(
    agent: DTTrainer,
    env: gym.Env,
    config: DTConfig,
    state_mean=0,
    state_std=1,
) -> float:

    eval_batch_size = 1  # required for forward pass

    results = {}
    total_reward = 0
    total_timesteps = 0

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # same as timesteps used for training the transformer
    timesteps = jnp.arange(0, config.max_eval_ep_len, 1, jnp.int32)
    # repeat
    timesteps = jnp.repeat(timesteps[None, :], eval_batch_size, axis=0)
    for _ in range(config.num_eval_episodes):
        # zeros place holders
        actions = jnp.zeros(
            (eval_batch_size, config.max_eval_ep_len, act_dim), dtype=jnp.float32
        )
        states = jnp.zeros(
            (eval_batch_size, config.max_eval_ep_len, state_dim), dtype=jnp.float32
        )
        rewards_to_go = jnp.zeros(
            (eval_batch_size, config.max_eval_ep_len, 1), dtype=jnp.float32
        )
        # init episode
        running_state = env.reset()
        running_reward = 0
        running_rtg = config.rtg_target / config.rtg_scale
        for t in range(config.max_eval_ep_len):
            total_timesteps += 1
            # add state in placeholder and normalize
            states = states.at[0, t].set(running_state)
            states = states.at[0, t].set((states[0, t] - state_mean) / state_std)
            # calcualate running rtg and add in placeholder
            running_rtg = running_rtg - (running_reward / config.rtg_scale)
            rewards_to_go = rewards_to_go.at[0, t].set(running_rtg)
            if t < config.context_len:
                act_preds = agent.get_action(
                    timesteps[:, : t + 1],
                    states[:, : t + 1],
                    actions[:, : t + 1],
                    rewards_to_go[:, : t + 1],
                )
                act = act_preds[0, t]
            else:
                act_preds = agent.get_action(
                    timesteps[:, t - config.context_len + 1 : t + 1],
                    states[:, t - config.context_len + 1 : t + 1],
                    actions[:, t - config.context_len + 1 : t + 1],
                    rewards_to_go[:, t - config.context_len + 1 : t + 1],
                )
                act = act_preds[0, -1]
            running_state, running_reward, done, _ = env.step(act)
            # add action in placeholder
            actions = actions.at[0, t].set(act)
            total_reward += running_reward
            if done:
                break
    normalized_score = (
        env.get_normalized_score(total_reward / config.num_eval_episodes) * 100
    )
    return normalized_score


if __name__ == "__main__":
    wandb.init(project=config.project, config=config)
    env = gym.make(config.env_name)
    rng = jax.random.PRNGKey(config.seed)
    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    trajectories, episode_num, traj_lengths, state_mean, state_std = (
        make_padded_trajectories(config)
    )
    # create trainer
    agent = create_trainer(state_dim, act_dim, config)
    for i in tqdm(range(config.max_steps)):
        rng, data_rng, update_rng = jax.random.split(rng, 3)
        traj_batch = sample_traj_batch(
            data_rng,
            trajectories,
            config.batch_size,
            config.context_len,
            episode_num,
            traj_lengths,
        )  # B x T x D
        agent, action_loss = agent.update(traj_batch, update_rng)  # update parameters

        if i % config.eval_interval == 0:
            # evaluate on env
            normalized_score = evaluate(agent, env, config, state_mean, state_std)
            print(i, normalized_score)
        wandb.log(
            {
                "action_loss": action_loss,
                f"{config.env_name}/normalized_score": normalized_score,
                "step": i,
            }
        )

    # final evaluation
    normalized_score = evaluate(agent, env, config, state_mean, state_std)
    wandb.log({f"{config.env_name}/final_normalized_score": normalized_score})
    wandb.finish()
