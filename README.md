# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), clean **single-file** implementations of offline RL algorithms with **solid performance reports**.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit` and `vmap`.
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few battle-tested algorithms**: Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms).
- 📈　Solid performance report ([README](https://github.com/nissymori/JAX-CORL?tab=readme-ov-file#reports-for-d4rl-mujoco), [Wiki](https://github.com/nissymori/JAX-CORL/wiki)).

JAX-CORL is complementing the single-file RL ecosystem by offering the combination of offline x JAX. 
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Online x PyTorch
- [purejaxrl](https://github.com/luchris429/purejaxrl): Online x JAX
- [CORL](https://github.com/tinkoff-ai/CORL): Offline x PyTorch
- **JAX-CORL(ours): Offline x JAX**

# Algorithms
|Algorithm|implementation|training time (CORL)|training time (ours)| wandb |
|---|---|---|---|---|
|[AWAC](https://arxiv.org/abs/2006.09359)| [algos/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/awac.py) |4.46h|11m(**24x faster**)|[link](https://api.wandb.ai/links/nissymori/mwi235j6) |
|[IQL](https://arxiv.org/abs/2110.06169)|  [algos/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/iql.py)   |4.08h|9m(**28x faster**)| [link]([https://api.wandb.ai/links/nissymori/hazajm9q](https://api.wandb.ai/links/nissymori/2mtlvha5)) |
|[TD3+BC](https://arxiv.org/pdf/2106.06860)| [algos/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/td3bc.py)  |2.47h|9m(**16x faster**)| [link](https://api.wandb.ai/links/nissymori/h21py327) |
|[CQL](https://arxiv.org/abs/2006.04779)| [algos/cql.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/cql.py)   |11.52h|56m(**12x faster**)|[link](https://api.wandb.ai/links/nissymori/cnxdwkgf)|
|[XQL](https://arxiv.org/abs/2301.02328)| [algos/xql.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/xql.py)   | - | 12m | [link]([algos/cql.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/cql.py)) |
|[DT](https://arxiv.org/abs/2106.01345) | [algos/dt.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/dt.py) |42m|11m(**4x faster**)|[link](https://api.wandb.ai/links/nissymori/yrpja8if)|

Training time is for `1000_000` update steps without evaluation for `halfcheetah-medium-expert v2` (little difference between different [D4RL](https://arxiv.org/abs/2004.07219) mujoco environments). The training time of ours includes the compile time for `jit`. The computations were performed using four [GeForce GTX 1080 Ti GPUs](https://versus.com/en/inno3d-ichill-geforce-gtx-1080-ti-x4). PyTorch's time is measured with CORL implementations.

# Reports for D4RL mujoco

### Normalized Score
Here, we used [D4RL](https://arxiv.org/abs/2004.07219) mujoco control tasks as the benchmark. We reported the mean and standard deviation of the average normalized score of 5 episodes over 5 seeds.
We plan to extend the verification to other D4RL benchmarks such as AntMaze. For those who would like to know about the source of hyperparameters and the validity of the performance, please refer to [Wiki](https://github.com/nissymori/JAX-CORL/wiki).
|env|AWAC|IQL|TD3+BC|CQL|XQL|DT|
|---|---|---|---|---|---|---|
|halfcheetah-medium-v2| $41.56\pm0.79$ |$46.23\pm0.23$   |$48.12\pm0.42$   |$48.65\pm 0.49$|$47.16\pm0.16$|$42.63 \pm 0.53$|
|halfcheetah-medium-expert-v2| $76.61\pm 9.60$ | $92.95\pm0.79$ | $92.99\pm 0.11$  |$53.76 \pm 14.53$|$86.33\pm$|$70.63\pm 14.70$|
|hopper-medium-v2| $51.45\pm 5.40$  | $56.78\pm3.50$  | $46.51\pm4.57$  |$77.56\pm 7.12$|$62.35\pm5.42$|$60.85\pm6.78$|
|hopper-medium-expert-v2| $51.89\pm2.11$  | $90.72\pm 14.80$  |$105.47\pm5.03$   |$90.37 \pm 31.29$|$104.12\pm5.39$|$109.07\pm 4.56$|
|walker2d-medium-v2| $68.12\pm12.08$ | $77.16\pm5.74$  |  $72.73\pm4.66$ |$80.16\pm 4.19$|$83.45\pm0.42$|$71.04 \pm5.64$|
|walker2d-medium-expert-v2| $91.36\pm23.13$  | $109.08\pm0.77$  | $109.17\pm0.71$  |$110.03 \pm 0.72$|$110.06\pm0.22$|$99.81\pm17.73$|


# How to use this codebase for your research
This codebase can be used independently as a baseline for D4RL projects. It is also designed to be flexible, allowing users to develop new algorithms or adapt them for datasets other than D4RL.

For researchers interested in using this code for their projects, we provide a detailed explanation of the code's shared structure:
##### Data structure

```py
Transition(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray

def get_dataset(...) -> Transition:
    ...
    return dataset
```
The code includes a `Transition` class, defined as a `NamedTuple`, which contains fields for observations, actions, rewards, next observations, and done flags. The get_dataset function is expected to output data in the Transition format, making it adaptable to any dataset that conforms to this structure.

##### Trainer class
```py
class AlgoTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState

class Algo(object):
    ...
    def update_actor(self, train_state: AlgoTrainState, batch: Transition, config) -> AlgoTrainState:
        ...
        return train_state

    def update_critic(self, train_state: AlgoTrainState, batch: Transition, config) -> AlgoTrainState:
        ...
        return train_state

    @partial(jax.jit, static_argnames("n_jitted_updates")
    def update_n_times(self, train_state: AlgoTrainState,  data, n_jitted_updates, config) -> AlgoTrainState:
      for _ in range(n_updates):
        batch = data.sample()
        train_state = self.update_actor(train_state, batch, config)
        agent = self.update_critic(train_state, batch, config)
      return train_state

def create_train_state(...) -> AlgoTrainState:
    # initialize models...
    return AlgoTrainState(
        acotor=actor,
        critic=critic,
    )
```
For all algorithms, we have `TrainState` class (e.g. `TD3BCTrainState` for TD3+BC) which encompasses all `flax` trainstate for models. Update logic is implemented as the method of `Algo` classes (e.g. TD3BC) Both `TrainState` and `Algo` classes are versatile and can be used outside of the provided files if the `create_train_state` function is properly implemented to meet the necessary specifications for the `TrainState` class.
**Note**: So far, we have not followed the policy for CQL due to technical issues. This will be handled in the near future.

# See also
**Great Offline RL libraries**
- [CORL](https://github.com/tinkoff-ai/CORL): Comprehensive single-file implementations of offline RL algorithms in pytorch.

**Implementations of offline RL algorithms in JAX**
- [jaxrl](https://github.com/ikostrikov/jaxrl): Includes implementatin of [AWAC](https://arxiv.org/abs/2006.09359).
- [JaxCQL](https://github.com/young-geng/JaxCQL): Clean implementation of [CQL](https://arxiv.org/abs/2006.04779).
- [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning): Official implementation of [IQL](https://arxiv.org/abs/2110.06169).
- [decision-transformer-jax](https://github.com/yun-kwak/decision-transformer-jax): Jax implementation of [Decision Transformer](https://arxiv.org/abs/2106.01345) with Haiku.
- [td3-bc-jax](https://github.com/ethanluoyc/td3_bc_jax): Direct port of [original implementation](https://github.com/sfujim/TD3_BC) with Haiku.

**Single-file implementations**
- [CleanRL](https://github.com/vwxyzjn/cleanrl): High-quality single-file implementations of online RL algorithms in PyTorch.
- [purejaxrl](https://github.com/luchris429/purejaxrl): High-quality single-file implementations of online RL algorithms in JAX.

# Cite JAX-CORL
```
@article{nishimori2024jaxcorl,
  title={JAX-CORL: Clean Sigle-file Implementations of Offline RL Algorithms in JAX},
  author={Soichiro Nishimori},
  year={2024},
  url={https://github.com/nissymori/JAX-CORL}
}
```

# Credits
- This project is inspired by [CORL](https://github.com/tinkoff-ai/CORL), clean single-file implementations of offline RL algorithm in pytorch.
- I would like to thank [@JohannesAck](https://github.com/johannesack) for his TD3-BC codebase and helpful advices.
- The IQL implementation is based on [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning).
- AWAC implementation is based on [jaxrl](https://github.com/ikostrikov/jaxrl).
- CQL implementation is based on [JaxCQL](https://github.com/young-geng/JaxCQL).
- DT implementation is based on [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer).

