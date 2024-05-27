# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), clean **single-file** implementations of offline RL algorithms with **solid performance reports**.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit` and `vmap`.
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few popular algorithms**: Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms).
- 📈　Solid performance report ([README](https://github.com/nissymori/JAX-CORL?tab=readme-ov-file#reports-for-d4rl-mujoco), [Wiki](https://github.com/nissymori/JAX-CORL/wiki))

JAX-CORL is complelenting single-file RL ecosystem by offering the combination of offline x JAX. 
- [CleanRL](https://github.com/vwxyzjn/cleanrl): Online x PyTorch
- [purejaxrl](https://github.com/luchris429/purejaxrl): Online x JAX
- [CORL](https://github.com/tinkoff-ai/CORL): Offline x PyTorch
- **JAX-CORL(ours): Offline x JAX**

# Algorithms
|Algorithm|implementation|training time| wandb |
|---|---|---|---|
|[AWAC](https://arxiv.org/abs/2006.09359)| [algos/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/awac.py) |665.5s(~11m)| [link](https://api.wandb.ai/links/nissymori/mwi235j6) |
|[IQL](https://arxiv.org/abs/2110.06169)|  [algos/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/iql.py)   |516.5s (~9m)| [link](https://api.wandb.ai/links/nissymori/iqo688bi) |
|[TD3+BC](https://arxiv.org/pdf/2106.06860)| [algos/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/td3bc.py)  |524.4s (~9m)| [link](https://api.wandb.ai/links/nissymori/h21py327) |
|[CQL](https://arxiv.org/abs/2006.04779)| [algos/cql.py](https://github.com/nissymori/JAX-CORL/blob/main/algos/cql.py)   |3304.1s (~56m)|[link](https://api.wandb.ai/links/nissymori/cnxdwkgf)|
|[DT](https://arxiv.org/abs/2106.01345) |🚧|-|-|

Training time is average training time for 1000_000 update steps with 1000_000 samples for halfcheetah-medium-expert v2 (little difference between different [D4RL](https://arxiv.org/abs/2004.07219) mujoco environment) over 5 seeds. It includes the compile time for `jit`. The computations were performed using four [GeForce GTX 1080 Ti GPUs](https://versus.com/en/inno3d-ichill-geforce-gtx-1080-ti-x4).

# Reports for D4RL mujoco

### Normalized Score
Here, we used [D4RL](https://arxiv.org/abs/2004.07219) mujoco control tasks as the benchmark. We reported mean and standard deviation of the average normalized of 5 episodes over 5 seeds.
We plan to extend the verification to other D4RL banchmarks such as AntMaze. For those who would like to know about the source of hyperparameters and the validity of the performance, please refer to [Wiki](https://github.com/nissymori/JAX-CORL/wiki)
|env|AWAC|IQL|TD3+BC|CQL|
|---|---|---|---|---|
|halfcheetah-medium-v2| $41.56\pm0.79$ |$43.78\pm0.39$   |$48.12\pm0.42$   |$48.65\pm 0.49$|
|halfcheetah-medium-expert-v2| $76.61\pm 9.60$ | $89.05\pm4.11$ | $92.99\pm 0.11$  |$53.76 \pm 14.53$| 
|hopper-medium-v2| $51.45\pm 5.40$  | $46.51\pm4.56$  | $46.51\pm4.57$  |$77.56\pm 7.12$|
|hopper-medium-expert-v2| $51.89\pm2.11$  | $52.73\pm7.80$  |$105.47\pm5.03$   |$90.37 \pm 31.29$|
|walker2d-medium-v2| $68.12\pm12.08$ | $77.87\pm3.16$  |  $72.73\pm4.66$ |$80.16\pm 4.19$|
|walker2d-medium-expert-v2| $91.36\pm23.13$  | $109.08\pm0.25$  | $109.17\pm0.71$  |$110.03 \pm 0.72$|


# How to use this codebase for your own research
This codebase can be used independently as a baseline for D4RL projects. It is also designed to be flexible, allowing users to develop new algorithms or adapt it for datasets other than D4RL.

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
The code includes a `Transition` class, defined as a `NamedTuple`, which includes fields for observations, actions, rewards, next observations, and done flags. The get_dataset function is expected to output data in the Transition format, making it adaptable to any dataset that conforms to this structure.

##### Trainer class
```py
class Trainer(NamedTuple):
    actor: TrainState
    critic: TrainState
    # hyper parameter
    discount: float = 0.99
    ...
    def update_actor(agent, batch: Transition):
        ...
        return agent

    def update_critic(agent, batch: Transition):
        ...
        return agent

    @partial(jax.jit, static_argnames("n_jitted_updates")
    def update_n_times(agent, data, n_jitted_updates)
      for _ in range(n_updates):
        batch = data.sample()
        agent = update_actor(batch)
        agent = update_critic(batch)
      return agent

def create_trainer(...):
    # initialize models...
    return Trainer(
        acotor=actor,
        critic=critic,
        discount=discount
    )
```
For all algorithms, we have `Trainer` class (e.g. `TD3BCTrainer` for TD3+BC) which encompasses all necessary components for the algorithm: models, hyperparameters, update logics. The Trainer class is versatile and can be used outside of the provided files if the create_trainer function is properly implemented to meet the necessary specifications for the Trainer class. This includes setting up the models and defining hyperparameters.

**Note**: So far, we could not follow the policy for CQL due to technical issue. This will be handled in near future.

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
@article{nishimori2022jaxcorl,
  title={JAX-CORL: Clean Sigle-file Implementations of Offline RL Algorithms in JAX},
  author={Soichiro Nishimori},
  year={2024},
  url={https://github.com/nissymori/JAX-CORL}
}
```

# Credits
- This project is inspired by [CORL](https://github.com/tinkoff-ai/CORL), a clean single-file implementations of offline RL algorithm in pytorch.
- I would like to thank [@JohannesAck](https://github.com/johannesack) for his TD3-BC codebase and helpful advices.
- The IQL implementation is based on [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning).
- AWAC implementation is based on [jaxrl](https://github.com/ikostrikov/jaxrl).
- CQL implementation is based on [JaxCQL](https://github.com/young-geng/JaxCQL).
- DT implementation is based on [min-decision-transformer](https://github.com/nikhilbarhate99/min-decision-transformer).

