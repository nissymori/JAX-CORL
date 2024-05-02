# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), clean **single-file** implementation of offline RL algorithms with **solid performance reports**.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit` and `vmap`
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few popular algorithms**: Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms)

# Algorithms
|Algorithm|implementation|training time| wandb |
|---|---|---|---|
|[AWAC](https://arxiv.org/abs/2006.09359)| [algo/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/awac.py) || [link](https://api.wandb.ai/links/nissymori/2i66gkvj) |
|[IQL](https://arxiv.org/abs/2110.06169)|  [algo/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/iql.py)   || [link](https://api.wandb.ai/links/nissymori/iqo688bi) |
|[TD3-BC](https://arxiv.org/pdf/2106.06860)| [algo/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/td3bc.py)  || [link](https://api.wandb.ai/links/nissymori/h21py327) |
|[CQL](https://arxiv.org/abs/2006.04779)| 🚧   |-|-|
|[DT](https://arxiv.org/abs/2106.01345) | 🚧  |-|-|

We measured the training time (includes compile time) and compile time for different `n_updates`. See the figures below. The GPU we used was [GeForce GTX 1080 Ti x4](https://versus.com/en/inno3d-ichill-geforce-gtx-1080-ti-x4).

# Reports with D4RL mujoco

### Normalized Score
Here, we used [D4RL](https://arxiv.org/abs/2004.07219) mujoco control tasks as the benchmark. We reported mean and standard deviation of the average normalized of 5 episodes over 5 seeds.
We plan to extend the verification to other D4RL banchmarks such as AntMaze.
|env|AWAC|IQL|TD3+BC|
|---|---|---|---|
|halfcheetah-medium-v2| $42.13\pm0.39$ |$43.78\pm0.39$   |$48.12\pm0.42$   |   
|halfcheetah-medium-expert-v2| $61.83\pm 5.89$ | $89.05\pm4.11$ | $92.99\pm 0.11$  |   
|hopper-medium-v2| $53.70\pm5.86$  | $46.51\pm4.56$  | $46.51\pm4.57$  |   
|hopper-medium-expert-v2| $50.90\pm6.48$  | $52.73\pm52.80$  |$105.47\pm5.03$   |   
|walker2d-medium-v2| $62.31\pm15.90$ | $77.87\pm3.16$  |  $72.73\pm4.66$ |   
|walker2d-medium-expert-v2| $78.81\pm27.89$  | $109.08\pm0.25$  | $109.17\pm0.71$  |   


```

# How to use this codebase
You can use this codebase on its own as baseline for D4RL. However, some people want to develop new algorithm based on our code or use this code for other dataset than D4RL.
For those who would like to use this code for your own research, we explain the shared structure of the code. 
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
For the dataset we use the class `Transitions` based on the NamedTuple above. Though we focus on D4RL, as long as you can implement `get_dataset` function to output `Transition` for your own desired dataset, you can use our implementation.

##### Trainer class
```
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

    @partial(jax.jit, static_argnames("n_updates")
    def update_n_times(agent, data, n_updates)
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
For all algorithms, we have `Trainer` class (`TD3BCTrainer` for TD3+BC) which encompasses all necessary components for the algorithm: models, hyperparameters, update logics. Therefore, you can use the class outside of this files if you implement `create_trainer` function satisfying required for `Trainer` class (models and hyperparameters).

# See also
**Great Offline RL libraries**
- [CORL](https://github.com/tinkoff-ai/CORL): Comprehensive single-file implementations of offline RL algorithms in pytorch.

**Implementations of offline RL algorithms in JAX**
- [jaxrl](https://github.com/ikostrikov/jaxrl): Includes implementatin of [AWAC](https://arxiv.org/abs/2006.09359).
- [JaxCQL](https://github.com/young-geng/JaxCQL): Clean implementation of [CQL](https://arxiv.org/abs/2006.04779).
- [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning): Official implementation of [IQL](https://arxiv.org/abs/2110.06169).
- [decision-transformer-jax](https://github.com/yun-kwak/decision-transformer-jax): Jax implementation of [Decision Transformer](https://arxiv.org/abs/2106.01345) with Haiku.
- [td3-bc-jax](https://github.com/ethanluoyc/td3_bc_jax): Direct port of [original implementation](https://github.com/sfujim/TD3_BC) with Haiku.


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
- AWAC implementation is based on [jaxrl](https://github.com/ikostrikov/jaxrl) and [CORL](https://github.com/tinkoff-ai/CORL).

