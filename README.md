# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), clean **single-file** implementation of offline RL algorithms with **solid performance reports**.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit` and `vmap`
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few popular algorithms**: Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms)

# Algorithms
|Algorithm|implementation|training time|
|---|---|---|
|[AWAC](https://arxiv.org/abs/2006.09359)| [algo/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/awac.py) ||
|[IQL](https://arxiv.org/abs/2110.06169)|  [algo/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/iql.py)   || 
|[TD3-BC](https://arxiv.org/pdf/2106.06860)| [algo/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/td3bc.py)  ||  
|[CQL](https://arxiv.org/abs/2006.04779)| 🚧   |-|   
|[DT](https://arxiv.org/abs/2106.01345) | 🚧  |-| 
The training time is the fastest w.r.t. the number of steps in a jitted epoch (`n_updates`). refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#training-speed-with-different-n_updates) for more details.

# Reports with D4RL mujoco

### Normalized Score
Here, we used [D4RL](https://arxiv.org/abs/2004.07219) mujoco control tasks as the benchmark. We reported mean and standard deviation of the average normalized of 5 episodes over 5 seeds.
We plan to extend the verification to other D4RL banchmarks such as AntMaze.
|env|AWAC|IQL|TD3+BC|
|---|---|---|---|
|halfcheetah-medium-expert-v2|   |   |   |   
|halfcheetah-medium-v2|   |   |   |   
|hopper-medium-expert-v2|   |   |   |   
|hopper-medium-v2|   |   |   |   
|walker2d-medium-expert-v2|   |   |   |   
|walker2d-medium-v2|   |   |   |   

### Training speed with different `n_updates`
Rough code for our update logic
```py
class Trainer(Namedtuple):
    critic: TrainState
    actor: TrainState

    def update_actor(agent, batch):
      ...
      return agent

    def update_critic(agent, batch):
      ...
      return agent

    @partial(jax.jit, static_argnames("n_updates")
    def update_n_times(agent, data, n_updates)
      for _ in range(n_updates):
        batch = data.sample()
        agent = update_actor(batch)
        agent = update_critic(batch)
      return agent
```
We measured the training time and first jit time for different `n_updates`. See the figures below. The GPU we used was [GeForce GTX 1080 Ti x4](https://versus.com/en/inno3d-ichill-geforce-gtx-1080-ti-x4).


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
- I would like to thank [@JohannesAck](https://github.com/johannesack) for his TD3-BC code base and helpful advices.
- The IQL implementation is based on [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning).
- AWAC implementation is based on [jaxrl](https://github.com/ikostrikov/jaxrl) and [CORL](https://github.com/tinkoff-ai/CORL).

