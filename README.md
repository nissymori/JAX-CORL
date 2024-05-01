# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), clean **single-file** implementation of offline RL algorithms with **solid performance reports**.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit` and `vmap`
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few important algorithms**: Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms)

# Algorithms
|Algorithm|implementation|training time|
|---|---|---|
|[AWAC](https://arxiv.org/abs/2006.09359)| [algo/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/awac.py) ||
|[IQL](https://arxiv.org/abs/2110.06169)|  [algo/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/iql.py)   || 
|[TD3-BC](https://arxiv.org/pdf/2106.06860)| [algo/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/td3bc.py)  ||  
|[CQL](https://arxiv.org/abs/2006.04779)| 🚧   |-|   
|[DT](https://arxiv.org/abs/2106.01345) | 🚧  |-| 




# Reports with D4RL mujoco

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
- AWAC implementation is based on [jaxrl](https://github.com/ikostrikov/jaxrl) and[CORL](https://github.com/tinkoff-ai/CORL).

