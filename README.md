# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), high-quality single-file implementation of offline RL algorithms.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit` and `vmap`
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few important algorithms**: we do not cover all algos. in [CORL](https://github.com/tinkoff-ai/CORL). Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms)

# Algorithms
|Algorithm|implementation|training time|
|---|---|---|
|AWAC| [algo/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/awac.py) ||
|IQL|  [algo/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/iql.py)   || 
|TD3-BC| [algo/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/td3bc.py)  ||  
|CQL| 🚧   |-|   
|[DT](https://arxiv.org/abs/2106.01345) | 🚧  |-| 




# Report

Normalized Score
|env|AWAC|IQL|TD3+BC|
|---|---|---|---|
|halfcheetah-medium-expert-v2|   |   |   |   
|halfcheetah-medium-v2|   |   |   |   
|hopper-medium-expert-v2|   |   |   |   
|hopper-medium-v2|   |   |   |   
|walker2d-medium-expert-v2|   |   |   |   
|walker2d-medium-v2|   |   |   |   

# See also
**General offline algo repo**
- [CORL](https://github.com/tinkoff-ai/CORL): Comprehensive single-file implementations of offline RL algorithms in pytorch.

**Implementations of offline RL algorithms in JAX**
- [jaxrl](https://github.com/ikostrikov/jaxrl): Includes implementatin of AWAC.
- [JaxCQL](https://github.com/young-geng/JaxCQL): Clean implementation of CQL
- [implicit_q_learning](https://github.com/ikostrikov/implicit_q_learning): Official implementation of IQL in jax
- [decision-transformer-jax](https://github.com/yun-kwak/decision-transformer-jax): Jax implementation of [Decision Transformer](https://arxiv.org/abs/2106.01345)
- [td3-bc-jax](https://github.com/ethanluoyc/td3_bc_jax): Direct port of [original implementation](https://github.com/sfujim/TD3_BC) in jax with Haiku

