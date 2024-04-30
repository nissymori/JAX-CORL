# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), high-quality single-file implementation of offline RL algorithms.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit`, `vmap`, and `pmap`.
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few important algorithms**: we do not cover all algos. in [CORL](https://github.com/tinkoff-ai/CORL). Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms)

# Algorithms
|Algorithm|implementation|training time|training time in corl|
|---|---|---|---|
|AWAC| [algo/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/awac.py) | | | 
|CQL| [algo/cql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/cql.py)  | | |   
|IQL|  [algo/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/iql.py)   |   |   |   
|TD3-BC| [algo/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/td3bc.py)  |   |   |   



# Report

Normalized Score
|env|AWAC|CQL|IQL|TD3+BC|
|---|---|---|---|---|
|halfcheetah-medium-expert-v2|   |   |   |   |
|halfcheetah-medium-v2|   |   |   |   |
|hopper-medium-expert-v2|   |   |   |   |
|hopper-medium-v2|   |   |   |   |
|walker2d-medium-expert-v2|   |   |   |   |
|walker2d-medium-v2|   |   |   |   |
