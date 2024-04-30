# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), high-quality single-file implementation of offline RL algorithms.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit`, `vmap`, and `pmap`.
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few important algorithms**: we do not cover all algos. in [CORL](https://github.com/tinkoff-ai/CORL). Refer [here](https://github.com/nissymori/JAX-CORL/blob/main/README.md#algorithms)

# Algorithms
| Algorithm | implementation | wandb report |
|---|---|---|
|AWAC| [algo/awac.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/awac.py)   |   |
|CQL|   |   |  
|IQL|  [algo/iql.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/iql.py)   |   |  
|TD3-BC| [algo/td3_bc.py](https://github.com/nissymori/JAX-CORL/blob/main/algo/td3bc.py)  |   |
|DT|   |   |  


# Report


# Implementation policy
1. Try to find working jax implementation (if not, find pytorch version).
2. Put the necessary code into single file.
3. Convert to Clean Jax code step by step.
