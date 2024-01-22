# JAX-CORL
This repository aims JAX version of [CORL](https://github.com/tinkoff-ai/CORL), high-quality single-file implementation of offline RL algorithms.
- 🌬️ Persuing **fast** training: speed up via jax functions such as `jit`, `vmap`, and `pmap`.
- 🔪 As **simple** as possible: implement minimum requirements.
- 💠 Focus on **a few important algorithms**: we do not cover all algos. in [CORL](https://github.com/tinkoff-ai/CORL). Refer to

# Algorithms
| Algorithm | copied single file | implementation | wandb report |
|---|---|---|---|


# Implementation policy
1. Try to find working jax implementation (if not, find pytorch version).
2. Put the necessary code into single file.
3. Convert to Clean Jax code step by step.
