# TD3-BC
cd algo
for seed in 1 2 3 4 5
do  
    python td3bc.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=4
    python td3bc.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=8
    python td3bc.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=16
    python td3bc.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=32
    python td3bc.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=64
done

# IQL
for seed in 1 2 3 4 5
do  
    python iql.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=4
    python iql.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=8
    python iql.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=16
    python iql.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=32
    python iql.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=64
done

# AWAC
for seed in 1 2 3 4 5
do  
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=4
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=8
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=16
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=32
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=measure_time n_updates=64
done

