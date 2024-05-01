# TD3-BC
cd algo
for seed in 1 2 3 4 5
do  
    python td3bc.py env_name=halfcheetah-medium-expert-v2 seed=$seed
    python td3bc.py env_name=hopper-medium-expert-v2 seed=$seed
    python td3bc.py env_name=walker2d-medium-expert-v2 seed=$seed

    python td3bc.py env_name=halfcheetah-medium-v2 seed=$seed
    python td3bc.py env_name=hopper-medium-v2 seed=$seed
    python td3bc.py env_name=walker2d-medium-v2 seed=$seed
done

# IQL
for seed in 1 2 3 4 5
do  
    python iql.py env_name=halfcheetah-medium-expert-v2 seed=$seed
    python iql.py env_name=hopper-medium-expert-v2 seed=$seed expectile=0.5 temperature=6.0
    python iql.py env_name=walker2d-medium-expert-v2 seed=$seed

    python iql.py env_name=halfcheetah-medium-v2 seed=$seed
    python iql.py env_name=hopper-medium-v2 seed=$seed
    python iql.py env_name=walker2d-medium-v2 seed=$seed
done

# AWAC
for seed in 1 2 3 4 5
do  
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed
    python awac.py env_name=hopper-medium-expert-v2 seed=$seed
    python awac.py env_name=walker2d-medium-expert-v2 seed=$seed

    python awac.py env_name=halfcheetah-medium-v2 seed=$seed
    python awac.py env_name=hopper-medium-v2 seed=$seed
    python awac.py env_name=walker2d-medium-v2 seed=$seed
done

