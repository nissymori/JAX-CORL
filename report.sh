# IQL
cd algo
for seed in 1 2 3 4 5
do  
    python iql.py env_name=hopper-medium-expert-v2 seed=$seed project=report
done

# AWAC
for seed in 1 2 3 4 5
do  
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=report
    python awac.py env_name=hopper-medium-expert-v2 seed=$seed project=report
    python awac.py env_name=walker2d-medium-expert-v2 seed=$seed project=report

    python awac.py env_name=halfcheetah-medium-v2 seed=$seed project=report
    python awac.py env_name=hopper-medium-v2 seed=$seed project=report
    python awac.py env_name=walker2d-medium-v2 seed=$seed project=report
done

