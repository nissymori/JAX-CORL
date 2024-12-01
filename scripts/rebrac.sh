project=rebrac-report

cd .. && cd algos
for seed in 1 2 3 4 5
do
    python rebrac.py env_name=halfcheetah-medium-v2 seed=$seed project=$project actor_bc_coef=0.001 critic_bc_coef=0.01
    python rebrac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=$project actor_bc_coef=0.01 critic_bc_coef=0.1
    python rebrac.py env_name=hopper-medium-v2 seed=$seed project=$project actor_bc_coef=0.01 critic_bc_coef=0.01
    python rebrac.py env_name=hopper-medium-expert-v2 seed=$seed project=$project actor_bc_coef=0.1 critic_bc_coef=0.01
    python rebrac.py env_name=walker2d-medium-v2 seed=$seed project=$project actor_bc_coef=0.05 critic_bc_coef=0.1
    python rebrac.py env_name=walker2d-medium-expert-v2 seed=$seed project=$project actor_bc_coef=0.01 critic_bc_coef=0.01
done