project=xql-report

cd .. && cd algos
for seed in 1 2 3 4 5
do
    python xql.py env_name=halfcheetah-medium-v2 seed=$seed project=$project max_clip=7.0 noise=true loss_temp=1.0
    python xql.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=$project max_clip=5.0 loss_tmp=1.0
    python xql.py env_name=hopper-medium-v2 seed=$seed project=$project max_clip=7.0 loss_temp=5.0
    python xql.py env_name=hopper-medium-expert-v2 seed=$seed project=$project max_clip=7.0 loss_temp=2.0 sample_random_times=1
    python xql.py env_name=walker2d-medium-v2 seed=$seed project=$project max_clip=7.0 loss_temp=10.0
    python xql.py env_name=walker2d-medium-expert-v2 seed=$seed project=$project max_clip=5.0 loss_temp=2.0
done
