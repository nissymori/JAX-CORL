project=dt-report

cd .. && cd algos
for seed in 1 2 3 4 5
do
    python dt.py env_name=halfcheetah-medium-v2 seed=$seed project=$project
    python dt.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=$project
    python dt.py env_name=hopper-medium-v2 seed=$seed project=$project
    python dt.py env_name=hopper-medium-expert-v2 seed=$seed project=$project
    python dt.py env_name=walker2d-medium-v2 seed=$seed project=$project
    python dt.py env_name=walker2d-medium-expert-v2 seed=$seed project=$project
done