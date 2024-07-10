project=td7-report

cd .. && cd algos
for seed in 1 2 3 4 5
do
    python td7.py env_name=halfcheetah-medium-v2 seed=$seed project=$project
    python td7.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=$project
    python td7.py env_name=hopper-medium-v2 seed=$seed project=$project
    python td7.py env_name=hopper-medium-expert-v2 seed=$seed project=$project
    python td7.py env_name=walker2d-medium-v2 seed=$seed project=$project
    python td7.py env_name=walker2d-medium-expert-v2 seed=$seed project=$project
done