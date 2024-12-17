project=awac-report-separate

cd .. && cd algos
for seed in 1 2 3 4 5 6 7 8 9 10
do
    python awac.py env_name=halfcheetah-medium-v2 seed=$seed project=$project
    python awac.py env_name=halfcheetah-medium-expert-v2 seed=$seed project=$project
    python awac.py env_name=hopper-medium-v2 seed=$seed project=$project
    python awac.py env_name=hopper-medium-expert-v2 seed=$seed project=$project
    python awac.py env_name=walker2d-medium-v2 seed=$seed project=$project
    python awac.py env_name=walker2d-medium-expert-v2 seed=$seed project=$project
done