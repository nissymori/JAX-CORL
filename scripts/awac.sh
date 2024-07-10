project=awac-report

cd .. && cd algos
for seed in 1 2 3 4 5
do
    python awac.py env=halfcheetah-medium-v2 seed=$seed project=$project
    python awac.py env=halfcheetah-medium-expert-v2 seed=$seed project=$project
    python awac.py env=hopper-medium-v2 seed=$seed project=$project
    python awac.py env=hopper-medium-expert-v2 seed=$seed project=$project
    python awac.py env=walker2d-medium-v2 seed=$seed project=$project
    python awac.py env=walker2d-medium-expert-v2 seed=$seed project=$project
done