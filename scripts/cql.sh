project=cql-report

cd .. && cd algos
for seed in 1 2 3 4 5
do
    python cql.py env=halfcheetah-medium-v2 seed=$seed project=$project
    python cql.py env=halfcheetah-medium-expert-v2 seed=$seed project=$project
    python cql.py env=hopper-medium-v2 seed=$seed project=$project
    python cql.py env=hopper-medium-expert-v2 seed=$seed project=$project
    python cql.py env=walker2d-medium-v2 seed=$seed project=$project
    python cql.py env=walker2d-medium-expert-v2 seed=$seed project=$project
done