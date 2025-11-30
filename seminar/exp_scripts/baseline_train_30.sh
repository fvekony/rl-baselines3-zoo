#!/bin/bash
'''
execute first:
./scripts/run_docker_cpu.sh bash
bash ./seminar/exp_scripts/baseline_train_30.sh 4
'''

cd /seminar

# Get n-jobs from command line argument (default to 4 if not provided)
N_JOBS=${1:-4}

# Generated from initial seed: 42
seeds=(17772 26794 1435 24388 11074 32198 5016 25179 767 5153 1205 3686 30815 10953 31240 3607 17915 1448 6718 24722 22432 31985 32044 17663 1645 29836 22006 24128 8200 31450)

# Run training n times with different seeds
runs=1

for ((i=0; i<runs; i++)); do
    echo "Running experiment $((i+1))/$runs with seed ${seeds[$i]} and $N_JOBS jobs"
    python utils/train_custom.py --env SeaquestNoFrameskip-v4 -f logs/test1_pretrained --algo dqn --conf experiments/seaquest_dqn_v1.yml --seed ${seeds[$i]} -P --n-jobs $N_JOBS -i /home/mambauser/code/rl_zoo3/rl-trained-agents/dqn/SeaquestNoFrameskip-v4_1/SeaquestNoFrameskip-v4.zip
done