#!/bin/bash
# Run trainings in parallel using tmux sessions

# to run:
# Create /seminar/.wandb_api_key file with your API key
# ./scripts/run_docker_cpu.sh bash
# apt-get update && apt-get install -y tmux
# tmux new-session -d -s schedule_manager "bash /seminar/exp_scripts/train_scheduler.sh"

pip install -e .
pip install protobuf==6.31.1
export PYTHONPATH=/home/mambauser/code/rl_zoo3:$PYTHONPATH


WANDB_KEY_FILE="/seminar/.wandb_api_key"
if [ -f "$WANDB_KEY_FILE" ]; then
    WANDB_API_KEY=$(cat "$WANDB_KEY_FILE" | tr -d '[:space:]')
    echo "Logging into wandb with API key from $WANDB_KEY_FILE..."
    wandb login "$WANDB_API_KEY"
else
    echo "Warning: No wandb API key found. tmux sessions need to login to wandb manually."
fi

cd /seminar

# Training parameters
ENV="SeaquestNoFrameskip-v4"
ALGO="dqn"
CONFIG="experiments/seaquest_dqn_balanced_surfacing_wbuffer.yml"
LOG_FOLDER="logs/balanced_surfacing_wbuffer"
N_JOBS=6
PRETRAINED_AGENT="/home/mambauser/code/rl_zoo3/rl-trained-agents/dqn/SeaquestNoFrameskip-v4_1/SeaquestNoFrameskip-v4.zip"
PROGRESS_BAR="-P"  # Use -P for progress bar, or "" to disable
WANDB_ARGS="--track --wandb-project-name dqn-seaquest"  # Enable wandb logging

seeds=(17772 26794 1435 24388)
# 4 initial test seeds: 17772 26794 1435 24388
# rest 26 seeds: 11074 32198 5016 25179 767 5153 1205 3686 30815 10953 31240 3607 17915 1448 6718 24722 22432 31985 32044 17663 1645 29836 22006 24128 8200 31450

parallel_runs=1  # Run trainings in parallel

total_seeds=${#seeds[@]}
echo "Total experiments to run: $total_seeds"
echo "Running up to $parallel_runs in parallel"
echo ""

config_name=$(basename "$CONFIG" .yml)
seed_index=0
running_sessions=()

# Function to start a training session
start_training() {
    local seed=$1
    local session_name="train_${config_name}_${seed}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting experiment with seed $seed in tmux session: $session_name"
    
    # Create tmux session and run training command
    tmux new-session -d -s "$session_name" "export PYTHONPATH=/home/mambauser/code/rl_zoo3:\$PYTHONPATH && cd /seminar && python utils/train_custom.py --env $ENV -f $LOG_FOLDER --algo $ALGO --conf $CONFIG --seed $seed $PROGRESS_BAR -i $PRETRAINED_AGENT --n-jobs $N_JOBS $WANDB_ARGS; echo 'Training finished. Press Ctrl+C to exit or session will close in 10s'; sleep 10"
    running_sessions+=("$session_name")
}

# Function to check and remove finished sessions
update_running_sessions() {
    local temp_sessions=()
    for session in "${running_sessions[@]}"; do
        if tmux has-session -t "$session" 2>/dev/null; then
            temp_sessions+=("$session")
        fi
    done
    running_sessions=("${temp_sessions[@]}")
}

# Start initial batch of parallel runs
while [ $seed_index -lt $parallel_runs ] && [ $seed_index -lt $total_seeds ]; do
    start_training "${seeds[$seed_index]}"
    ((seed_index++))
    sleep 5
done

# Monitor and start new runs as slots become available
while [ $seed_index -lt $total_seeds ]; do
    update_running_sessions
    
    # If there's a free slot, start a new run
    while [ ${#running_sessions[@]} -lt $parallel_runs ] && [ $seed_index -lt $total_seeds ]; do
        start_training "${seeds[$seed_index]}"
        ((seed_index++))
        sleep 5
    done
    
    # Check every 30 seconds
    sleep 30
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments have been started!"
echo "Waiting for remaining experiments to complete..."

# Wait for all remaining sessions to finish
while [ ${#running_sessions[@]} -gt 0 ]; do
    update_running_sessions
    if [ ${#running_sessions[@]} -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Still running: ${running_sessions[@]}"
        sleep 30
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All training sessions completed!"
echo "Ctrl+d to exit."
exec bash