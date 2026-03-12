# Seminar Scripts

## Training & Experiment Management

### train_custom.py
Run training with custom wrappers from YAML config.
```bash
python seminar/utils/train_custom.py --env SeaquestNoFrameskip-v4 \
    -f logs/experiment --algo dqn --conf seminar/experiments/config.yml \
    --seed 42 -P --track --wandb-project-name project
```

### train_scheduler.sh
Parallel training launcher using tmux sessions inside Docker.
```bash
#setup wandb key first
echo "your_api_key" > seminar/.wandb_api_key

#inside docker container
apt-get update && apt-get install -y tmux
tmux new-session -d -s schedule_manager "bash seminar/exp_scripts/train_scheduler.sh"
```

## Environment Analysis

### play_with_ram_debug.py
Interactive environment controller with RAM state display.
```bash
python seminar/utils/play_with_ram_debug.py

#controls: w=up, a=left, s=down, d=right, f=fire, enter=noop, q=quit
```

### annotate_ram.py
Create annotated RAM snapshots during manual gameplay for analysis.
```bash
python seminar/utils/annotate_ram.py

#type description at prompt to save snapshot with current RAM state, empty input for NOOP action
#controls: w=up, a=left, s=down, d=right, f=fire, q=quit
```

### visualize_observations.py
Visualize agent observations after each wrapper is applied.
```bash
python seminar/utils/visualize_observations.py --config CONFIG --env ENV [--mode MODE] [--steps STEPS]

--config CONFIG    YAML config file path defining wrapper stack to visualize
--env ENV          Environment ID to create (default: SeaquestNoFrameskip-v4)
--mode MODE        Visualization mode: 'static' saves images, 'live' shows animation (default: static)
--steps STEPS      Number of random steps to take for static visualization (default: 10)
```

### test_exp_wrapper.py
Test wrapper behavior without training (mocks wandb logging).
```bash
python seminar/utils/test_exp_wrapper.py --config CONFIG [--env ENV] [--observations]

--config CONFIG       YAML config file path defining wrapper stack to test
--env ENV             Environment ID to create (default: SeaquestNoFrameskip-v4)
--observations        Display agent observation space showing what the agent actually sees
```

## Results Analysis

### dl_wandb.py
Download training artifacts and history from wandb.
```bash
python seminar/utils/dl_wandb.py [--groups GROUP [GROUP ...]]

--groups GROUP [GROUP ...]    Only download runs from specified group names (default: all groups)
```

### plot_training_curves.py
Plot training curves comparing multiple experiment groups.
```bash
python seminar/utils/plot_training_curves.py --groups GROUP [GROUP ...] --metric METRIC SMOOTH [METRIC SMOOTH ...] [OPTIONS]

--groups GROUP [GROUP ...]           Group names to plot (required)
--labels LABEL [LABEL ...]           Custom labels for each group (defaults to group names)
--metric METRIC SMOOTH [...]         Metrics to plot as pairs of metric_name and smoothing_window (default: rollout/ep_rew_mean 0)
--x-axis COLUMN                      Column name to use for x-axis (default: global_step)
--artifacts-path PATH                Base path to downloaded artifacts folder (default: seminar/artifacts)
--plots-path PATH                    Base path where plots will be saved (default: seminar/plots)
--folder FOLDER                      Subfolder within plots-path to save outputs
--max-steps N                        Maximum number of steps to include in plot
--resample-freq N                    Resampling frequency for aligning x-axis across runs (default: 1000)
--file-name NAME                     Output filename (defaults to metric name)
--title TITLE                        Plot title override
--verbose                            Enable debug logging to see processing details
--no-million                         Do not convert x-axis to millions
--std                                Plot standard deviation instead of standard error
```

### plot_full_logs.py
Generate detailed analysis plots (heatmaps, radar charts, distributions).
```bash
python seminar/utils/plot_full_logs.py --group GROUP [OPTIONS]

--group GROUP                        Group name to generate plots for (required)
--artifacts-path PATH                Base path to downloaded artifacts folder (default: seminar/artifacts)
--plots-path PATH                    Base path where plots will be saved (default: seminar/plots)
--heatmap [START END]                Create player position heatmap, optionally filtered to global_step range
--radar [START END]                  Create action distribution radar chart, optionally filtered to global_step range
--divers [START END]                 Create divers distribution plot, optionally filtered to global_step range
```
