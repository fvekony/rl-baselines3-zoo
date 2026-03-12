# Anwendungsspezifische Reinforcement-Learning Ansätze

This is the seminar-specific readme. For details on the zoo, please see the [original readme](zoo_README.md).

## Setup Stable-Baselines3-Zoo (Docker)

### Clone the repo (including submodules with trained agents)

```bash
git clone https://github.com/fvekony/rl-baselines3-zoo.git
cd rl-baselines3-zoo
git submodule update --init --recursive
```

### Setup DISPLAY to allow rendering from within Docker

```bash
export DISPLAY=:0
xhost +local:docker
```

If running withing docker, this needs to be set up so that the video tests are not skipped. Not strictly necessary for running, though.

### Build the Docker image

Build the docker image locally (because the latest tag has not been pushed). This makes sure the docker image version aligns with the code.
Choose between CPU or GPU version.
More details in the [docs](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).

```bash
make docker-cpu
```

```bash
make docker-gpu
```

### Verify installation

```bash
./scripts/run_docker_cpu.sh pytest tests/test_*.py
```

Runs the existing pytests. Tests should pass.
Warnings about using gym instead of gymnasium are expected and can be ignored for now.

```bash
./scripts/run_docker_cpu.sh bash

pip install -e . &&
pip install protobuf==6.31.1 &&
export PYTHONPATH=/home/mambauser/code/rl_zoo3:$PYTHONPATH

python -m rl_zoo3.enjoy

python -m rl_zoo3.enjoy --env SeaquestNoFrameskip-v4 --algo dqn -f seminar/logs/baseline_pretrained_1M_full-logs -n 10000 --exp-id 1
python -m rl_zoo3.enjoy --env SeaquestNoFrameskip-v4 --algo dqn -f logs -n 10000 --exp-id 1
```

This starts an interactive docker container session and runs the enjoy script with default parameters. You should see a window pop up with a rendering of the environment (CartPole-v1).

## Tailoring environments / algorithms (suggested workflow)

### Implement your changes as wrapper(s)

Create a new file in `seminar/wrapper/` where you implement your wrapper. For an example, see [`seminar/wrapper/custom_wrapper.py`](seminar/wrapper/custom_wrapper.py). Make sure to inherit from `gym.Wrapper` or any of its subclasses. For a general documentation on wrappers see the [gymnasium documentation](https://gymnasium.farama.org/api/wrappers/) and for specific examples of wrappers in this repo see the existing wrappers in `rl_zoo3/wrappers/`.

### Create config file

Create a new config file in `seminar/experiments` based on the original hyperparameter config files in the `hyperparams` folder. You can copy the existing settings for `atari` for the algorithm you want to use and modify them as needed. Make sure to specify your custom wrapper in the config file. For an example, see [`seminar/experiments/sample_experiment_a2c.yml`](seminar/experiments/sample_experiment_a2c.yml).

### Run training (inside docker)

You can run training using the provided training scripts. This will make it easiest to plot the results afterwards. For example, to train an A2C agent with your custom wrapper and config file, you can run:

```bash
./scripts/run_docker_cpu.sh bash
pip install -e .
pip install protobuf==6.31.1
export PYTHONPATH=/home/mambauser/code/rl_zoo3:$PYTHONPATH
cd /seminar
python utils/train_custom.py --env SeaquestNoFrameskip-v4 -f logs/dqn/test --algo dqn --conf experiments/seaquest_dqn_reduced_observations.yml --seed 1337 -i /home/mambauser/code/rl_zoo3/rl-trained-agents/dqn/SeaquestNoFrameskip-v4_1/SeaquestNoFrameskip-v4.zip -P --track --wandb-project-name dqn-seaquest

```

This will start training the agent with the specified environment, algorithm, config file, and seed. The trained model and logs will be saved in the `logs/` directory.

### Plot training progress

After training, you can plot the training progress using the provided plotting scripts. For example, to plot the results of your training run, you can run:

```bash
./scripts/run_docker_cpu.sh bash
cd /seminar
python -m rl_zoo3.plots.plot_train --algo dqn --env SeaquestNoFrameskip-v4 --exp-folder logs/baseline_pretrained_3M logs/test --file_name dqn_plot_train.png
```

This will generate a plot of the training progress and save them in the current directory with the given name. It will include multiple results runs if they exist.

### Plot evaluation results

When training with the script, the agents are intermittently evaluated using an EvaluationCallback. The results of these evaluations are saved in the `logs/` directory as well. You can plot these evaluation results using the following command:

```bash
./scripts/run_docker_cpu.sh bash
python -m rl_zoo3.plots.all_plots --algo dqn --env SeaquestNoFrameskip-v4 --exp-folder logs/baseline_pretrained_3M logs/test --file_name dqn_all_plots.png
```

This will generate a plot including confidence intervals for the evaluation results and save it in the current directory with the given name. It will include multiple results runs if they exist.

## Experiment tips and tricks

A rough example report can be found in [`seminar/example_report.md`](seminar/example_report.md).

### Practical concerns

Training RL agents can be quite time consuming. Make sure to plan accordingly and consider using a machine with a dedicated GPU if possible (and then use the GPU versions of the scripts - as of yet untested, though). Some thoughts to potentially reduce training time:

* The exact choice of timesteps depends on the setting. It makes sense to:
  * Start with a lower number of timesteps to quickly test if everything works as intended.
  * Once everything works, increase the number of timesteps to a point that makes sense (either due to training duration, or, ideally, because the training converges).
  * Then use that number of timesteps for all experiments to ensure comparability.
* Consider using a compatible pre-trained agent as a starting point. This can significantly reduce training time.

### Repeatability and reproducability

To ensure repeatability, please ensure that all commands, seeds and package versions are documented. The complete results (including logs) should also be saved alongside the report and made available.

The rl-zoo also provides an integration to [wandb](https://wandb.ai/) for experiment tracking as well as tensorboard logging. For more details see the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html). You are definitely welcome and encouraged to use these tools for your experiments. I will also provide support where I can, but won't provide a full tutorial on these tools.

### Multiple repeats

All plotting scripts automatically find all data in the passed folder that matches the passed algorithm and environment. So the training could just be repeated multiple times with different seeds and the results would be aggregated in the plots. This is needed to be able to show appropriate confidence intervals.

At least 30 times

```
python utils/train_custom.py --env PongNoFrameskip-v4 --algo a2c --conf experiments/v1.yml --seed seeds[i]
```

where `seeds` is a list of different seed values and `i` ranges from 0 to 29.

The plotting script automatically increments the folder names, so the results show up in folder `{logs}/{algo}/{env}_i` where i is the run index. When passing the algorithm and env to the plotting script, all the results are incorporated as intended.

### Comparing across versions of the same environment

:alert: The plot scripts all find the relevant data by comparing the directory names. Passing the algorithm and env command line parameters thus causes the script to include data found in the folder in `{logs}/{algor}/{env}_i`. Therefore, if we want to test slight variations of the environment, we can simply rename the folders.

Let's say we have to versions of our experiments. Both have the same algorithm (a2c) and the same environment (PongNoFrameskip-v4), but one uses a custom wrapper that modifies the reward structure. This is specified in a different config file. Then, I could:

Run first version first (30 times)

```
python utils/train_custom.py --env PongNoFrameskip-v4 --algo a2c --conf experiments/v1.yml --seed 1337 
```

Then rename all created folders from `logs/a2c/PongNoFrameskip-v4_i` to `logs/a2c/PongNoFrameskipV1-v4_i` (where i is the run index).

Then run the second version (30 times)

```
python utils/train_custom.py --env PongNoFrameskip-v4 --algo a2c --conf experiments/v2.yml --seed 1337 
```

Then rename all created folders from `logs/a2c/PongNoFrameskip-v4_i` to `logs/a2c/PongNoFrameskipV2-v4_i` (where i is the run index).

Now we can use `PongNoFrameskipV1-v4` and `PongNoFrameskipV2-v4` as environment names to compare the two versions in the plot scripts.

### Exchange ideas and code

The marks for this seminar are individually awarded, but they mainly are about defining the ideas for tailoring and their presentation (in the talk and in the report). So, as long as you disclose it, I think it makes a lot of sense to share code, especially for additional tools such as plotting scripts, or even wrappers that you think others might find useful. Just make sure to give credit where credit is due.
