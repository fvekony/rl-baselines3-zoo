import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import argparse
import importlib
import json
import yaml

#register atari environments
import ale_py
gym.register_envs(ale_py)

sys.path.insert(0, '/home/fvekony/rl-baselines3-zoo')

from rl_zoo3 import ALGOS
from rl_zoo3.exp_manager import ExperimentManager


def create_envs_step_by_step(config_file, env_id):
    #load config
    with open(config_file, 'r') as f:
        all_configs = yaml.safe_load(f)
    
    #get atari config
    config = all_configs.get('atari', {})
    
    envs = []
    
    #create base environment
    base_env = gym.make(env_id, render_mode='rgb_array')
    envs.append((base_env, 'Base env'))
    
    #apply wrappers one by one
    current_env = base_env
    if 'env_wrapper' in config:
        for wrapper_config in config['env_wrapper']:
            if isinstance(wrapper_config, dict):
                for wrapper_path, wrapper_kwargs in wrapper_config.items():
                    #import wrapper class
                    module_path, class_name = wrapper_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    wrapper_class = getattr(module, class_name)
                    
                    if wrapper_kwargs is None:
                        wrapper_kwargs = {}
                    
                    #disable wandb for visualization
                    if 'enable_wandb' in wrapper_kwargs:
                        wrapper_kwargs = dict(wrapper_kwargs)
                        wrapper_kwargs['enable_wandb'] = False
                    
                    current_env = wrapper_class(current_env, **wrapper_kwargs)
                    label = f"{class_name}"
                    envs.append((current_env, label))
                    print(f"Applied wrapper: {wrapper_path} with {wrapper_kwargs}")
            else:
                #import wrapper class
                module_path, class_name = wrapper_config.rsplit('.', 1)
                module = importlib.import_module(module_path)
                wrapper_class = getattr(module, class_name)
                
                current_env = wrapper_class(current_env)
                label = f"{class_name}"
                envs.append((current_env, label))
                print(f"Applied wrapper: {wrapper_config}")
    
    return envs, config


def visualize_static(envs, n_steps=10):
    n_envs = len(envs)
    
    #collect observations for each env
    all_obs = [[] for _ in range(n_envs)]
    labels = [label for _, label in envs]
    
    #reset all envs
    obs_list = [env.reset() for env, _ in envs]
    
    for step_idx in range(n_steps):
        #collect current observations
        for env_idx, (obs_data) in enumerate(obs_list):
            if isinstance(obs_data, tuple):
                obs = obs_data[0]
            else:
                obs = obs_data
            
            #keep RGB or grayscale as-is for proper display
            all_obs[env_idx].append(obs)
        
        #step all environments
        for env_idx, (env, _) in enumerate(envs):
            action = env.action_space.sample()
            result = env.step(action)
            obs, reward, terminated, truncated, info = result if len(result) == 5 else (*result, {})
            done = terminated if isinstance(terminated, bool) else (truncated if isinstance(truncated, bool) else False)
            
            if done:
                obs_list[env_idx] = env.reset()
            else:
                obs_list[env_idx] = obs
    
    #close all envs
    for env, _ in envs:
        env.close()
    
    #plot comparisons
    fig, axes = plt.subplots(n_steps, n_envs, figsize=(5*n_envs, 4*n_steps))
    if n_steps == 1:
        axes = axes.reshape(1, -1)
    if n_envs == 1:
        axes = axes.reshape(-1, 1)
    
    for step_idx in range(n_steps):
        for env_idx in range(n_envs):
            frame = all_obs[env_idx][step_idx]
            
            #display RGB if 3 channels, otherwise grayscale
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                axes[step_idx, env_idx].imshow(frame)
            else:
                #grayscale or single channel
                if len(frame.shape) == 3:
                    frame = frame[:, :, 0]
                axes[step_idx, env_idx].imshow(frame, cmap='gray', vmin=0, vmax=255)
            
            axes[step_idx, env_idx].set_title(f'Step {step_idx} - {labels[env_idx]} {frame.shape}')
            axes[step_idx, env_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('/seminar/observation_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison to /seminar/observation_comparison.png")
    plt.close()


def visualize_live(envs):
    n_envs = len(envs)
    labels = [label for _, label in envs]
    
    #reset all envs
    obs_list = [env.reset() for env, _ in envs]
    
    #create subplot grid
    fig, axes = plt.subplots(1, n_envs, figsize=(7*n_envs, 7))
    if n_envs == 1:
        axes = [axes]
    
    #initialize images
    images = []
    for env_idx, obs_data in enumerate(obs_list):
        if isinstance(obs_data, tuple):
            obs = obs_data[0]
        else:
            obs = obs_data
        
        #display RGB if 3 channels, otherwise grayscale
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            im = axes[env_idx].imshow(obs)
            axes[env_idx].set_title(f'{labels[env_idx]} {obs.shape}')
        else:
            #grayscale or single channel
            frame = obs[:, :, 0] if len(obs.shape) == 3 else obs
            im = axes[env_idx].imshow(frame, cmap='gray', vmin=0, vmax=255)
            axes[env_idx].set_title(f'{labels[env_idx]} {frame.shape}')
        
        axes[env_idx].axis('off')
        images.append(im)
    
    def update(frame_num):
        nonlocal obs_list
        
        #step all environments
        for env_idx, (env, _) in enumerate(envs):
            action = env.action_space.sample()
            result = env.step(action)
            obs, reward, terminated, truncated, info = result if len(result) == 5 else (*result, {})
            done = terminated if isinstance(terminated, bool) else (truncated if isinstance(truncated, bool) else False)
            
            if done:
                obs_list[env_idx] = env.reset()
            else:
                obs_list[env_idx] = obs
            
            #update display
            if isinstance(obs_list[env_idx], tuple):
                display_obs = obs_list[env_idx][0]
            else:
                display_obs = obs_list[env_idx]
            
            #display RGB if 3 channels, otherwise grayscale
            if len(display_obs.shape) == 3 and display_obs.shape[2] == 3:
                images[env_idx].set_array(display_obs)
                axes[env_idx].set_title(f'{labels[env_idx]} {display_obs.shape}')
            else:
                frame = display_obs[:, :, 0] if len(display_obs.shape) == 3 else display_obs
                images[env_idx].set_array(frame)
                axes[env_idx].set_title(f'{labels[env_idx]} {frame.shape}')
        
        return images
    
    anim = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
    
    #close all envs
    for env, _ in envs:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'live'],
                       help='Visualization mode: static (save images) or live (animated)')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of steps for static visualization')
    parser.add_argument('--config', type=str, required=True,
                       help='YAML config file (e.g., experiments/seaquest_dqn_reduced_actions.yml)')
    parser.add_argument('--env', type=str, default='SeaquestNoFrameskip-v4',
                       help='Environment ID')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Creating environments from config: {args.config}")
    print("Showing observations after each wrapper is applied")
    print(f"{'='*60}\n")
    
    envs, config = create_envs_step_by_step(args.config, args.env)
    print(f"\n{'='*60}")
    print(f"Created {len(envs)} environments (base + wrappers)")
    print(f"{'='*60}\n")
    
    if args.mode == 'static':
        visualize_static(envs, n_steps=args.steps)
    else:
        visualize_live(envs)
