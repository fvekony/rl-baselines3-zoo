
import gymnasium as gym
import sys
import os
import argparse
import importlib
import yaml
import numpy as np

#mock wandb to print instead of logging
class MockWandb:
    @staticmethod
    def log(metrics, step=None):
        print(f"\n[STEP {step}] WANDB LOG:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

#replace wandb before importing wrapper
sys.modules['wandb'] = MockWandb()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True,
                    help='YAML config file (e.g., experiments/seaquest_dqn_reduced_actions.yml)')
parser.add_argument('--env', type=str, default='SeaquestNoFrameskip-v4',
                    help='Environment ID')
parser.add_argument('--observations', action='store_true',
                    help='Display agent observation space (what the agent sees)')
args, unknown = parser.parse_known_args()

#load config
with open(args.config, 'r') as f:
    all_configs = yaml.safe_load(f)

config = all_configs.get('atari', {})

#register atari environments
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

#action mappings
action_map = {
    'w': 2,   # UP
    's': 5,   # DOWN
    'a': 4,   # LEFT
    'd': 3,   # RIGHT
    'i': 10,  # UPFIRE
    'k': 13,  # DOWNFIRE
    'j': 12,  # LEFTFIRE
    'l': 11,  # RIGHTFIRE
    'f': 1,   # FIRE
    '.': 0,   # NOOP
}

#check if using reduced actions
uses_reduced_actions = False
if 'env_wrapper' in config:
    for wrapper_config in config['env_wrapper']:
        if isinstance(wrapper_config, dict):
            for wrapper_path in wrapper_config.keys():
                if 'ReducedActionsWrapper' in wrapper_path:
                    uses_reduced_actions = True
                    break

if uses_reduced_actions:
    action_map = {
        'w': 1,   # UPFIRE (index 1 in allowed_actions)
        's': 4,   # DOWNFIRE
        'a': 3,   # LEFTFIRE
        'd': 2,   # RIGHTFIRE
        '.': 0,   # NOOP
    }

print("\n" + "="*70)
print("WRAPPER TEST - CONFIG-BASED ENVIRONMENT")
print("="*70)
print("\nControls (type letter then Enter):")
if uses_reduced_actions:
    print("  w - UPFIRE      s - DOWNFIRE")
    print("  a - LEFTFIRE    d - RIGHTFIRE")
    print("  . - NOOP")
else:
    print("  w - UP          s - DOWN")
    print("  a - LEFT        d - RIGHT")
    print("  i - UP+FIRE     k - DOWN+FIRE")
    print("  j - LEFT+FIRE   l - RIGHT+FIRE")
    print("  f - FIRE        . - NOOP")
print("  r - Auto-run 20 random steps")
print("  q - Quit")
print("\nApplying wrappers from config...")
print("="*70 + "\n")

#create base environment
base_env = gym.make(args.env, render_mode="human")

#apply all wrappers from config
env = base_env
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
                
                #disable wandb for testing
                if 'enable_wandb' in wrapper_kwargs:
                    wrapper_kwargs = dict(wrapper_kwargs)
                    wrapper_kwargs['enable_wandb'] = True  #keep enabled to test mock
                
                env = wrapper_class(env, **wrapper_kwargs)
                print(f"Applied: {class_name} with {wrapper_kwargs}")
        else:
            #import wrapper class
            module_path, class_name = wrapper_config.rsplit('.', 1)
            module = importlib.import_module(module_path)
            wrapper_class = getattr(module, class_name)
            
            env = wrapper_class(env)
            print(f"Applied: {class_name}")

print(f"\nFinal action space: {env.action_space}")
print("="*70 + "\n")

#setup observation visualization if requested
obs_fig = None
obs_ax = None
obs_im = None

if args.observations:
    import matplotlib.pyplot as plt
    plt.ion()
    obs_fig, obs_ax = plt.subplots(1, 1, figsize=(6, 6))
    obs_ax.set_title('Agent Observation')
    obs_ax.axis('off')
    obs_fig.show()

obs, info = env.reset()

#update observation visualization
if args.observations and obs_fig:
    display_obs = obs
    if len(display_obs.shape) == 3 and display_obs.shape[2] == 1:
        display_obs = display_obs[:, :, 0]
    
    if obs_im is None:
        obs_im = obs_ax.imshow(display_obs, cmap='gray', vmin=0, vmax=255)
        obs_ax.set_title(f'Agent Observation {display_obs.shape}')
    else:
        obs_im.set_array(display_obs)
        obs_ax.set_title(f'Agent Observation {display_obs.shape}')
    obs_fig.canvas.draw()
    obs_fig.canvas.flush_events()

#access ALE through wrapper stack
unwrapped_env = env
while hasattr(unwrapped_env, 'env'):
    unwrapped_env = unwrapped_env.env
ale = unwrapped_env.ale

#find any wrapper with tracking stats (generic approach)
tracking_wrapper = None
current = env
while current is not None:
    #check for common wrapper attributes
    if hasattr(current, 'starting_lives') or hasattr(current, 'total_steps'):
        tracking_wrapper = current
        break
    current = current.env if hasattr(current, 'env') else None

if tracking_wrapper:
    if hasattr(tracking_wrapper, 'starting_lives'):
        print(f"\nStarting lives: {tracking_wrapper.starting_lives}")
        print(f"Current lives: {tracking_wrapper.current_lives}")
    if hasattr(tracking_wrapper, 'total_steps'):
        print(f"Total steps: {tracking_wrapper.total_steps}")
else:
    print("\nNo tracking wrapper found in stack")

running = True
step_count = 0

while running:
    #display current game state
    ram = ale.getRAM()
    score = int(f"{ram[56]:02x}{ram[57]:02x}{ram[58]:02x}")
    lives = ram[59]
    oxygen = ram[102]
    divers_onboard = ram[62]
    diver_lanes = [ram[i] for i in range(113, 117)]
    
    print(f"\n--- Game State ---")
    print(f"Score: {score:6d} | Lives: {lives} | Oxygen: {oxygen:3d} | Divers onboard: {divers_onboard}")
    print(f"Diver lanes (1=bottom, 4=top): {diver_lanes} | Active: {[i+1 for i, x in enumerate(diver_lanes) if x == 1]}")
    if tracking_wrapper:
        if hasattr(tracking_wrapper, 'life_length'):
            print(f"Life length: {tracking_wrapper.life_length} | Divers picked up: {tracking_wrapper.life_divers_picked_up}")
    
    user_input = input("\nAction: ").lower().strip()
    
    if user_input == 'q':
        break
    
    if user_input == 'r':
        print("\nRunning 20 random steps...")
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            #update observation visualization
            if args.observations and obs_fig:
                display_obs = obs
                if len(display_obs.shape) == 3 and display_obs.shape[2] == 1:
                    display_obs = display_obs[:, :, 0]
                
                obs_im.set_array(display_obs)
                obs_ax.set_title(f'Agent Observation {display_obs.shape}')
                obs_fig.canvas.draw()
                obs_fig.canvas.flush_events()
            
            if reward != 0:
                print(f"  Step {i+1}: action={action}, reward={reward}")
            
            if terminated or truncated:
                print("\n!!! EPISODE ENDED !!!")
                obs, info = env.reset()
                
                #update observation visualization after reset
                if args.observations and obs_fig:
                    display_obs = obs
                    if len(display_obs.shape) == 3 and display_obs.shape[2] == 1:
                        display_obs = display_obs[:, :, 0]
                    
                    obs_im.set_array(display_obs)
                    obs_ax.set_title(f'Agent Observation {display_obs.shape}')
                    obs_fig.canvas.draw()
                    obs_fig.canvas.flush_events()
                
                break
        continue
    
    action = action_map.get(user_input, 0)
    
    #take single step
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    
    #update observation visualization
    if args.observations and obs_fig:
        display_obs = obs
        if len(display_obs.shape) == 3 and display_obs.shape[2] == 1:
            display_obs = display_obs[:, :, 0]
        
        obs_im.set_array(display_obs)
        obs_ax.set_title(f'Agent Observation {display_obs.shape}')
        obs_fig.canvas.draw()
        obs_fig.canvas.flush_events()
    
    if reward != 0:
        print(f"\n>>> REWARD: {reward}")
    
    if terminated or truncated:
        print("\n!!! EPISODE ENDED !!!")
        obs, info = env.reset()
        
        #update observation visualization after reset
        if args.observations and obs_fig:
            display_obs = obs
            if len(display_obs.shape) == 3 and display_obs.shape[2] == 1:
                display_obs = display_obs[:, :, 0]
            
            obs_im.set_array(display_obs)
            obs_ax.set_title(f'Agent Observation {display_obs.shape}')
            obs_fig.canvas.draw()
            obs_fig.canvas.flush_events()
        
        if tracking_wrapper and hasattr(tracking_wrapper, 'starting_lives'):
            print(f"\nNew episode - Starting lives: {tracking_wrapper.starting_lives}")

env.close()
if args.observations and obs_fig:
    plt.close(obs_fig)
print(f"\nTotal steps: {step_count}")
print("Test completed.")
