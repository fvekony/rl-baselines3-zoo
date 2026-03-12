import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional
import wandb

class MyTestWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_rewards = []
        self.episode_length = 0
        self.total_steps = 0  #track global timesteps
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.episode_rewards = []
        self.episode_length = 0
        obs, info = self.env.reset(seed=seed, options=options)
        
        obs = self._modify_observation(obs)
        
        return obs, info
    
    def step(self, action):
        action = self._modify_action(action)
        
        #step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._modify_observation(obs)
        
        original_reward = reward
        reward = self._modify_reward(reward, obs, action)
        
        #track episode stats
        self.episode_rewards.append(reward)
        self.episode_length += 1
        self.total_steps += 1
        
        metrics = {
            "step_reward": reward,
            "step_reward_delta": reward - original_reward,
        }
        
        # Add episode-level metrics when episode ends
        if terminated or truncated:
            metrics["episode_reward"] = sum(self.episode_rewards)
            metrics["episode_length"] = self.episode_length
        
        # Single wandb.log call per timestep
        wandb.log(metrics, step=self.total_steps)
        
        return obs, reward, terminated, truncated, info
    
    def _modify_observation(self, obs):
        
        return obs
    
    def _modify_action(self, action):
        
        return action
    
    def _modify_reward(self, reward, obs, action):
        
        return reward