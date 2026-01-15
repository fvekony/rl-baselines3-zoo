import gymnasium as gym
import numpy as np
from typing import Optional
import wandb
from collections import Counter


class ExpWrapper(gym.Wrapper):
    
    def __init__(self, env: gym.Env, log_interval: int = 50):
        super().__init__(env)
        self.log_interval = log_interval
        
        # Cumulative stats across all episodes
        self.total_steps = 0
        self.total_episodes = 0
        
        #get action space size
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
        else:
            self.n_actions = None
            
        # get action meanings
        self.action_meanings = self._get_action_meanings()
        if self.action_meanings:
            print("\n=== Action Space ===")
            for i, meaning in enumerate(self.action_meanings):
                print(f"Action {i}: {meaning}")
            print("===================\n")
            
        # init episode stats
        self.reset_episode_stats()
        
    def _get_action_meanings(self):
        try:
            # Unwrap to find the base environment
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            if hasattr(env, 'get_action_meanings'):
                return env.get_action_meanings()
        except:
            pass
        return None
        
    def reset_episode_stats(self):
        self.episode_rewards = []
        self.episode_length = 0
        self.episode_action_counts = [0] * self.n_actions if self.n_actions else []
        self.episode_lives_lost = 0
        self.starting_lives = None
        self.current_lives = None
        self.last_action = None
        self.consecutive_action_count = 0
        self.max_consecutive_actions = 0
        # TODO: Add RAM-based death cause tracking (oxygen, collision, etc.)
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        #log previous episode summary if this isn't the first reset
        if self.episode_length > 0:
            self._log_episode_summary()
            
        self.reset_episode_stats()
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Track starting lives from info dict
        self.starting_lives = info.get('lives', 3)
        self.current_lives = self.starting_lives
        
        return self._modify_observation(obs), info
        
    def step(self, action):
        action = self._modify_action(action)
        
        if self.n_actions:
            self.episode_action_counts[action] += 1
            
        #track consecutive actions to detect stuck behavior
        if action == self.last_action:
            self.consecutive_action_count += 1
            self.max_consecutive_actions = max(
                self.max_consecutive_actions, 
                self.consecutive_action_count
            )
        else:
            self.consecutive_action_count = 1
        self.last_action = action
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._modify_observation(obs)
        
        # track lives and detect life loss
        current_lives = info.get('lives', self.current_lives)
        if current_lives < self.current_lives:
            self.episode_lives_lost += 1
            
            # Log life loss event immediately
            wandb.log({
                "event/life_lost": 1,
                "event/lives_remaining": current_lives,
                # TODO: Add death cause when RAM reading is implemented
            }, step=self.total_steps)
            
        self.current_lives = current_lives
        
        original_reward = reward
        reward = self._modify_reward(reward, obs, action, info)
        
        # Track episode stats
        self.episode_rewards.append(reward)
        self.episode_length += 1
        self.total_steps += 1
        
        # Log step-level metrics periodically or at episode end
        should_log = (self.total_steps % self.log_interval == 0) or terminated or truncated
        
        if should_log:
            metrics = {
                "step_reward": reward,
                "step_reward_delta": reward - original_reward,
            }
            
            # Add episode-level metrics when episode ends
            if terminated or truncated:
                metrics.update({
                    "episode_reward": sum(self.episode_rewards),
                    "episode_length": self.episode_length,
                    "episode_lives_lost": self.episode_lives_lost,
                })
                
            wandb.log(metrics, step=self.total_steps)
        
        return obs, reward, terminated, truncated, info
        
    def _log_episode_summary(self):
        self.total_episodes += 1
        
        total_reward = sum(self.episode_rewards)
        avg_reward_per_step = total_reward / max(1, self.episode_length)
        
        # calc action distribution metrics
        action_metrics = {}
        if self.n_actions and self.episode_length > 0:
            #individual action counts
            for i, count in enumerate(self.episode_action_counts):
                action_metrics[f"actions/action_{i}_count"] = count
                action_metrics[f"actions/action_{i}_pct"] = count / self.episode_length
                
            #action entropy (measure of exploration)
            action_probs = np.array(self.episode_action_counts) / self.episode_length
            action_probs = action_probs[action_probs > 0]  # Remove zeros
            action_entropy = -np.sum(action_probs * np.log(action_probs))
            action_metrics["actions/entropy"] = action_entropy
            
            #unique actions used
            unique_actions = np.count_nonzero(self.episode_action_counts)
            action_metrics["actions/unique_count"] = unique_actions
        
        # episode summary metrics
        summary = {
            # Core episode stats
            "episode/total_reward": total_reward,
            "episode/length": self.episode_length,
            "episode/lives_lost": self.episode_lives_lost,
            "episode/avg_reward_per_step": avg_reward_per_step,
            
            # Behavioral patterns
            "episode/max_consecutive_actions": self.max_consecutive_actions,
            
            # Survival metrics
            "episode/survival_rate": (self.starting_lives - self.episode_lives_lost) / max(1, self.starting_lives),
            "episode/steps_per_life": self.episode_length / max(1, self.episode_lives_lost + 1),
            
            # Add action metrics
            **action_metrics,
        }
        
        wandb.log(summary, step=self.total_steps)
        
    def _modify_observation(self, obs):
        
        return obs
    
    def _modify_action(self, action):
        
        return action
    
    def _modify_reward(self, reward, obs, action, info):
        
        return reward
