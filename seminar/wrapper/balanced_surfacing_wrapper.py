import gymnasium as gym
import numpy as np
from typing import Optional
import wandb
from collections import Counter


class BalancedSurfacingWrapper(gym.Wrapper):
    
    def __init__(self, env: gym.Env, log_interval: int = 50, enable_wandb: bool = False):
        super().__init__(env)
        self.log_interval = log_interval
            
        self.wandb_enabled = enable_wandb
        
        self.total_steps = 0
        self.total_episodes = 0
        
        #get action space size
        if isinstance(env.action_space, gym.spaces.Discrete):
            self.n_actions = env.action_space.n
        else:
            self.n_actions = None
            
        self.action_meanings = self._get_action_meanings()
        if self.action_meanings:
            print("\n=== Action Space ===")
            for i, meaning in enumerate(self.action_meanings):
                print(f"Action {i}: {meaning}")
            print("===================\n")
            
        #map action indices to movement directions
        self.movement_actions = {
            1: 'FIRE',
            2: 'UP', 
            3: 'RIGHT',
            4: 'LEFT',
            5: 'DOWN',
            6: 'UPRIGHT',
            7: 'UPLEFT',
            8: 'DOWNRIGHT',
            9: 'DOWNLEFT',
        }
        
        self.previous_ram = None
        
        # init life stats
        self.reset_life_stats()
        
        # init round stats (tracks across multiple lives until game over)
        self.reset_round_stats()
        self.previous_lives_on_reset = None
        
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
        
    def reset_life_stats(self):
        self.life_rewards = []
        self.life_length = 0
        self.life_action_counts = [0] * self.n_actions if self.n_actions else []
        self.life_lives_lost = 0
        self.starting_lives = None
        self.current_lives = None
        self.last_action = None
        self.consecutive_action_count = 0
        self.max_consecutive_actions = 0
        
        #life-level counters
        self.life_divers_picked_up = 0
        self.life_enemies_killed = 0
        self.life_enemies_outlived = 0
        self.life_useless_actions = 0
        self.previous_divers_onboard = 0
        self.previous_enemy_flags = [0, 0, 0, 0]
        self.previous_diver_lanes = [0, 0, 0, 0]
        self.previous_score = 0
        
    def reset_round_stats(self):
        self.round_total_score = 0
        self.round_total_reward = 0
        self.round_total_steps = 0
        self.round_lives_used = 0
        self.round_divers_picked_up = 0
        self.round_enemies_killed = 0
        self.round_enemies_outlived = 0
        self.round_useless_actions = 0
        self.round_action_counts = [0] * self.n_actions if self.n_actions else []
        self.round_episodes = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        #log previous life summary if this isn't the first reset
        if self.life_length > 0:
            self._log_life_summary()
        
        #get current state before reset to detect round end
        if self.previous_ram is not None:
            prev_lives = self.previous_ram[59]
        else:
            prev_lives = None
            
        self.reset_life_stats()
        obs, info = self.env.reset(seed=seed, options=options)
        
        #initialize RAM tracking
        self.previous_ram = self.env.unwrapped.ale.getRAM().copy()
        self.previous_divers_onboard = self.previous_ram[62]
        self.previous_enemy_flags = [self.previous_ram[i] for i in range(40, 44)]
        self.previous_diver_lanes = [self.previous_ram[i] for i in range(113, 117)]
        self.previous_score = int(f"{self.previous_ram[56]:02x}{self.previous_ram[57]:02x}{self.previous_ram[58]:02x}")
        
        #track starting lives from RAM (more reliable than info dict)
        self.starting_lives = self.previous_ram[59]
        self.current_lives = self.starting_lives
        
        #detect round end: lives went from low (0-1) back to starting_lives (new game)
        if prev_lives is not None and prev_lives == 0 and self.starting_lives == 3:
            self._log_round_summary()
            self.reset_round_stats()
        
        self.previous_lives_on_reset = self.starting_lives
        
        return self._modify_observation(obs), info
        
    def step(self, action):
        action = self._modify_action(action)
        
        if self.n_actions:
            self.life_action_counts[action] += 1
            
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
        
        #store player position before action
        prev_player_x = self.previous_ram[70] if self.previous_ram is not None else 0
        prev_player_y = self.previous_ram[97] if self.previous_ram is not None else 0
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        obs = self._modify_observation(obs)
        
        #get current RAM state
        ram = self.env.unwrapped.ale.getRAM()
        
        #decode game state
        score = int(f"{ram[56]:02x}{ram[57]:02x}{ram[58]:02x}")
        divers_onboard = ram[62]
        player_x = ram[70]
        player_y = ram[97]
        oxygen = ram[102]
        death_timer = ram[105]
        lives = ram[59]
        
        #detect useless action
        is_useless = self._detect_useless_action(action, prev_player_x, prev_player_y, player_x, player_y)
        if is_useless:
            self.life_useless_actions += 1
        
        #track life changes using RAM
        life_lost = 0
        life_gained = 0
        
        if lives != self.current_lives:
            if lives < self.current_lives:
                life_lost = self.current_lives - lives
                self.life_lives_lost += life_lost
            else:
                life_gained = lives - self.current_lives
            
        self.current_lives = lives
        
        #track divers picked up and missed
        divers_picked_up_now = 0
        divers_missed_now = 0
        
        if self.previous_ram is not None and divers_onboard > self.previous_divers_onboard:
            divers_picked_up_now = divers_onboard - self.previous_divers_onboard
            self.life_divers_picked_up += divers_picked_up_now
        
        #detect missed divers (lane went from 1 to 0 without pickup)
        if self.previous_ram is not None:
            current_diver_lanes = [ram[i] for i in range(113, 117)]
            for i in range(4):
                if self.previous_diver_lanes[i] == 1 and current_diver_lanes[i] == 0:
                    divers_missed_now += 1
            self.previous_diver_lanes = current_diver_lanes
        
        self.previous_divers_onboard = divers_onboard
        
        #track enemies outlived/killed
        if self.previous_ram is not None:
            enemy_flags = [ram[i] for i in range(40, 44)]
            for i in range(4):
                if self.previous_enemy_flags[i] != 0 and enemy_flags[i] == 0:
                    self.life_enemies_outlived += 1
                    #check if score increased = enemy was killed vs despawned
                    if score > self.previous_score:
                        self.life_enemies_killed += 1
            self.previous_enemy_flags = enemy_flags
        
        self.previous_score = score
        
        original_reward = reward
        reward = self._modify_reward(reward, obs, action, info, divers_picked_up_now, divers_missed_now, life_lost)
        
        # track life stats
        self.life_rewards.append(reward)
        self.life_length += 1
        self.total_steps += 1
        
        #get lane information
        enemy_lanes = self._get_active_lanes(ram[40:44])
        diver_lanes = self._get_active_lanes(ram[113:117], is_diver=True)
        
        #log per-step game state
        step_metrics = {
            "game/score": score,
            "game/lives": lives,
            "game/life_lost": life_lost,
            "game/life_gained": life_gained,
            "game/divers_onboard": divers_onboard,
            "game/player_x": player_x,
            "game/player_y": player_y,
            "game/enemy_lane_1": enemy_lanes[3],
            "game/enemy_lane_2": enemy_lanes[2],
            "game/enemy_lane_3": enemy_lanes[1],
            "game/enemy_lane_4": enemy_lanes[0],
            "game/diver_lane_1": diver_lanes[3],
            "game/diver_lane_2": diver_lanes[2],
            "game/diver_lane_3": diver_lanes[1],
            "game/diver_lane_4": diver_lanes[0],
            "game/death_status": 1 if death_timer != 0 else 0,
            "game/oxygen": oxygen,
            "game/step_reward": reward,
            "game/useless_action": 1 if is_useless else 0,
        }
        
        #add death_cause only when death starts
        if self.previous_ram is not None and self.previous_ram[105] == 0 and death_timer > 0:
            death_cause = self._detect_death_cause(self.previous_ram, ram, prev_player_x, prev_player_y)
            step_metrics["game/death_cause"] = death_cause
        
        if self.wandb_enabled:
            wandb.log(step_metrics, step=self.total_steps)
        
        #update previous RAM state
        self.previous_ram = ram.copy()
        
        return obs, reward, terminated, truncated, info
        
    def _log_life_summary(self):
        self.total_episodes += 1
        
        total_reward = sum(self.life_rewards)
        avg_reward_per_step = total_reward / max(1, self.life_length)
        life_score = self.previous_score  #final score at end of life
        
        # calc action distribution metrics
        action_metrics = {}
        if self.n_actions and self.life_length > 0:
            #individual action counts
            for i, count in enumerate(self.life_action_counts):
                action_name = self.action_meanings[i] if self.action_meanings else f"action_{i}"
                action_metrics[f"actions/{i}_{action_name}_count"] = count
                action_metrics[f"actions/{i}_{action_name}_pct"] = count / self.life_length
                
            #action entropy (measure of exploration)
            action_probs = np.array(self.life_action_counts) / self.life_length
            action_probs = action_probs[action_probs > 0]  #remove zeros
            action_entropy = -np.sum(action_probs * np.log(action_probs))
            action_metrics["actions/entropy"] = action_entropy
            
            #unique actions used
            unique_actions = np.count_nonzero(self.life_action_counts)
            action_metrics["actions/unique_count"] = unique_actions
        
        # life summary metrics
        summary = {
            # core life stats
            "life/total_reward": total_reward,
            "life/score": life_score,
            "life/length": self.life_length,
            "life/avg_reward_per_step": avg_reward_per_step,
            
            #behavior patterns
            "life/max_consecutive_actions": self.max_consecutive_actions,
            
            #survival metrics
            "life/steps_per_life": self.life_length, #/ max(1, self.life_lives_lost),
            
            # Game-specific metrics
            "life/divers_picked_up": self.life_divers_picked_up,
            "life/enemies_killed": self.life_enemies_killed,
            "life/enemies_outlived": self.life_enemies_outlived,
            "life/lives_difference": self.life_lives_lost,
            "life/useless_actions": self.life_useless_actions,
            
            **action_metrics,
        }
        
        if self.wandb_enabled:
            wandb.log(summary, step=self.total_steps)
        
        #accumulate life stats into round stats
        self.round_total_score = max(self.round_total_score, self.previous_score)  #track max score in round
        self.round_total_reward += total_reward
        self.round_total_steps += self.life_length
        self.round_lives_used += self.life_lives_lost
        self.round_divers_picked_up += self.life_divers_picked_up
        self.round_enemies_killed += self.life_enemies_killed
        self.round_enemies_outlived += self.life_enemies_outlived
        self.round_useless_actions += self.life_useless_actions
        self.round_episodes += 1
        
        #accumulate action counts
        for i, count in enumerate(self.life_action_counts):
            self.round_action_counts[i] += count
    
    def _log_round_summary(self):
        if self.round_episodes == 0:
            return
        
        #calc round action metrics
        round_action_metrics = {}
        if self.n_actions and self.round_total_steps > 0:
            #action entropy
            action_probs = np.array(self.round_action_counts) / self.round_total_steps
            action_probs = action_probs[action_probs > 0]
            action_entropy = -np.sum(action_probs * np.log(action_probs))
            round_action_metrics["round/action_entropy"] = action_entropy
            
            #unique actions used
            unique_actions = np.count_nonzero(self.round_action_counts)
            round_action_metrics["round/unique_actions_used"] = unique_actions
        
        round_summary = {
            "round/total_score": self.round_total_score,
            "round/total_reward": self.round_total_reward,
            "round/total_steps": self.round_total_steps,
            "round/lives_used": self.round_lives_used,
            "round/avg_steps_per_life": self.round_total_steps / max(1, self.round_lives_used),
            "round/total_divers_picked_up": self.round_divers_picked_up,
            "round/total_enemies_killed": self.round_enemies_killed,
            "round/total_enemies_outlived": self.round_enemies_outlived,
            "round/total_useless_actions": self.round_useless_actions,
            "round/episodes_in_round": self.round_episodes,
            **round_action_metrics,
        }
        
        if self.wandb_enabled:
            wandb.log(round_summary, step=self.total_steps)

    def _detect_useless_action(self, action, prev_x, prev_y, curr_x, curr_y):
        if action not in self.movement_actions:
            return False
        
        #action_name = self.action_meanings[action] if self.action_meanings else self.movement_actions.get(action, '')
        
        #check if position changed
        moved = (prev_x != curr_x) or (prev_y != curr_y)
        
        #movement action that didn't change position = useless
        if action in [2, 3, 4, 5, 6, 7, 8, 9]:  #UP, RIGHT, LEFT, DOWN, diagonals
            return not moved
        
        return False
    
    def _detect_death_cause(self, ram_prev, ram_current, player_x, player_y):
        #oxygen depletion - check both current=0 and previous<=5 to handle frame skipping
        if ram_current[102] == 0 or ram_prev[102] <= 5:
            return "OXYGEN_DEPLETION"
        else:
            #TODO: could be more specific by checking if enemy or torpedo deswpawned at the same time
            return "ENEMY_HIT_COLLISION"
    
    def _get_active_lanes(self, flags, is_diver=False):
        if is_diver:
            #diver lanes: value == 1 means active
            return tuple(1 if f == 1 else 0 for f in flags)
        else:
            #enemy lanes: value != 0 means active
            return tuple(1 if f != 0 else 0 for f in flags)
        
    def _modify_observation(self, obs):
        
        return obs
    
    def _modify_action(self, action):
        
        return action
    
    def _modify_reward(self, reward, obs, action, info, divers_picked_up=0, divers_missed=0, life_lost=0):
        #get current RAM state for additional checks
        ram = self.env.unwrapped.ale.getRAM()
        divers_onboard = ram[62]
        player_y = ram[97]
        
        #get previous player_y
        prev_player_y = self.previous_ram[97] if self.previous_ram is not None else 0
        
        #add +20 for each diver picked up (lane goes 1->255, divers_onboard increases)
        reward += divers_picked_up * 50
        
        #add -10 for each diver missed (lane goes 1->0)
        reward -= divers_missed * 10
        
        #penalty for surfacing with less than 6 divers (but not at life start, repeated surface frames, or death respawn)
        if player_y == 13 and self.life_length >= 40 and prev_player_y != 13 and life_lost == 0:  #just arrived at surface
            missing_divers = int(6 - divers_onboard)
            if missing_divers > 0:
                reward += missing_divers * (-2)
        
        #rewards/penalties based on actions when having 6 divers
        if divers_onboard == 6 and player_y != 13:
            #upward actions: UP(2), UPFIRE(10), UPRIGHT(6), UPRIGHTFIRE(14), UPLEFT(7), UPLEFTFIRE(15)
            upward_actions = [2, 10, 6, 14, 7, 15]
            #downward actions: DOWN(5), DOWNFIRE(13), DOWNRIGHT(8), DOWNRIGHTFIRE(16), DOWNLEFT(9), DOWNLEFTFIRE(17)
            downward_actions = [5, 13, 8, 16, 9, 17]
            
            if action in upward_actions:
                reward += 1.5  #encourage going up with 6 divers
            elif action in downward_actions:
                reward -= 1.5  #discourage going down with 6 divers
        
        return reward
    
    
