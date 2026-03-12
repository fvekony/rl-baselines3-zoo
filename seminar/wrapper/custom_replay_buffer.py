import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class BadExperienceReplayBuffer(ReplayBuffer):
    
    def __init__(self, *args, bad_threshold=-1.0, retention_factor=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.bad_threshold = bad_threshold
        self.retention_factor = retention_factor
        self.experience_priorities = np.ones(self.buffer_size, dtype=np.float32)
        
    def add(self, obs, next_obs, action, reward, done, infos):
        #mark bad experiences (negative rewards or deaths) with higher priority
        is_bad = reward < self.bad_threshold or done
        
        super().add(obs, next_obs, action, reward, done, infos)
        
        #assign priority based on how bad the experience was
        if is_bad:
            priority = self.retention_factor * abs(reward) if reward < 0 else self.retention_factor
            self.experience_priorities[self.pos - 1] = priority
        else:
            self.experience_priorities[self.pos - 1] = 1.0
    
    def sample(self, batch_size, env=None):
        #sample with probability proportional to priorities
        if self.full:
            valid_indices = np.arange(self.buffer_size)
        else:
            valid_indices = np.arange(self.pos)
        
        #normalize priorities to probabilities
        valid_priorities = self.experience_priorities[valid_indices]
        probs = valid_priorities / valid_priorities.sum()
        
        #sample indices based on priority
        batch_inds = np.random.choice(
            valid_indices,
            size=batch_size,
            replace=True,
            p=probs
        )
        
        return self._get_samples(batch_inds, env=env)
