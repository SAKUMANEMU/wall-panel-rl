"""Pure-policy rollouts from frozen four-box TRAIN layouts only."""
import copy
import json
from pathlib import Path
import time
import gymnasium as gym
import numpy as np


class FourPolicyTrainingEnv(gym.Wrapper):
    def __init__(self, env, cases, *, excluded=(), seed=61, reward_scale=.01, journal=None, sampling_probabilities=None):
        super().__init__(env)
        seeds = [c['scene']['seed'] for c in cases]
        if not cases or len(set(seeds)) != len(seeds) or set(seeds) & set(excluded):
            raise ValueError('Invalid train/held-out split')
        if any(len(c['scene']['obstacles']) != 4 or c['scene']['obstacle_motion'] for c in cases):
            raise ValueError('Expected four static training obstacles')
        if reward_scale <= 0: raise ValueError('Invalid reward scale')
        self.cases = copy.deepcopy(cases); self.rng = np.random.default_rng(seed)
        self.reward_scale = reward_scale; self.journal = Path(journal) if journal else None
        self.sampling_probabilities = None
        if sampling_probabilities is not None:
            probabilities = np.asarray(sampling_probabilities, dtype=float)
            if probabilities.shape != (len(cases),) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.):
                raise ValueError('Invalid training sampling probabilities')
            self.sampling_probabilities = probabilities.copy()
        self.records = []; self.started = time.perf_counter()

    def reset(self, *, seed=None, options=None):
        if options: raise ValueError('Training resets cannot override the frozen bank')
        if seed is not None: self.rng = np.random.default_rng(seed)
        index = self.rng.integers(len(self.cases)) if self.sampling_probabilities is None else self.rng.choice(len(self.cases), p=self.sampling_probabilities)
        self.case = self.cases[int(index)]; self.steps = 0; self.total = 0.
        obs, info = self.env.reset(options={'scenario': self.case['scene'], 'task': 'static', 'strength': self.case['strength']})
        info['episode_controller'] = 'ppo'; return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.steps += 1; self.total += reward
        info = dict(info, episode_controller='ppo', unscaled_reward=float(reward))
        if done or truncated:
            row = {'seed': self.case['scene']['seed'], 'reason': info['termination_reason'], 'steps': self.steps,
                   'raw_return': self.total, 'scaled_return': self.total*self.reward_scale, 'controller': 'ppo'}
            self.records.append(row)
            if self.journal:
                self.journal.parent.mkdir(parents=True, exist_ok=True)
                with self.journal.open('a') as f: f.write(json.dumps(row)+'\n')
            info['episode'] = {'r': self.total*self.reward_scale, 'l': self.steps, 't': time.perf_counter()-self.started}
        return obs, float(reward*self.reward_scale), done, truncated, info
