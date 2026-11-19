"""Episode-level mixture and post-update validation for alignment-policy fine-tuning."""
import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.metrics import rollout


TASKS = ('alignment', 'static', 'dynamic')


def alignment_scene(scene, remaining=None):
    """Remove obstacles; optionally choose a near-goal reference-path configuration."""
    scene = copy.deepcopy(scene)
    scene['obstacles'] = []
    scene['obstacle_motion'] = []
    if remaining is not None:
        path = np.asarray(scene['reference_joint_path'])
        if len(path) < 2 or not 0 < remaining <= 1:
            raise ValueError('Alignment needs a reference path and remaining fraction in (0,1]')
        index = (1-remaining)*(len(path)-1)
        left = min(int(index), len(path)-2)
        fraction = index-left
        # Interpolate along an already collision-checked edge, not start-to-goal.
        scene['initial_q'] = ((1-fraction)*path[left]+fraction*path[left+1]).tolist()
    return scene


class MixedCurriculumEnv(gym.Wrapper):
    def __init__(self, config, scenes, mixture, remaining_range, seed):
        super().__init__(WallPanelEnv(config))
        self.scenes = scenes
        self.probabilities = np.array([mixture[k] for k in TASKS], dtype=float)
        if np.any(self.probabilities < 0) or not np.isclose(self.probabilities.sum(), 1):
            raise ValueError('Mixture probabilities must be nonnegative and sum to one')
        self.remaining_range = remaining_range
        self.rng = np.random.default_rng(seed)
        self.task = None

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.task = str(self.rng.choice(TASKS, p=self.probabilities))
        scene = self.scenes[int(self.rng.integers(len(self.scenes)))]
        if self.task == 'alignment':
            scene = alignment_scene(scene, self.rng.uniform(*self.remaining_range))
        # Always reset the flag: static episodes must not freeze subsequent dynamic ones.
        self.env.config['obstacles']['freeze_motion'] = self.task == 'static'
        obs, info = self.env.reset(seed=scene['seed'], options={
            'scenario': scene, 'noise_seed': int(self.rng.integers(200000, 2000000000))})
        info['curriculum_task'] = self.task
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        info['curriculum_task'] = self.task
        return obs, reward, done, truncated, info


def summarize(rows):
    """Timeout errors are conditional on timeout, not averages over collisions."""
    if not rows:
        raise ValueError('Validation set is empty')
    timeouts = [r for r in rows if not any(r[k] for k in
                ('success', 'collision', 'joint_limit_violation'))]
    return {
        'episodes': len(rows),
        'success_rate': float(np.mean([r['success'] for r in rows])),
        'collision_rate': float(np.mean([r['collision'] for r in rows])),
        'joint_limit_rate': float(np.mean([r['joint_limit_violation'] for r in rows])),
        'timeout_rate': len(timeouts)/len(rows),
        'timeout_count': len(timeouts),
        'timeout_mean_position_error_m': float(np.mean([r['position_error_m'] for r in timeouts])) if timeouts else None,
        'timeout_mean_orientation_error_deg': float(np.mean([r['orientation_error_deg'] for r in timeouts])) if timeouts else None,
        'mean_reward': float(np.mean([r['cumulative_reward'] for r in rows])),
    }


def validate_policy(model, config, scenes, near_fraction):
    """Never reads test scenes; all suites use fixed validation seeds/noise streams."""
    summaries, episodes = {}, []
    for suite in ('dynamic', 'static', 'alignment_far', 'alignment_near'):
        c = copy.deepcopy(config)
        c['obstacles']['freeze_motion'] = suite == 'static'
        env = WallPanelEnv(c)
        rows = []
        try:
            for original in scenes:
                scene = original
                if suite.startswith('alignment'):
                    scene = alignment_scene(original, near_fraction if suite == 'alignment_near' else None)
                row, _ = rollout(env, 'ppo', scene['seed'], model, scenario=scene)
                # PPO has no planner; exclude its NaN planner-specific fields from JSON.
                row = {k: v for k, v in row.items() if k not in ('planning_success','ik_success')}
                row['suite'] = suite
                rows.append(row)
        finally:
            env.close()
        summaries[suite] = summarize(rows)
        episodes.extend(rows)
    return summaries, episodes


class PostUpdateValidation(BaseCallback):
    """Validate updated weights, plus independently reload the final saved artifact.

    SB3's rollout_end precedes the optimizer update. The next rollout_start follows
    it; training_end handles the final update, for which there is no next rollout.
    """
    def __init__(self, directory, config, scenes, source_steps, eval_every, near_fraction):
        super().__init__()
        self.directory = Path(directory)
        self.config, self.scenes = config, scenes
        self.source_steps, self.eval_every = source_steps, eval_every
        self.near_fraction = near_fraction
        self.last_evaluated = None
        self.best_score = (-1., -float('inf'))

    def _on_step(self):
        return True

    def _on_training_start(self):
        self.evaluate('baseline')

    def _on_rollout_start(self):
        added = self.model.num_timesteps-self.source_steps
        if added > 0 and added % self.eval_every == 0 and added != self.last_evaluated:
            self.evaluate('post_update')

    def _on_training_end(self):
        path = self.directory/'final_model.zip'
        self.model.save(path)
        saved = PPO.load(path, device='cpu')
        self.evaluate('final_saved_reload', saved)

    def evaluate(self, kind, model=None):
        model = self.model if model is None else model
        added = model.num_timesteps-self.source_steps
        path = self.directory/'checkpoints'/f'checkpoint_{added}.zip'
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        summary, rows = validate_policy(model, self.config, self.scenes, self.near_fraction)
        record = {'additional_steps': added, 'total_steps': model.num_timesteps,
                  'kind': kind, 'metrics': summary}
        with (self.directory/'validation.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False)+'\n')
        with (self.directory/'validation_episodes.jsonl').open('a') as f:
            for row in rows:
                f.write(json.dumps({'additional_steps': added, 'kind': kind, **row}, allow_nan=False)+'\n')
        # Preserve the earlier study's selection rule; report retention independently.
        score = (summary['dynamic']['success_rate'], summary['dynamic']['mean_reward'])
        if score > self.best_score:
            self.best_score = score
            model.save(self.directory/'best_model.zip')
            (self.directory/'selection.json').write_text(json.dumps(
                {'additional_steps': added, 'total_steps': model.num_timesteps,
                 'kind': kind, 'score': score, 'metrics': summary}, indent=2))
        (self.directory/'progress.json').write_text(json.dumps(record, indent=2))
        self.last_evaluated = added
        print(json.dumps(record), flush=True)
