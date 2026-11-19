"""Check a mixed-curriculum fine-tuning plan; train only with explicit --train.

This entry point never launches formal test evaluation or subsequent long runs.
"""
import _bootstrap
import argparse
import functools
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.mixed_curriculum import MixedCurriculumEnv, PostUpdateValidation, TASKS
from wall_panel_rl.utils.config import load_config


def check_settings(study):
    for key in ('workers', 'n_steps', 'batch_size', 'additional_timesteps', 'eval_every'):
        if type(study[key]) is not int or study[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    rollout_size = study['workers']*study['n_steps']
    if study['additional_timesteps'] % rollout_size or study['eval_every'] % rollout_size:
        raise ValueError('Budget and evaluation interval must be multiples of workers*n_steps')
    if rollout_size % study['batch_size'] or study['batch_size'] < 2:
        raise ValueError('Minibatches must divide rollout_size and contain at least two samples')
    mixture = study['mixture']
    if set(mixture) != set(TASKS):
        raise ValueError(f'Mixture must contain exactly {TASKS}')
    values = np.array(list(mixture.values()), dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1):
        raise ValueError('Mixture probabilities must be finite, nonnegative and sum to one')
    low, high = study['alignment_remaining_fraction']
    if not 0 < low <= high <= 1 or not 0 < study['validation_alignment_remaining_fraction'] <= 1:
        raise ValueError('Alignment fractions must be in (0,1]')


def make_worker(config, scenes, study, rank, filename):
    torch.set_num_threads(1)
    env = MixedCurriculumEnv(config, scenes, study['mixture'],
                             study['alignment_remaining_fraction'], study['seed']+rank*1000)
    return Monitor(env, filename, info_keywords=('curriculum_task',))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ppo_mixed_pilot.yaml')
    parser.add_argument('--condition', choices=['clean', 'noise'], default='clean')
    parser.add_argument('--train', action='store_true', help='Actually start the configured short run')
    args = parser.parse_args()
    torch.set_num_threads(1)
    study = yaml.safe_load(Path(args.config).read_text())
    check_settings(study)
    config = load_config(study['environment_configs'][args.condition])
    source = Path(study['source_models'][args.condition])
    scenes = json.loads(Path(study['train_scenes']).read_text())
    validation = json.loads(Path(study['validation_scenes']).read_text())
    train_seeds = {s['seed'] for s in scenes}
    val_seeds = {s['seed'] for s in validation}
    if not scenes or not validation or train_seeds & val_seeds:
        raise ValueError('Nonempty, disjoint training and validation banks are required')
    if len(train_seeds) != len(scenes) or len(val_seeds) != len(validation):
        raise ValueError('Duplicate scene seeds')
    config['ppo'].update(n_steps=study['n_steps'], batch_size=study['batch_size'],
                         total_timesteps=study['additional_timesteps'],
                         eval_freq=study['eval_every'], eval_episodes=len(validation))
    model = PPO.load(source, device='cpu', seed=study['seed'], n_envs=study['workers'],
                     n_steps=study['n_steps'], batch_size=study['batch_size'])
    env = WallPanelEnv(config)
    try:
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError('Source model and environment spaces differ')
    finally:
        env.close()
    source_steps = model.num_timesteps
    directory = Path(study['output_root'])/f'ppo_{args.condition}'
    if directory.exists():
        raise FileExistsError(f'Refusing to overwrite {directory}; choose a new output_root')
    plan = {'mode': 'train' if args.train else 'check_only', 'condition': args.condition,
            'source': str(source), 'source_steps': source_steps,
            'additional_steps': study['additional_timesteps'],
            'target_total_steps': source_steps+study['additional_timesteps'],
            'mixture_per_episode': study['mixture'], 'output': str(directory),
            'validation_suites': ['dynamic','static','alignment_far','alignment_near'],
            'evaluation_timing': 'baseline, after updates, final saved model reloaded',
            'formal_test_evaluation': False}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.train:
        return
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'study.yaml').write_text(yaml.safe_dump(study))
    (directory/'environment.yaml').write_text(yaml.safe_dump(config))
    files = [source, Path(study['train_scenes']), Path(study['validation_scenes'])]
    plan['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    plan['ppo'] = {k: getattr(model, k) for k in ('learning_rate','n_steps','batch_size',
                    'n_epochs','gamma','gae_lambda','ent_coef')}
    plan['optimizer_state_restored'] = True
    (directory/'plan.json').write_text(json.dumps(plan, indent=2))
    started = time.time()
    vec = SubprocVecEnv([functools.partial(make_worker, config, scenes, study, rank,
                       str(directory/f'mixed_{rank}.monitor.csv'))
                       for rank in range(study['workers'])], start_method='spawn')
    try:
        model.set_env(vec)
        model.tensorboard_log = str(directory/'tensorboard')
        callback = PostUpdateValidation(directory, config, validation, source_steps,
                     study['eval_every'], study['validation_alignment_remaining_fraction'])
        model.learn(total_timesteps=study['additional_timesteps'], reset_num_timesteps=False,
                    callback=callback, tb_log_name='mixed')
        if model.num_timesteps != plan['target_total_steps']:
            raise RuntimeError('Additional training budget mismatch')
        (directory/'training.json').write_text(json.dumps({
            'status': 'completed', 'source_steps': source_steps,
            'additional_steps': model.num_timesteps-source_steps,
            'total_steps': model.num_timesteps, 'elapsed_seconds': time.time()-started}, indent=2))
    finally:
        vec.close()


if __name__ == '__main__':
    main()
