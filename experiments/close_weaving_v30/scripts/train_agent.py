"""Continue the supplied PPO checkpoint on the separate development training bank.

No validation or demo cases are sampled. No predictive/expert actions are mixed
into the on-policy rollout buffer. Outputs never overwrite the bundled model.
"""
import _bootstrap
import argparse
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import CaseBankEnv, load_agent, load_json, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/training.yaml')
    parser.add_argument('--steps', type=int, help='Overrides the short default budget')
    parser.add_argument('--bank', type=Path, default=ROOT/'data/train_cases.json')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    steps = spec['steps'] if args.steps is None else args.steps
    if steps <= 0 or steps % spec['rollout_steps']:
        parser.error('Steps must be a positive multiple of rollout_steps (64 by default)')
    if spec['rollout_steps'] % spec['batch_size']:
        parser.error('batch_size must divide rollout_steps')
    output = args.output or ROOT/'outputs'/f'ppo_{time.time_ns()}'
    if output.exists():
        parser.error('Refusing to overwrite an existing run directory')
    output.mkdir(parents=True)
    split = load_json(ROOT/'data/split.json')
    train = load_json(args.bank)
    training_ids = {c['scene']['seed'] for c in train}
    if not train or len(training_ids)!=len(train):
        parser.error('The training bank must be nonempty with unique seeds')
    if training_ids & (set(split['validation']) | set(split['demo'])):
        parser.error('Training cases overlap the frozen validation/demo split')
    torch.set_num_threads(1)
    original = ROOT/'models/current_ppo.zip'
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    env = DummyVecEnv([lambda: CaseBankEnv(train, reward_scale=spec['reward_scale'])])
    try:
        env.seed(spec['seed'])
        model = load_agent(env=env, seed=spec['seed'], n_steps=spec['rollout_steps'],
            batch_size=spec['batch_size'], n_epochs=spec['n_epochs'], learning_rate=spec['learning_rate'],
            clip_range=spec['clip_range'], target_kl=spec['target_kl'], gamma=spec['gamma'],
            gae_lambda=spec['gae_lambda'], ent_coef=spec['ent_coef'], vf_coef=spec['vf_coef'],
            max_grad_norm=spec['max_grad_norm'], tensorboard_log=None)
        model.policy.configure_learning(train_critic=True)
        model.set_logger(configure(str(output), ['stdout', 'csv']))
        # Checkpoints can retain rolling statistics from older experiments.
        # Start fresh monitoring buffers while preserving weights/optimizer and
        # the cumulative timestep counter used for continuation.
        model.ep_info_buffer = None
        model.ep_success_buffer = None
        model._episode_num = 0
        old_steps, old_updates = model.num_timesteps, model._n_updates
        # SB3 stores its sampled actions and their ORIGINAL log probabilities.
        # The environment's joint-bound projection is part of its dynamics.
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        model.save(output/'model.zip')
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        save_json(output/'training_summary.json', {
            'additional_steps': model.num_timesteps-old_steps,
            'optimizer_epoch_updates': model._n_updates-old_updates,
            'total_checkpoint_steps': model.num_timesteps,
            'settings': spec, 'source_model_sha256': digest,
            'source_model_unchanged': True, 'validation_used_for_training': False,
            'convergence_claim': False})
        print(f'Saved training run: {output.resolve()}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
