"""Prepare BC-to-PPO static tracking; only --train permits optimization."""
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
from wall_panel_rl.rl.waypoint_ppo import WaypointPPOEnv, load_for_finetuning, TrackerValidation
from wall_panel_rl.utils.config import load_config


def worker(config, settings, rank, output):
    torch.set_num_threads(1)
    return Monitor(WaypointPPOEnv(config,seed=settings['seed']+rank,
        margin=settings['joint_margin'],progress_weight=settings['progress_weight'],
        train_seed_range=settings['train_seed_range']),str(output/f'worker_{rank}.monitor.csv'),
        info_keywords=('termination_reason',))


def check_settings(s):
    for k in ('workers','n_steps','batch_size','n_epochs','additional_timesteps','eval_every'):
        if type(s[k]) is not int or s[k]<=0:raise ValueError(f'Invalid {k}')
    batch=s['workers']*s['n_steps']
    if s['batch_size']<2 or batch%s['batch_size']:raise ValueError('Invalid minibatch size')
    if s['additional_timesteps']%batch or s['eval_every']%batch:raise ValueError('Budgets must align to rollouts')
    for k in ('learning_rate','clip_range','target_kl','action_std','joint_margin'):
        if not np.isfinite(s[k]) or s[k]<=0:raise ValueError(f'Invalid {k}')
    if not np.isfinite(s['progress_weight']) or s['progress_weight']<0:raise ValueError('Invalid shaping weight')
    lo,hi=s['train_seed_range']
    if type(lo) is not int or type(hi) is not int or lo<0 or lo>=hi:raise ValueError('Invalid seed range')
    if lo<=211099 and hi>211000:raise ValueError('Independent test seeds must remain reserved')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/waypoint_ppo_pilot.yaml')
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    settings=yaml.safe_load(Path(args.config).read_text());check_settings(settings)
    release=Path(settings['release']);output=Path(settings['output'])
    config=load_config(release/'config.yaml')
    if config['obstacles'].get('dynamic',{}).get('enabled'):raise ValueError('Static config required')
    manifest=json.loads((release/'evaluation/manifest.json').read_text())
    lo,hi=settings['train_seed_range']
    if any(lo<=case['seed']<hi for case in manifest):raise ValueError('Train/validation seed overlap')
    if not all(any(c['suite']==s for c in manifest) for s in ('alignment_far','alignment_near')):
        raise ValueError('Both alignment validation suites are required')
    source=release/'tracker_bc.zip'
    model=load_for_finetuning(source,settings)
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):raise ValueError('Model shape mismatch')
    probe=WaypointPPOEnv(config,margin=settings['joint_margin'])
    try:
        if not (np.allclose(model.policy.joint_lower,probe.env.lower) and
                np.allclose(model.policy.joint_upper,probe.env.upper)):
            raise ValueError('Model and robot joint limits differ')
    finally:probe.close()
    if not np.isclose(model.policy.joint_margin,settings['joint_margin']):raise ValueError('Margin mismatch')
    if not np.isclose(model.policy.joint_increment,np.deg2rad(config['controller']['max_joint_increment_deg'])):
        raise ValueError('Action increment mismatch')
    if output.exists():raise FileExistsError(output)
    plan={'mode':'train' if args.train else 'check_only','settings':settings,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_steps':model.num_timesteps,'validation_cases':len(manifest),
        'trainable':{n:p.numel() for n,p in model.policy.named_parameters() if p.requires_grad},
        'action_processing':'raw Gaussian -> SB3 box clipping -> environment joint-margin projection',
        'training_actor':'14 gain/bias coefficients; velocity/std frozen; critic enabled',
        'validation':'shared-path ordinary/BC/PPO; post-update and final saved reload',
        'independent_test_executed':False,'optimizer_state_entries':len(model.policy.optimizer.state),
        'effective_ppo':{'learning_rate':model.learning_rate,'clip_range':model.clip_range(1),
            'target_kl':model.target_kl,'n_epochs':model.n_epochs,'batch_size':model.batch_size,
            'n_steps':model.n_steps,'std':model.policy.log_std.detach().exp().tolist()}}
    print(json.dumps(plan,indent=2),flush=True)
    if not args.train:return
    output.mkdir(parents=True,exist_ok=False)
    (output/'plan.json').write_text(json.dumps(plan,indent=2))
    (output/'environment.yaml').write_text(yaml.safe_dump(config))
    (output/'settings.yaml').write_text(yaml.safe_dump(settings))
    (output/'validation_manifest.json').write_text(json.dumps(manifest,indent=2))
    baseline=PPO.load(source,device='cpu')
    env=SubprocVecEnv([functools.partial(worker,config,settings,i,output) for i in range(settings['workers'])],start_method='spawn')
    start=time.time();source_steps=model.num_timesteps
    try:
        model.set_env(env);model.tensorboard_log=str(output/'tensorboard')
        model.learn(total_timesteps=settings['additional_timesteps'],reset_num_timesteps=False,
            callback=TrackerValidation(output,baseline,manifest,release,source_steps,
                settings['eval_every'],settings['progress_weight']))
        assert model.num_timesteps==source_steps+settings['additional_timesteps']
        (output/'training.json').write_text(json.dumps({'source_steps':source_steps,
            'additional_steps':model.num_timesteps-source_steps,'elapsed_seconds':time.time()-start}))
    finally:env.close()


if __name__=='__main__':main()
