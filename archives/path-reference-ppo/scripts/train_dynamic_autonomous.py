"""Autonomous dynamic avoidance pilot. Default checks only; --train prepares and learns."""
import _bootstrap
import argparse
import copy
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
from wall_panel_rl.planning.rrt_connect import RRTConnect,interpolate
from wall_panel_rl.rl.autonomous_dynamic_ppo import (
    initialize_from_tracking,AutonomousDynamicEnv,DynamicValidation,scenario_at_strength)
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import capture_scenario


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def prepare(config,s):
    directory=Path(s['data'])
    if (directory/'ready.json').exists():return
    directory.mkdir(parents=True,exist_ok=False)
    env=WallPanelEnv(config)
    failed=[]
    try:
        for split,count,start in [('train',s['train_scenes'],s['train_start_seed']),
                                  ('validation_full',s['validation_scenes'],s['validation_start_seed'])]:
            scenes=[];seed=start
            while len(scenes)<count and seed<start+1000:
                try:
                    env.reset(seed=seed);scenes.append(capture_scenario(env,seed))
                except RuntimeError as e:failed.append({'split':split,'seed':seed,'error':str(e)})
                seed+=1
                dump(directory/f'{split}.json',scenes);dump(directory/'generation_failures.json',failed)
                if len(scenes)%10==0:print('prepared',split,len(scenes),flush=True)
            if len(scenes)!=count:raise RuntimeError(f'Could not prepare {split}')
        full=json.loads((directory/'validation_full.json').read_text())
        cases=[]
        for scene in full:
            cases.extend([
                {'suite':'dynamic_full','task':'dynamic','strength':[.4,.4],'scene':scene},
                {'suite':'dynamic_easy','task':'dynamic','strength':[.1,.05],
                 'scene':scenario_at_strength(scene,.1,.05)},
                {'suite':'static','task':'static','strength':[.4,.4],'scene':scene}])
        release=Path(s['release'])
        manifest=json.loads((release/'evaluation/manifest.json').read_text())
        for case in manifest:
            if case['suite'] not in ('alignment_far','alignment_near'):continue
            scene=json.loads((release/case['scene']).read_text())
            env.reset(seed=case['seed'],options={'scenario':scene})
            q=env.joint_state()[0]
            planner=RRTConnect(env.lower,env.upper,env.collision.is_configuration_valid,
                env.collision.edge_valid,config['rrt'],case['seed'])
            path=planner.plan(q,env.goal_q)
            if path is None:raise RuntimeError('Could not prepare alignment reference')
            dense=[path[0]]
            for a,b in zip(path[:-1],path[1:]):dense.extend(interpolate(a,b,.025)[1:])
            scene['reference_joint_path']=[v.tolist() for v in dense]
            cases.append({'suite':case['suite'],'task':'alignment','strength':[.4,.4],'scene':scene})
        dump(directory/'validation.json',cases)
        dump(directory/'ready.json',{'train_scenes':s['train_scenes'],'validation_cases':len(cases),
            'failed_generation':len(failed),'reference':'obstacle-free; no execution planner',
            'training_phase_randomization':True,'full_envelope_endpoint_placement':True})
    finally:env.close()


def worker(config,scenes,s,rank,output):
    torch.set_num_threads(1)
    return Monitor(AutonomousDynamicEnv(config,scenes,s,rank),str(output/f'worker_{rank}.monitor.csv'),
        info_keywords=('termination_reason','curriculum_task','amplitude','max_speed'))


def check(s):
    for key in ('additional_timesteps','workers','n_steps','batch_size','eval_every','train_scenes','validation_scenes'):
        if type(s[key]) is not int or s[key]<=0:raise ValueError(key)
    batch=s['workers']*s['n_steps']
    if s['additional_timesteps']%batch or s['eval_every']%batch or batch%s['batch_size']:
        raise ValueError('Budget/evaluation/minibatch must align to rollouts')
    if set(s['mixture'])!={'alignment','static','dynamic'} or not np.isclose(sum(s['mixture'].values()),1):
        raise ValueError('Invalid episode mixture')
    if any(v<0 for v in s['mixture'].values()):raise ValueError('Negative episode probability')
    if s['residual_scale']<2:raise ValueError('Residual must be able to reverse saturated prior actions')
    if s['train_start_seed']+1000>s['validation_start_seed']:raise ValueError('Seed ranges overlap')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/dynamic_autonomous_pilot.yaml')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--train',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    s=yaml.safe_load(Path(args.config).read_text());check(s)
    config=load_config(s['environment'])
    assert config['observation']['obstacle_velocity'] and config['obstacles']['max_count']==5
    assert config['noise']=={'position_std':0.,'orientation_std_deg':0.}
    config['obstacles']['freeze_motion']=False
    config['obstacles']['dynamic'].update(enabled=True,placement='rrt_path',motion_mode='random_3d',amplitude=.4,max_speed=.4)
    source=Path(s['source']);output=Path(s['output'])
    if output.exists():raise FileExistsError(output)
    model=initialize_from_tracking(source,s)
    plan={'settings':s,'source_steps':model.num_timesteps,'additional_steps':s['additional_timesteps'],
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'prior_and_critic_transferred':True,'optimizer_reset_for_new_architecture':True,
        'online_avoidance_planner':False,'reference':'obstacle-free RRT only at scene generation',
        'residual_initial_output':0.,'actor_trainable':{n:p.numel() for n,p in model.policy.named_parameters()
            if p.requires_grad and not (n.startswith('value_net.') or n.startswith('mlp_extractor.value_net.'))},
        'difficulty':'first 20% easy; ramp to full at 80%; final 20% full',
        'mixture_unit':'episodes; alignment splits equally near/far',
        'validation':'same frozen scenarios and obstacle-free reference for all methods',
        'test_seeds_211000_211099_used':False,'mode':'train' if args.train else 'prepare' if args.prepare else 'check_only'}
    print(json.dumps(plan,indent=2),flush=True)
    if args.prepare or args.train:prepare(config,s)
    if not args.train:return
    scenes=json.loads((Path(s['data'])/'train.json').read_text())
    validation=json.loads((Path(s['data'])/'validation.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    dump(output/'plan.json',plan);(output/'settings.yaml').write_text(yaml.safe_dump(s))
    (output/'environment.yaml').write_text(yaml.safe_dump(config))
    dump(output/'data_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(s['data']).glob('*.json')})
    source_model=PPO.load(source,device='cpu')
    env=SubprocVecEnv([functools.partial(worker,config,scenes,s,i,output) for i in range(s['workers'])],start_method='spawn')
    start=time.time();source_steps=model.num_timesteps
    try:
        model.set_env(env);model.tensorboard_log=str(output/'tensorboard')
        model.learn(total_timesteps=s['additional_timesteps'],reset_num_timesteps=False,
            callback=DynamicValidation(output,source_model,config,s,validation,source_steps))
        assert model.num_timesteps==source_steps+s['additional_timesteps']
        dump(output/'training.json',{'source_steps':source_steps,'additional_steps':model.num_timesteps-source_steps,
             'total_steps':model.num_timesteps,'elapsed_seconds':time.time()-start})
    finally:env.close()


if __name__=='__main__':main()
