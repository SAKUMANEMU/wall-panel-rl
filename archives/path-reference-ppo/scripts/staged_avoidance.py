"""Explicit stage workflow: check -> prepare -> collect -> imitate -> PPO -> assess.

No action flag performs checks only. No operation automatically launches another
stage, increases budget, or starts optimization without its own explicit flag.
"""
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
from train_dynamic_autonomous import prepare as prepare_full, dump, check as check_base
from wall_panel_rl.rl.avoidance_curriculum import (
    initialize_gated,stage_scene,StagedAvoidanceEnv,StagedValidation,assess_history,gate_result,
    INTERFACE_CONTRACT,ReferencePreservingPolicy)
from wall_panel_rl.rl.autonomous_dynamic_ppo import evaluate
from wall_panel_rl.rl.avoidance_imitation import collect_teacher,pretrain_actor
from wall_panel_rl.utils.config import load_config


def resolve(root,name):
    s=copy.deepcopy(root);stages=s['stages'];names=[v['name'] for v in stages]
    if name not in names:raise ValueError(f'Unknown stage: {name}')
    index=names.index(name);s['active_stage']=stages[index];s['stage_index']=index
    s['train_start_seed']=s['train_seed_base']+index*10000
    s['validation_start_seed']=s['validation_seed_base']+index*10000
    s['data']=str(Path(s['data_root'])/name)
    s['stage_output']=str(Path(s['output_root'])/name)
    s['output']=str(Path(s['stage_output'])/'ppo')
    s['source_model']=(s['source_first'] if index==0 else
        str(Path(s['output_root'])/stages[index-1]['name']/'ppo/best_model.zip'))
    return s


def check(s):
    check_base(s)
    if s.get('interface_contract')!=INTERFACE_CONTRACT:
        raise ValueError('Obsolete training contract; archived v2 is diagnostic-only')
    if s['terminal_radius']<=0:raise ValueError('Invalid terminal radius')
    recovery=s['reference_recovery']
    if recovery['radius']<=0 or recovery['min_distance']<0 or recovery['improvement']<=0:
        raise ValueError('Invalid reference recovery')
    if not 0<s['initial_prior_gate']<1 or not 0<s['action_std']<=.5:raise ValueError('Invalid gate/exploration')
    if s['clearance_distance']<=0 or s['clearance_weight']<0 or s['progress_weight']<0:
        raise ValueError('Invalid reward settings')
    if s['validation_scenes']<s['gate']['minimum_episodes']:raise ValueError('Too few validation cases for gate')
    if s['gate']['consecutive_checks']<2:raise ValueError('Require two updated-policy checks')
    if s['additional_timesteps']//s['eval_every']<s['gate']['consecutive_checks']:
        raise ValueError('Budget cannot satisfy promotion checks')
    for k in ('minimum_success_rate','maximum_collision_rate','minimum_alignment_success_rate'):
        if not 0<=s['gate'][k]<=1:raise ValueError(k)
    t=s['teacher']
    for k in ('alignment_episodes','avoidance_episodes','maximum_attempts','minimum_unique_scenes','updates','batch_size'):
        if type(t[k]) is not int or t[k]<=0:raise ValueError(k)
    if t['maximum_attempts']<t['alignment_episodes']+t['avoidance_episodes']:
        raise ValueError('Teacher attempts below quota')
    if t['learning_rate']<=0:raise ValueError('Invalid imitation learning rate')
    if t['batch_size']<2 or any(t[k]<0 for k in ('terminal_weight','gate_retention_weight','residual_weight')):
        raise ValueError('Invalid retention loss settings')
    bc=s['bc_gate']
    if (bc['minimum_episodes']<1 or bc['minimum_episodes']>s['validation_scenes'] or
        not 1<=bc['minimum_alignment_episodes']<=10 or bc['maximum_goal_action']<=0 or
        any(not 0<=bc[k]<=1 for k in ('minimum_success_rate','maximum_collision_rate','minimum_alignment_success_rate'))):
        raise ValueError('Invalid BC closed-loop thresholds')
    names=[];ranges=[]
    for i,stage in enumerate(s['stages']):
        if stage['name'] in names or stage['count'] not in (1,2,3,4,5) or stage['mode'] not in ('static','dynamic'):
            raise ValueError('Invalid stage')
        if stage['amplitude']<=0 or stage['speed']<=0:raise ValueError('Invalid motion bound')
        names.append(stage['name'])
        for base in (s['train_seed_base'],s['validation_seed_base']):ranges.append((base+i*10000,base+i*10000+1000))
    for i,(lo,hi) in enumerate(ranges):
        if lo<211100 and hi>211000:raise ValueError('Reserved independent seeds overlap')
        if any(lo<b and hi>a for a,b in ranges[i+1:]):raise ValueError('Stage seed ranges overlap')
    last=s['stages'][-1]
    if (last['count'],last['mode'],last['amplitude'],last['speed'])!=(5,'dynamic',.4,.4):
        raise ValueError('Final full-difficulty task must remain five dynamic obstacles at .4/.4')


def require_previous(s):
    if s['stage_index']==0:return
    previous=s['stages'][s['stage_index']-1]['name']
    folder=Path(s['output_root'])/previous/'ppo'
    assessment=json.loads((folder/'assessment.json').read_text())
    model=folder/'best_model.zip'
    if not assessment['passed'] or assessment['stage']!=previous or assessment['thresholds']!=s['gate']:
        raise ValueError('Previous stage has not passed current promotion criteria')
    if hashlib.sha256(model.read_bytes()).hexdigest()!=assessment['selected_model_sha256']:
        raise ValueError('Previous selected model changed after assessment')


def prepare_stage(config,s):
    prepare_full(config,s)
    data=Path(s['data']);meta_path=data/'stage_metadata.json'
    if meta_path.exists():
        cached=json.loads(meta_path.read_text())
        if (cached['stage']!=s['active_stage'] or cached['environment_config']!=config or
            cached.get('interface_contract')!=s['interface_contract'] or
            cached.get('reference_recovery')!=s['reference_recovery']):
            raise ValueError('Data stage/physical configuration changed')
        return
    full=json.loads((data/'validation_full.json').read_text())
    cases=[];stage=s['active_stage']
    for scene in full:
        cases.append({'suite':'stage_target','task':'static' if stage['mode']=='static' else 'dynamic',
            'strength':[stage['amplitude'],stage['speed']],'scene':stage_scene(scene,stage)})
    prior=json.loads((data/'validation.json').read_text())
    cases.extend(c for c in prior if c['suite'].startswith('alignment'))
    if stage['count']<5 or stage['speed']<.4:
        for scene in full[:10]:cases.append({'suite':'five_full_diagnostic','task':'dynamic',
            'strength':[.4,.4],'scene':scene})
    dump(data/'validation_staged.json',cases)
    dump(meta_path,{'stage':stage,'cases':len(cases),'target_cases':len(full),
         'seed_ranges_disjoint':True,'full_diagnostic_used_for_promotion':False,
         'interface_contract':s['interface_contract'],'reference_recovery':s['reference_recovery'],
         'environment_config':config,'data_hashes':{name:hashlib.sha256((data/name).read_bytes()).hexdigest()
           for name in ('train.json','validation_full.json','validation_staged.json')}})


def make_worker(config,scenes,s,rank,output):
    torch.set_num_threads(1)
    return Monitor(StagedAvoidanceEnv(config,scenes,s,rank),str(output/f'worker_{rank}.monitor.csv'),
        info_keywords=('termination_reason','curriculum_task','amplitude','max_speed','reward_obstacle_clearance'))


def score_bc_model(model,source,config,s,cases):
    """Shared inference-only score for BC checkpoints and the PPO release gate."""
    metrics,rows=evaluate(model,source,config,s,cases,env_factory=StagedAvoidanceEnv)
    result=gate_result(metrics['dynamic_ppo'],s['bc_gate'])
    current=metrics['dynamic_ppo']
    result['checks']['alignment_collision_free']=all(current[k]['collision']==0 and current[k]['joint_limit']==0
        for k in ('alignment_far','alignment_near'))
    case=next(c for c in cases if c['suite']=='alignment_far')
    env=StagedAvoidanceEnv(config,[],s)
    goal_actions=[]
    try:
        for present in (False,True):
            scene=copy.deepcopy(case['scene']);goal=config['robot']['goal_q']
            scene.update(initial_q=goal,reference_joint_path=[goal,goal],obstacle_motion=[],
                obstacles=[{'center':[5.,5.,5.],'dimensions':[.1,.1,.1]}] if present else [])
            observation,info=env.reset(options={'scenario':scene,'task':'static','strength':[.05,.02]})
            action=model.predict(observation,deterministic=True)[0]
            goal_actions.append({'obstacles_present':present,'initial_success':bool(info['success']),
                                 'max_abs_action':float(np.max(np.abs(action)))})
    finally:env.close()
    result['checks']['goal_convergence']=all(v['initial_success'] and
        v['max_abs_action']<=s['bc_gate']['maximum_goal_action'] for v in goal_actions)
    result['passed']=all(result['checks'].values())
    result.update(interface_contract=INTERFACE_CONTRACT,training_performed=False,
        stage=s['active_stage']['name'],thresholds=s['bc_gate'],goal_probes=goal_actions,metrics=metrics)
    return result,rows


def validate_bc(config,s,data,bc_dir):
    """Fresh release check before any PPO output/worker/learn is created."""
    meta=json.loads((bc_dir/'metadata.json').read_text())
    if meta.get('interface_contract')!=INTERFACE_CONTRACT or meta['settings']!=s:
        raise ValueError('BC contract/settings mismatch; v2 BC cannot initialize v3 PPO')
    model=PPO.load(bc_dir/'model.zip',device='cpu')
    if not isinstance(model.policy,ReferencePreservingPolicy):raise ValueError('Wrong BC policy class')
    source=PPO.load(s['source_model'],device='cpu')
    if meta['source_sha256']!=hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest():
        raise ValueError('Source changed after BC')
    teacher=data/'teacher/teacher.npz'
    teacher_meta=json.loads((data/'teacher/metadata.json').read_text())
    if (teacher_meta.get('interface_contract')!=INTERFACE_CONTRACT or
        teacher_meta['environment_config']!=config):
        raise ValueError('Teacher/physical configuration changed')
    if meta['teacher_sha256']!=hashlib.sha256(teacher.read_bytes()).hexdigest():
        raise ValueError('Teacher changed after BC')
    if meta.get('data_reuse_contract') is not None:
        from wall_panel_rl.rl.avoidance_imitation import teacher_reuse_contract
        if teacher_reuse_contract(teacher_meta,s,config)!=meta['data_reuse_contract']:
            raise ValueError('Data reuse contract changed after imitation')
    cases=json.loads((data/'validation_staged.json').read_text())
    result,rows=score_bc_model(model,source,config,s,cases)
    result.update(
        bc_model_sha256=hashlib.sha256((bc_dir/'model.zip').read_bytes()).hexdigest(),
        validation_sha256=hashlib.sha256((data/'validation_staged.json').read_bytes()).hexdigest())
    dump(bc_dir/'closed_loop_gate.json',result);dump(bc_dir/'closed_loop_episodes.json',rows)
    print('BC closed-loop gate',json.dumps({k:result[k] for k in ('passed','checks','rates','alignment')}),flush=True)
    return result


def train(config,s,data,bc_dir,output):
    require_previous(s)
    if output.exists():raise FileExistsError(output)
    meta=json.loads((bc_dir/'metadata.json').read_text())
    teacher_dir=data/'teacher';teacher_meta=json.loads((teacher_dir/'metadata.json').read_text())
    if meta['settings']!=s:raise ValueError('Imitation settings changed; use a new explicit experiment')
    if meta['source_sha256']!=hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest():
        raise ValueError('Source changed after imitation')
    if teacher_meta['environment_config']!=config:raise ValueError('Physical environment changed')
    if teacher_meta.get('interface_contract')!=INTERFACE_CONTRACT:raise ValueError('Obsolete teacher contract')
    if meta['teacher_sha256']!=hashlib.sha256((teacher_dir/'teacher.npz').read_bytes()).hexdigest():
        raise ValueError('Teacher data changed')
    gate=validate_bc(config,s,data,bc_dir)
    if not gate['passed']:
        raise RuntimeError(f"BC closed-loop gate failed; PPO not started. See {bc_dir/'closed_loop_gate.json'}")
    model=PPO.load(bc_dir/'model.zip',device='cpu',n_envs=s['workers'],n_steps=s['n_steps'],
        batch_size=s['batch_size'],learning_rate=s['learning_rate'],n_epochs=s['n_epochs'],
        clip_range=s['clip_range'],target_kl=s['target_kl'],seed=s['seed'],ent_coef=0.)
    with torch.no_grad():model.policy.log_std.fill_(np.log(s['action_std']))
    source=PPO.load(s['source_model'],device='cpu')
    scenes=json.loads((data/'train.json').read_text());cases=json.loads((data/'validation_staged.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    (output/'settings.yaml').write_text(yaml.safe_dump(s));(output/'environment.yaml').write_text(yaml.safe_dump(config))
    dump(output/'input_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(s['source_model']),bc_dir/'model.zip',data/'train.json',data/'validation_staged.json',teacher_dir/'teacher.npz']})
    env=SubprocVecEnv([functools.partial(make_worker,config,scenes,s,i,output) for i in range(s['workers'])],start_method='spawn')
    started=time.time();source_steps=model.num_timesteps
    try:
        model.set_env(env);model.tensorboard_log=str(output/'tensorboard')
        model.learn(total_timesteps=s['additional_timesteps'],reset_num_timesteps=False,
            callback=StagedValidation(output,source,config,s,cases,source_steps))
        assert model.num_timesteps-source_steps==s['additional_timesteps']
        dump(output/'training.json',{'source_steps':source_steps,'additional_steps':model.num_timesteps-source_steps,
            'elapsed_seconds':time.time()-started,'automatic_stage_advance':False})
        assess(s,output)
    finally:env.close()


def assess(s,output):
    completed=json.loads((output/'training.json').read_text())
    if completed['additional_steps']!=s['additional_timesteps']:
        raise ValueError('Stage training budget incomplete')
    history=[json.loads(v) for v in (output/'validation.jsonl').read_text().splitlines()]
    result=assess_history(history,s['gate'])
    selected=json.loads((output/'selection.json').read_text())
    selected_gate=gate_result(selected['metrics']['dynamic_ppo'],s['gate'])
    result['selected_model_gate']=selected_gate
    result['passed']=result['passed'] and selected_gate['passed']
    result.update(stage=s['active_stage']['name'],
        selected_model_sha256=hashlib.sha256((output/'best_model.zip').read_bytes()).hexdigest())
    dump(output/'assessment.json',result);print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/staged_avoidance.yaml')
    parser.add_argument('--stage',default='one_static')
    group=parser.add_mutually_exclusive_group()
    for flag in ('prepare','collect-teacher','pretrain','validate-bc','train','assess'):group.add_argument('--'+flag,action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    s=resolve(yaml.safe_load(Path(args.config).read_text()),args.stage);check(s)
    config=load_config(s['environment'])
    if config['observation'].get('obstacle_velocity')!=True or config['obstacles']['max_count']!=5:
        raise ValueError('Expected 70 base observations with five slots')
    if config['noise']!={'position_std':0.,'orientation_std_deg':0.}:raise ValueError('Noise-free study required')
    config['obstacles']['freeze_motion']=False
    config['obstacles']['dynamic'].update(enabled=True,placement='rrt_path',motion_mode='random_3d',amplitude=.4,max_speed=.4)
    data=Path(s['data']);bc_dir=Path(s['stage_output'])/'bc';output=Path(s['output'])
    action=next((flag for flag in ('prepare','collect_teacher','pretrain','validate_bc','train','assess') if getattr(args,flag)),None)
    print(json.dumps({'mode':action or 'check_only','stage':s['active_stage'],'source':s['source_model'],
        'source_exists':Path(s['source_model']).exists(),'data':str(data),'bc_output':str(bc_dir),'ppo_output':str(output),
        'ppo_additional_steps':s['additional_timesteps'],'action_std':s['action_std'],'prior_gate_initial':s['initial_prior_gate'],
        'waypoint_reward':s['progress_weight'],'clearance_distance':s['clearance_distance'],
        'clearance_weight':s['clearance_weight'],'gate':s['gate'],'automatic_stage_advance':False},indent=2))
    if action is None:
        if Path(s['source_model']).exists():
            model=initialize_gated(s['source_model'],s)
            assert model.observation_space.shape==(77,) and model.action_space.shape==(7,)
            print('Model/space checks passed. No files, rollouts or optimization created.')
        return
    if action=='prepare':prepare_stage(config,s);return
    if action=='assess':assess(s,output);return
    if not (data/'stage_metadata.json').exists():raise ValueError('Explicit --prepare is required first')
    data_meta=json.loads((data/'stage_metadata.json').read_text())
    if (data_meta['stage']!=s['active_stage'] or data_meta['environment_config']!=config or
        data_meta.get('interface_contract')!=s['interface_contract'] or
        data_meta.get('reference_recovery')!=s['reference_recovery']):
        raise ValueError('Stage data/physical configuration mismatch')
    if any(hashlib.sha256((data/name).read_bytes()).hexdigest()!=digest for name,digest in data_meta['data_hashes'].items()):
        raise ValueError('Prepared scene data changed')
    if action=='collect_teacher':
        scenes=json.loads((data/'train.json').read_text());collect_teacher(config,scenes,s,data/'teacher');return
    require_previous(s)
    if action=='pretrain':
        model=initialize_gated(s['source_model'],s);pretrain_actor(model,data/'teacher',s,bc_dir)
        validate_bc(config,s,data,bc_dir)
    elif action=='validate_bc':validate_bc(config,s,data,bc_dir)
    elif action=='train':train(config,s,data,bc_dir,output)


if __name__=='__main__':main()
