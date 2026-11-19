"""Fresh split banks and physically successful, prevalidated RRT demonstrations.

The original99 cases remain regression only. Validation/test banks receive no
four-obstacle expert rollouts. No BC or PPO optimization happens in this script.
"""
import _bootstrap
import argparse
import copy
from datetime import datetime
from pathlib import Path
import json
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from run_static_expansion import known_scene_seeds
from wall_panel_rl.rl.static_demo_scene import fresh_static_case
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import make_rrt
from wall_panel_rl.rl.avoidance_imitation import save_teacher,AlignmentTeacher
from wall_panel_rl.planning.static_validation import dense_route_valid


def collect(model,config,settings,cases,offsets_to_try,output,status):
    output=Path(output);output.mkdir(parents=True)
    observations=[];actions=[];raw=[];offsets=[0];seeds=[];tasks=[];terminals=[];anchors=[];attempts=[]
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings);anchor=AlignmentTeacher(model)
    def save():
        save_teacher(output/'teacher.npz',observations,actions,raw,offsets,seeds,tasks,terminals,anchors)
        dump(output/'attempts.json',attempts)
    try:
        for i,case in enumerate(cases):
            for attempt,delta in enumerate(offsets_to_try):
                obs,info=env.reset(options={'scenario':case['scene'],'task':'static','strength':case['strength']})
                original_reference=env.cursor.path.copy();planner_seed=int(case['scene']['seed'])+delta
                expert=make_rrt(env,planner_seed);x=[];y=[];z=[]
                if expert.planning_success and expert.static_path_verified:
                    while True:
                        command=expert.action(obs);previous=obs.copy()
                        obs,_,done,truncated,info=env.step(command)
                        x.append(previous);y.append(info['executed_action']);z.append(command.copy())
                        if done or truncated:break
                    reason=info['termination_reason']
                    postchecked=dense_route_valid(env.env.collision,expert.path,expert.static_validation_resolution)
                else:reason='planning_failure';postchecked=False
                accepted=reason=='success' and postchecked and bool(expert.static_path_verified)
                if not np.array_equal(original_reference,env.cursor.path):raise AssertionError('Expert replaced policy reference')
                attempts.append({'seed':case['scene']['seed'],'attempt':attempt,'planner_seed':planner_seed,
                    'reason':reason,'planning_success':expert.planning_success,'pre_execution_verified':expert.static_path_verified,
                    'post_execution_verified':postchecked,'accepted':accepted,'steps':env.env.steps,
                    'planning_seconds':expert.planning_time,'position_m':float(info['position_error']),
                    'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                    'saved_episode':len(seeds) if accepted else None})
                if accepted:
                    observations.extend(x);actions.extend(y);raw.extend(z);offsets.append(len(observations))
                    seeds.append(case['scene']['seed']);tasks.append('four_static');terminals.append(obs.copy());anchors.append(anchor.action(obs))
                    break
            save();status('collect_verified_teacher',layouts_completed=i+1,episodes=len(seeds),attempts=len(attempts))
    finally:env.close()
    meta={'requested_layouts':len(cases),'episodes':len(seeds),'transitions':len(observations),
        'attempts':len(attempts),'accepted_seeds':seeds,'unresolved_train_seeds':[c['scene']['seed'] for c in cases if c['scene']['seed'] not in set(seeds)],
        'all_accepted_pre_and_post_verified':all(a['pre_execution_verified'] and a['post_execution_verified'] for a in attempts if a['accepted']),
        'all_accepted_physically_successful':all(a['reason']=='success' for a in attempts if a['accepted']),
        'original_reference_preserved':True,'one_route_per_layout':len(seeds)==len(set(seeds)),
        'teacher_sha256':sha(output/'teacher.npz'),'ready_for_bc':len(seeds)==len(cases),'training_steps':0}
    dump(output/'metadata.json',meta);return meta


def audit_replay(config,settings,cases,teacher):
    by_seed={c['scene']['seed']:c for c in cases};checks=[]
    with np.load(teacher/'teacher.npz',allow_pickle=False) as z:d={name:z[name].copy() for name in z.files}
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings)
    try:
        for episode in range(min(3,len(d['seeds']))):
            case=by_seed[int(d['seeds'][episode])]
            obs,_=env.reset(options={'scenario':case['scene'],'task':'static','strength':case['strength']});error=0.
            for k in range(d['offsets'][episode],d['offsets'][episode+1]):
                error=max(error,float(np.max(np.abs(obs-d['obs'][k]))))
                obs,_,done,truncated,info=env.step(d['raw_actions'][k])
                error=max(error,float(np.max(np.abs(np.asarray(info['executed_action'])-d['actions'][k]))))
            error=max(error,float(np.max(np.abs(obs-d['terminal_obs'][episode]))))
            if not(done or truncated) or info['termination_reason']!='success' or error>1e-6:raise AssertionError('Teacher physical replay differs')
            checks.append({'seed':int(d['seeds'][episode]),'reason':info['termination_reason'],'maximum_error':error})
    finally:env.close()
    return checks


def run(spec):
    source=Path(spec['source_report']);data=Path(spec['data_root']);report=Path(spec['report_root'])
    if data.exists() or report.exists():raise FileExistsError('Refusing dataset overwrite')
    regression=Path(spec['verified_regression'])
    if read(regression/'status.json')['status']!='completed':raise RuntimeError('Complete verified99 regression first')
    parent=read(source/'manifest.json');source_hash=sha(source/'selected_model.zip')
    if source_hash!=parent['model_sha256']:raise ValueError('Wrong initialization source')
    config=copy.deepcopy(parent['configuration']);settings=copy.deepcopy(parent['evaluation_settings'])
    config['collision']['planning_clearance']=spec['planning_clearance_m']
    config['rrt'].update(verify_static_path=True,static_validation_resolution=spec['static_validation_resolution_rad'])
    model=PPO.load(source/'selected_model.zip',device='cpu');data.mkdir(parents=True);report.mkdir(parents=True)
    state={'status':'running','training_steps':0,'started_at':datetime.now().astimezone().isoformat()}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('SAFE_TEACHER',json.dumps(state),flush=True)
    code=['scripts/prepare_four_safe_teacher.py','configs/four_safe_teacher.yaml','src/wall_panel_rl/baselines/classical.py',
        'src/wall_panel_rl/planning/rrt_connect.py','src/wall_panel_rl/planning/static_validation.py']
    dump(report/'protocol.json',{'spec':spec,'configuration':config,'evaluation_settings':settings,
        'source_model_sha256':source_hash,'regression_case_sha256':sha(spec['regression_cases']),
        'code_hashes':{p:sha(p) for p in code},'training_steps':0,
        'label_rule':'one pre/post-verified, physically successful route per NEW training layout',
        'validation_and_test_obstacle_episodes_executed':False})
    try:
        forbidden=known_scene_seeds();banks={};rejected=[]
        for name,count,start in [('train',spec['train_count'],spec['train_seed_start']),
            ('validation',spec['validation_count'],spec['validation_seed_start']),('test',spec['test_count'],spec['test_seed_start'])]:
            cases=[]
            for seed in range(start,start+count*spec['maximum_generation_factor']):
                if seed in forbidden:raise ValueError('New scene seed already used')
                try:case,_=fresh_static_case(config,settings,seed,model=model)
                except RuntimeError as exc:rejected.append({'bank':name,'seed':seed,'reason':str(exc)});continue
                cases.append(case)
                if len(cases)==count:break
            if len(cases)!=count:raise RuntimeError('Bounded scene-generation budget exhausted')
            banks[name]=cases;forbidden.update(c['scene']['seed'] for c in cases)
            dump(data/f'{name}_cases.json',cases);status('freeze_'+name,cases=len(cases),generation_rejections=len(rejected))
        dump(data/'generation_rejections.json',rejected)
        sets={name:{c['scene']['seed'] for c in cases} for name,cases in banks.items()}
        old={c['scene']['seed'] for c in read(spec['regression_cases'])}
        if any(sets[a]&sets[b] for a in sets for b in sets if a!=b) or any(v&old for v in sets.values()):raise AssertionError('Scene split leakage')
        dump(report/'split.json',{name:sorted(values) for name,values in sets.items()})
        dump(report/'data_hashes.json',{str(data/f'{name}_cases.json'):sha(data/f'{name}_cases.json') for name in banks})
        meta=collect(model,config,settings,banks['train'],spec['planner_seed_offsets'],data/'teacher',status)
        if not set(meta['accepted_seeds']).issubset(sets['train']):raise AssertionError('Non-training labels')
        dump(report/'teacher_replay.json',audit_replay(config,settings,banks['train'],data/'teacher'))
        if sha(source/'selected_model.zip')!=source_hash:raise AssertionError('Policy weights changed')
        dump(report/'summary.json',{'teacher':meta,'bank_counts':{k:len(v) for k,v in banks.items()},
            'split_disjoint':True,'regression_excluded_from_training':True,'source_weights_unchanged':True,
            'validation_and_test_not_used_for_expert_collection':True,'training_steps':0})
        for path in code:
            dest=report/'code_archive'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        status('complete',status='completed',ready_for_bc=meta['ready_for_bc'],episodes=meta['episodes'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_safe_teacher.yaml');p.add_argument('--run',action='store_true')
    args=p.parse_args();torch.set_num_threads(1);spec=yaml.safe_load(Path(args.config).read_text())
    if spec['training_steps']!=0:raise ValueError('Demonstration preparation only')
    print({'mode':'run' if args.run else 'check_only','train':64,'validation':32,'test':64,'training_steps':0},flush=True)
    if args.run:run(spec)


if __name__=='__main__':main()
