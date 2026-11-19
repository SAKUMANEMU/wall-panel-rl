"""Diagnose collisions, collect late corrective labels, compare two BC continuations.

No PPO or stage promotion. Test scenes are frozen before fitting, scored only
after BOTH selections. No training/collection runs without --run.
"""
import _bootstrap
import argparse
import copy
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pybullet as p
import torch
import yaml
from stable_baselines3 import PPO
import staged_avoidance as stage
from run_bc_budget import environment,check_bank
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv,stage_scene,clearance_feedback
from wall_panel_rl.rl.avoidance_imitation import (
    AlignmentTeacher,make_teacher,save_teacher,pretrain_actor,teacher_reuse_contract)
from wall_panel_rl.rl.autonomous_dynamic_ppo import without_obstacles
from wall_panel_rl.utils.metrics import capture_scenario


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,data):stage.dump(Path(path),data)


def contact_details(e):
    """Read physical contacts without changing robot configuration or the verdict."""
    result=[]
    pairs=[(body,obj,name) for obj,name in [(v,'obstacle') for v in e.obstacle_ids]+[(e.wall,'wall')]
           for body in (e.robot,e.panel)]
    pairs.extend([(e.robot,e.robot,'self'),(e.robot,e.panel,'arm_panel')])
    for a,b,kind in pairs:
        contacts=list(p.getContactPoints(a,b,physicsClientId=e.client))
        contacts+=list(p.getClosestPoints(a,b,0.,physicsClientId=e.client))
        seen=set()
        for c in contacts:
            if c[8]>0 or (kind=='self' and abs(c[3]-c[4])<=1) or (kind=='arm_panel' and c[3]==e.ee):continue
            key=(c[3],c[4])
            if key in seen:continue
            seen.add(key)
            result.append({'kind':kind,'moving':'panel' if a==e.panel else 'arm',
                           'link_a':c[3],'link_b':c[4],'distance_m':float(c[8])})
    return result


def route_distance(q,path):
    path=np.asarray(path,dtype=float);v=np.diff(path,axis=0)
    t=np.clip(np.sum((q-path[:-1])*v,axis=1)/np.maximum(np.sum(v*v,axis=1),1e-12),0,1)
    return float(np.min(np.linalg.norm(q-path[:-1]-t[:,None]*v,axis=1)))


def trace(model,config,s,case):
    env=StagedAvoidanceEnv(config,[],s);states=[];total=0.
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        while True:
            q,dq=env.env.joint_state()
            _,distance=clearance_feedback(env.env,.30,0.)
            action=model.predict(obs,deterministic=True)[0]
            row={'step':env.env.steps,'obs':obs.tolist(),'q':q.tolist(),'dq':dq.tolist(),
                 'clearance_m':distance,'route_distance_rad':route_distance(q,case['scene']['reference_joint_path']),
                 'action':action.tolist(),'position_m':float(info['position_error'])}
            obs,reward,done,truncated,info=env.step(action);total+=reward
            row['executed_action']=info['executed_action'];states.append(row)
            if done or truncated:break
        return {'seed':case['scene']['seed'],'reason':info['termination_reason'],'steps':env.env.steps,
                'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                'reward':total,'contacts':contact_details(env.env),'states':states,'case':case}
    finally:env.close()


def handoffs(t,leads):
    """Choose BEFORE the failing action; replayed prefixes contain no expert labels.

    Include late states even if no distance threshold fired. A self collision or
    bad detour may happen far from the obstacle. The model's actual full rollout
    determines every candidate, rather than a fixed short prefix budget.
    """
    if t['reason']=='success':return []
    last=len(t['states'])-1
    candidates={max(0,last-lead) for lead in leads}
    near=[r['step'] for r in t['states'] if r['clearance_m'] is not None and r['clearance_m']<.12]
    if near:candidates.add(max(0,near[0]-2))
    deviated=[r['step'] for r in t['states'] if r['route_distance_rad']>.20]
    if deviated:candidates.add(max(0,deviated[0]-2))
    return sorted(candidates,reverse=True)


def collect_corrections(config,s,model,traces,output,study,status,*,handoff_selector=None):
    output.mkdir(parents=True,exist_ok=False)
    selector=handoff_selector or (lambda t:handoffs(t,study['handoff_leads']))
    points={id(t):selector(t) for t in traces}
    candidates=[(t,k) for t in traces for k in points[id(t)]]
    # Round-robin across layouts and phases before revisiting one failure.
    candidates.sort(key=lambda item:(points[id(item[0])].index(item[1]),item[0]['seed']))
    if not candidates:raise RuntimeError('No failing training rollouts for corrective collection')
    observations=[];actions=[];raw=[];offsets=[0];seeds=[];tasks=[];terminals=[];terminal_actions=[];attempts=[]
    anchor=AlignmentTeacher(model);env=StagedAvoidanceEnv(config,[],s)
    try:
        for t,k in candidates[:study['maximum_correction_attempts']]:
            case=t['case'];scene=case['scene']
            obs,info=env.reset(options={'scenario':scene,'task':case['task'],'strength':case['strength']})
            warmup=[]
            for r in t['states'][:k]:
                obs,_,done,truncated,info=env.step(r['executed_action']);warmup.append(info['executed_action'])
                if done or truncated:raise RuntimeError('Saved deterministic prefix unexpectedly terminated')
            expected=np.asarray(t['states'][k]['obs'])
            error=float(np.max(np.abs(obs-expected)))
            if error>1e-6:raise RuntimeError(f'Prefix state replay differs: {error}')
            expert=make_teacher(env.env,model,'correction',scene,scene['seed']+len(attempts))
            x=[];y=[];z=[];action_error=[];sign_error=[]
            if expert.planning_success:
                while True:
                    command=expert.action(obs);pred=model.predict(obs,deterministic=True)[0]
                    next_obs,_,done,truncated,info=env.step(command)
                    x.append(obs.copy());y.append(info['executed_action']);z.append(np.asarray(command).copy())
                    action_error.append(float(np.mean((pred-np.asarray(info['executed_action']))**2)))
                    active=np.abs(info['executed_action'])>.05
                    if np.any(active):sign_error.append(float(np.mean(np.sign(pred[active])!=np.sign(np.asarray(info['executed_action'])[active]))))
                    obs=next_obs
                    if done or truncated:break
                reason=info['termination_reason']
            else:reason='planning_failure'
            a={'attempt':len(attempts),'group':'correction','seed':scene['seed'],'reason':reason,
               'steps':env.env.steps,'warmup_actions':warmup,'warmup_steps':len(warmup),'scene':scene,
               'prefix_replay_error':error,'original_policy_failure':t['reason'],'original_failure_step':t['steps'],
               'policy_expert_action_mse':float(np.mean(action_error)) if action_error else None,
               'policy_expert_sign_disagreement':float(np.mean(sign_error)) if sign_error else None}
            attempts.append(a)
            if reason=='success':
                observations.extend(x);actions.extend(y);raw.extend(z);offsets.append(len(observations))
                seeds.append(scene['seed']);tasks.append('avoidance');terminals.append(obs.copy())
                terminal_actions.append(anchor.action(obs))
            if len(attempts)%10==0 or len(seeds)==study['correction_successes']:
                save_teacher(output/'teacher.npz',observations,actions,raw,offsets,seeds,tasks,terminals,terminal_actions)
                dump(output/'attempts.json',attempts)
                status('collect_corrections',correction_successes=len(seeds),correction_attempts=len(attempts))
            if len(seeds)>=study['correction_successes']:break
        save_teacher(output/'teacher.npz',observations,actions,raw,offsets,seeds,tasks,terminals,terminal_actions)
        dump(output/'attempts.json',attempts)
        if len(seeds)<study['correction_successes']:
            raise RuntimeError(f'Bounded corrective quota not met ({len(seeds)}/{study["correction_successes"]}); no automatic retries')
        dump(output/'metadata.json',{'completed':True,'episodes':len(seeds),'transitions':len(observations),
            'teacher_sha256':sha(output/'teacher.npz'),'model_sha256':sha(study['initial_model']),
            'training_scenes_only':True,'prefixes_are_not_labels':True,'fixed_endpoint_expert':True,
            'settings':s,'environment_config':config,'warmup_steps':[a['warmup_steps'] for a in attempts if a['reason']=='success']})
    finally:env.close()


def merge_teacher(base,corrections,output,s,config):
    output.mkdir(parents=True,exist_ok=False)
    with np.load(base/'teacher.npz',allow_pickle=False) as a,np.load(corrections/'teacher.npz',allow_pickle=False) as b:
        save_teacher(output/'teacher.npz',np.concatenate([a['obs'],b['obs']]),
            np.concatenate([a['actions'],b['actions']]),np.concatenate([a['raw_actions'],b['raw_actions']]),
            np.r_[a['offsets'],b['offsets'][1:]+len(a['obs'])],np.r_[a['seeds'],b['seeds']],
            np.r_[a['tasks'],b['tasks']],np.concatenate([a['terminal_obs'],b['terminal_obs']]),
            np.concatenate([a['terminal_actions'],b['terminal_actions']]))
        unique=len(np.unique(np.r_[a['seeds'],b['seeds']]))
        task_counts=dict(Counter(np.r_[a['tasks'],b['tasks']].tolist()))
        count=len(a['seeds'])+len(b['seeds']);transitions=len(a['obs'])+len(b['obs'])
    meta=copy.deepcopy(read(base/'metadata.json'))
    meta.update(completed=True,settings=s,environment_config=config,unique_scene_seeds=unique,
        transitions=transitions,terminal_anchors=count,teacher_sha256=sha(output/'teacher.npz'),
        source_sha256=sha(s['source_model']),parent_teacher_sha256=sha(base/'teacher.npz'),
        correction_teacher_sha256=sha(corrections/'teacher.npz'),
        collection='parent_plus_targeted_corrections',fixed_heldout_split=True)
    meta['counts']=task_counts
    dump(output/'metadata.json',meta)


def make_test_bank(config,s,study,output,forbidden,status):
    env=WallPanelEnv(config);cases=[];failures=[]
    try:
        for seed in range(study['independent_start_seed'],study['independent_start_seed']+study['maximum_generation_attempts']):
            if seed in forbidden:raise ValueError('Independent seed intersects prior training/validation')
            try:
                env.reset(seed=seed);original=capture_scenario(env,seed)
            except RuntimeError as exc:
                failures.append({'seed':seed,'error':str(exc)});continue
            selected=stage_scene(original,s['active_stage'])
            cases.append({'suite':'stage_target','task':'static','strength':[.05,.02],'scene':selected})
            if len(cases)%10==0:status('freeze_independent_bank',generated=len(cases))
            if len(cases)==study['independent_scenes']:break
    finally:env.close()
    if len(cases)!=study['independent_scenes']:raise RuntimeError('Independent generation budget exhausted')
    target=copy.deepcopy(cases)
    for i,case in enumerate(target[:20]):
        near=i>=10;scene=without_obstacles(case['scene'],.10 if near else None)
        cases.append({'suite':'alignment_near' if near else 'alignment_far','task':'alignment',
                      'strength':[.05,.02],'scene':scene})
    dump(output,cases);dump(output.parent/'independent_generation_failures.json',failures)
    return cases


def score_only(model,config,s,cases):
    """Candidate-only independent test; not supplied to checkpoint selection."""
    rows=[];env=StagedAvoidanceEnv(config,[],s)
    try:
        for case in cases:
            obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            while True:
                obs,_,done,truncated,info=env.step(model.predict(obs,deterministic=True)[0])
                if done or truncated:break
            rows.append({'suite':case['suite'],'seed':case['scene']['seed'],'reason':info['termination_reason'],
                'steps':env.env.steps,'position_m':float(info['position_error']),
                'orientation_deg':float(np.rad2deg(info['orientation_error']))})
    finally:env.close()
    metrics={}
    for suite in sorted({r['suite'] for r in rows}):
        group=[r for r in rows if r['suite']==suite]
        metrics[suite]={'episodes':len(group),**{k:sum(r['reason']==k for r in group)
            for k in ('success','collision','joint_limit','timeout')}}
    return {'metrics':metrics,'rows':rows,'used_for_selection':False}


def check(study):
    if study['additional_updates']!=10000 or study['checkpoints']!=[2000,4000,6000,8000,10000]:
        raise ValueError('Explicit bounded two-run experiment required')
    for key in ('initial_model','initial_state','initial_metadata'):
        if not Path(study[key]).is_file():raise FileNotFoundError(study[key])
    meta=read(study['initial_metadata'])
    if meta['selected_update']!=10000:raise ValueError('Expected the selected 10k BC baseline')
    a=PPO.load(study['initial_model'],device='cpu')
    b=PPO.load(Path(study['initial_metadata']).parent/'model.zip',device='cpu')
    if a.num_timesteps!=b.num_timesteps or any(not torch.equal(v,b.policy.state_dict()[k])
                                               for k,v in a.policy.state_dict().items()):
        raise ValueError('Continuation model does not match selected baseline')
    state=torch.load(study['initial_state'],map_location='cpu',weights_only=False)
    if state['updates']!=10000:raise ValueError('Wrong optimizer checkpoint')
    if not 0<study['correction_successes']<=study['maximum_correction_attempts']:raise ValueError('Invalid correction budget')
    return meta


def run(study):
    meta=check(study);root=Path(study['output_root']);data=Path(study['data_root']);report=Path(study['report_root'])
    if any(p.exists() for p in (root,data,report)):raise FileExistsError('Refusing to overwrite/restart a study')
    s=copy.deepcopy(meta['settings']);config=environment(s);check_bank(config,s)
    binding=teacher_reuse_contract(read(Path(study['base_data'])/'teacher/metadata.json'),s,config)
    if sha(Path(study['base_data'])/'teacher/teacher.npz')!=binding['teacher_sha256']:raise ValueError('Original data changed')
    for folder in (root,data,report):folder.mkdir(parents=True)
    state={'status':'running','runner_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),
           'ppo_environment_steps':0,'additional_bc_updates_per_variant':study['additional_updates']}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('TARGETED',json.dumps(state),flush=True)
    manifest={'study':study,'baseline_hash':sha(study['initial_model']),'optimizer_hash':sha(study['initial_state']),
        'base_data_binding':binding,'configuration':config,'test_used_for_selection':False,
        'same_initial_weights_optimizer_rng':True,'code_hashes':{p:sha(p) for p in
            ['scripts/run_targeted_bc.py','src/wall_panel_rl/rl/avoidance_imitation.py',
             'src/wall_panel_rl/rl/avoidance_curriculum.py','scripts/staged_avoidance.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    try:
        model=PPO.load(study['initial_model'],device='cpu');source=PPO.load(s['source_model'],device='cpu')
        old_cases=read(Path(study['base_data'])/'validation_staged.json')
        bank=read(Path(study['base_data'])/'train.json');held=set(meta['heldout_scene_seeds'])
        # The original supervised held-out seeds remain held out in BOTH arms.
        train_bank=[scene for scene in bank if scene['seed'] not in held]
        train_cases=[{'suite':'train_diagnostic','task':'static','strength':[.05,.02],
                      'scene':stage_scene(scene,s['active_stage'])} for scene in train_bank]
        traces={}
        for name,cases in [('validation',[c for c in old_cases if c['suite']=='stage_target']),('train',train_cases)]:
            status('diagnose_'+name);traces[name]=[]
            for case in cases:
                traces[name].append(trace(model,config,s,case))
                if len(traces[name])%10==0:status('diagnose_'+name,diagnosed=len(traces[name]))
            dump(report/f'{name}_traces.json',traces[name])
        diagnostic={name:{'episodes':len(ts),'outcomes':dict(Counter(t['reason'] for t in ts)),
            'collision_objects':dict(Counter(c['moving']+'_'+c['kind'] for t in ts if t['reason']=='collision' for c in t['contacts'])),
            'collision_steps':[t['steps'] for t in ts if t['reason']=='collision'],
            'unclassified_collisions':sum(t['reason']=='collision' and not t['contacts'] for t in ts)}
            for name,ts in traces.items()}
        dump(report/'collision_diagnosis.json',diagnostic)
        if diagnostic['validation']['outcomes']!={'success':10,'collision':9,'timeout':1}:
            raise RuntimeError('Baseline failed to reproduce historical 10/20; investigate before fitting')
        forbidden={c['scene']['seed'] for c in old_cases}|{c['seed'] for c in bank}
        tests=make_test_bank(config,s,study,data/'independent_cases.json',forbidden,status)
        dump(report/'independent_bank.json',{'sha256':sha(data/'independent_cases.json'),
            'target_scenes':study['independent_scenes'],'scored_after_both_model_selections':True})
        status('collect_corrections')
        collect_corrections(config,s,model,traces['train'],data/'corrections',study,status)
        status('audit_corrections')
        subprocess.run([sys.executable,'scripts/audit_correction_data.py','--data',str(data/'corrections'),
                        '--output',str(report/'correction_replay.json')],check=True)
        replay=read(report/'correction_replay.json')
        if replay['success']!=replay['sampled_corrections'] or replay['max_observation_error']>1e-6:
            raise RuntimeError('Corrective labels failed physical replay audit')
        merge_teacher(Path(study['base_data'])/'teacher',data/'corrections',data/'mixed_teacher',s,config)
        with np.load(data/'corrections/teacher.npz',allow_pickle=False) as z:
            if held & set(z['seeds'].tolist()):raise AssertionError('Corrections leaked supervised held-out layouts')
        selected={}
        for name,teacher in [('original_only',Path(study['base_data'])/'teacher'),('targeted_mix',data/'mixed_teacher')]:
            variant=copy.deepcopy(s);variant['stage_output']=str(root/name);variant['output']=str(root/name/'ppo')
            variant['teacher'].update(updates=study['additional_updates'],closed_loop_updates=study['checkpoints'])
            output=root/name/'bc';status('bc_'+name,bc_update=0)
            def evaluator(candidate,update,loss):
                result,rows=stage.score_bc_model(candidate,source,config,variant,old_cases)
                dump(output/'evaluations'/f'update_{update}.json',result)
                dump(output/'evaluations'/f'episodes_{update}.json',rows)
                status('bc_'+name,bc_update=update,validation_target=result['metrics']['dynamic_ppo']['stage_target'])
                return result
            pretrain_actor(PPO.load(study['initial_model'],device='cpu'),teacher,variant,output,
                reuse_environment=config,evaluator=evaluator,resume_state=study['initial_state'],
                heldout_seeds=meta['heldout_scene_seeds'],include_initial=True)
            selected[name]=read(output/'metadata.json')
            dump(report/f'{name}.json',selected[name])
        # Test scores cannot change the selected checkpoints or winner.
        status('independent_test')
        from wall_panel_rl.rl.avoidance_imitation import closed_loop_score
        winner=max(selected,key=lambda k:closed_loop_score(selected[k]['selected_closed_loop'],selected[k]['heldout_selection_objective']))
        independent={}
        for name,path in [('baseline',study['initial_model'])]+[(k,root/k/'bc/model.zip') for k in selected]:
            independent[name]=score_only(PPO.load(path,device='cpu'),config,s,tests)
            dump(report/f'independent_{name}.json',independent[name])
            status('independent_test',evaluated_model=name,independent_target=independent[name]['metrics']['stage_target'])
        # Export using original validation only; independent test is reporting-only.
        shutil.copy2(root/winner/'bc/model.zip',report/'selected_bc_model.zip')
        dump(report/'summary.json',{'winner_by_original_validation':winner,'selected':{
            k:{'selected_additional_updates':v['selected_update'],
               'selected_cumulative_bc_updates':v['selected_cumulative_bc_updates'],
               'validation':v['selected_closed_loop']['metrics']['dynamic_ppo'],
               'passed_bc_gate':v['selected_closed_loop']['passed']} for k,v in selected.items()},
            'independent':{k:v['metrics'] for k,v in independent.items()},'ppo_environment_steps':0,
            'export_sha256':sha(report/'selected_bc_model.zip')})
        status('complete',status='completed',finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state['phase'],status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/targeted_bc_study.yaml')
    p.add_argument('--run',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
