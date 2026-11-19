"""Two static obstacles -> strong mixed-task RRT BC -> gated short PPO adaptation."""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import torch
import numpy as np
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from run_static_expansion import known_scene_seeds,freeze_fresh,project_cases
from train_conditional_ppo import score_model
from train_two_static_ppo import train as train_ppo
from wall_panel_rl.rl.avoidance_curriculum import stage_scene,gate_result
from wall_panel_rl.rl.avoidance_imitation import save_teacher,AlignmentTeacher
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import make_rrt,evaluate_controller
from wall_panel_rl.rl.multi_obstacle_warmup import warmup_actor,teacher_arrays


def stage(settings,count):
    result=copy.deepcopy(settings);result['active_stage'].update(count=count,name='one_static' if count==1 else 'two_static',mode='static')
    return result


def validation(model,config,two,one,two_cases,one_cases):
    a,arows=score_model(model,None,config,two,two_cases)
    b,brows=score_model(model,None,config,one,one_cases)
    at=a['metrics']['dynamic_ppo'];bt=b['metrics']['dynamic_ppo']
    checks={'two_gate':gate_result(at,two['gate'])['passed'],'one_gate':gate_result(bt,one['gate'])['passed'],
        'one_zero_collision':bt['stage_target']['collision']==0,
        'goals':a['checks']['goal_convergence'] and b['checks']['goal_convergence'],
        'alignment_collision_free':a['checks']['alignment_collision_free'] and b['checks']['alignment_collision_free']}
    eligible=all(checks.values());x=at['stage_target'];y=bt['stage_target']
    ready=(gate_result(at,two['bc_gate'])['passed'] and all(checks[k] for k in
        ('one_gate','one_zero_collision','goals','alignment_collision_free')))
    rank=([2,-x['collision'],x['success'],-y['collision'],y['success'],-x['timeout']] if eligible else
          [int(ready),x['success'],-x['collision'],-x['joint_limit'],y['success'],-y['collision']])
    return {'eligible':eligible,'ppo_ready':ready,'score':rank,'checks':checks,'two':at,'one':bt},arows,brows


def collect(study,config,one,two,scenes,source,output,status):
    output=Path(output);output.mkdir(parents=True)
    obs=[];actions=[];raw=[];offsets=[0];seeds=[];groups=[];terminals=[];anchors=[];attempts=[]
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],two);reference=AlignmentTeacher(source)
    try:
        for index,original in enumerate(scenes):
            for group,quota in study['demonstrations_per_scene'].items():
                for repeat in range(quota):
                    selected=two if group=='two_static' else one
                    rng=np.random.default_rng(original['seed']+repeat*10000+(200000 if group=='two_static' else 100000))
                    scene=stage_scene(original,selected['active_stage'],rng)
                    if group=='alignment':
                        from wall_panel_rl.rl.autonomous_dynamic_ppo import without_obstacles
                        scene=without_obstacles(scene,.1 if index%2 else None)
                    success=False
                    for retry in range(study['maximum_attempts_per_demonstration']):
                        x,info=env.reset(options={'scenario':scene,'task':'alignment' if group=='alignment' else 'static',
                            'strength':[selected['active_stage']['amplitude'],selected['active_stage']['speed']]})
                        planner_seed=original['seed']+(200000 if group=='two_static' else 100000)+retry*1000
                        expert=reference if group=='alignment' else make_rrt(env,planner_seed)
                        xo=[];ya=[];ra=[]
                        if expert.planning_success:
                            while True:
                                action=expert.action(x);previous=x.copy()
                                x,_,done,truncated,info=env.step(action)
                                xo.append(previous);ya.append(info['executed_action']);ra.append(np.asarray(action).tolist())
                                if done or truncated:break
                            reason=info['termination_reason']
                        else:reason='planning_failure'
                        success=reason=='success'
                        attempts.append({'group':group,'seed':original['seed'],'repeat':repeat,'retry':retry,'reason':reason,
                            'steps':env.env.steps,'planner_seed':planner_seed,'planning_seconds':expert.planning_time,
                            'scene':scene,'saved_episode':len(seeds) if success else None})
                        if success:
                            obs.extend(xo);actions.extend(ya);raw.extend(ra);offsets.append(len(obs));seeds.append(original['seed']);groups.append(group)
                            terminals.append(x.copy());anchors.append(reference.action(x));break
                    if not success:raise RuntimeError(f'Teacher quota failed for {original["seed"]} {group}')
            if (index+1)%4==0 or index==len(scenes)-1:
                save_teacher(output/'teacher.npz',obs,actions,raw,offsets,seeds,groups,terminals,anchors)
                dump(output/'attempts.json',attempts)
                status('collect_teacher',completed_layouts=index+1,demonstrations=len(seeds),attempts=len(attempts))
    finally:env.close()
    dump(output/'metadata.json',{'completed':True,'episodes':len(seeds),'transitions':len(obs),
        'group_counts':{g:groups.count(g) for g in set(groups)},'source_sha256':sha(study['initial_model']),
        'teacher_sha256':sha(output/'teacher.npz'),'raw_scenes':study['raw_train'],
        'controller':'RRT for one/two-static, frozen reference servo for alignment'})


def audit_teacher(path,config,two):
    with np.load(path/'teacher.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    attempts=read(path/'attempts.json');checked=[]
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],two)
    try:
        for group in ('two_static','one_static','alignment'):
            episode=int(np.flatnonzero(data['tasks']==group)[0]);attempt=next(a for a in attempts if a['saved_episode']==episode)
            x,_=env.reset(options={'scenario':attempt['scene'],'task':'alignment' if group=='alignment' else 'static',
                'strength':[two['active_stage']['amplitude'],two['active_stage']['speed']]})
            error=0.
            for i in range(data['offsets'][episode],data['offsets'][episode+1]):
                error=max(error,float(np.max(np.abs(x-data['obs'][i]))))
                x,_,done,truncated,info=env.step(data['raw_actions'][i])
                error=max(error,float(np.max(np.abs(np.asarray(info['executed_action'])-data['actions'][i]))))
            if not (done or truncated) or info['termination_reason']!='success' or error>1e-6:
                raise AssertionError(f'Teacher physical replay failed: {group} {error}')
            checked.append({'group':group,'max_error':error,'reason':info['termination_reason']})
    finally:env.close()
    return checked


def run(study):
    data=Path(study['data_root']);models=Path(study['model_root']);report=Path(study['report_root'])
    if any(p.exists() for p in (data,models,report)):raise FileExistsError('Refusing repeated study/overwrite')
    parent=read(Path(study['source_report'])/'manifest.json')
    if sha(study['initial_model'])!=parent['model_sha256']:raise ValueError('Wrong one-static source')
    forbidden=known_scene_seeds();data.mkdir(parents=True);models.mkdir(parents=True);report.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'automatic_dynamic_stage':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('TWO_STATIC',json.dumps(state),flush=True)
    config=copy.deepcopy(parent['configuration']);one=stage(parent['evaluation_settings'],1);two=stage(one,2)
    held=set(read(study['heldout_metadata'])['heldout_scene_seeds'])
    raw_scenes=read(study['raw_train']);source_hash=sha(study['initial_model'])
    dump(report/'protocol.json',{'study':study,'configuration':config,'source_sha256':source_hash,
        'two_indices':[2,3],'heldout_teacher_layouts':sorted(held),'test_used_for_selection':False,
        'ppo_start_gate':'two success>=50%, collision<=25%, no limits; one>=80% and zero validation collisions; retain alignment/goal',
        'deployment_gate':'two success>=80%, collision<=10%, no limits; preserve one/alignment/goal',
        'ppo_after_bc':'fresh optimizer and lower LR, separate auxiliary optimizer seeded with BC training data'})
    try:
        status('freeze_fresh')
        fresh,failures=freeze_fresh(study,config,one,forbidden,status)
        two_test=project_cases(fresh,two['active_stage']);dump(data/'two_test.json',two_test)
        dump(data/'fresh_full.json',fresh);dump(data/'generation_failures.json',failures)
        one_cases=[c for c in read(study['one_validation']) if c['suite']!='five_full_diagnostic']
        two_cases=project_cases(read(study['raw_validation']),two['active_stage'])
        dump(data/'one_validation.json',one_cases);dump(data/'two_validation.json',two_cases)
        train_ids={s['seed'] for s in raw_scenes};val_ids={c['scene']['seed'] for c in two_cases}
        if {s['seed'] for s in fresh}&(forbidden|train_ids|val_ids):raise ValueError('Fresh scene leakage')
        dump(report/'data_hashes.json',{str(p):sha(p) for p in (data/'two_test.json',data/'fresh_full.json',
            data/'one_validation.json',data/'two_validation.json')})
        source=PPO.load(study['initial_model'],device='cpu')
        base,arows,brows=validation(source,config,two,one,two_cases,one_cases);dump(report/'baseline.json',base)
        status('collect_teacher',two_baseline=base['two']['stage_target'],one_baseline=base['one']['stage_target'])
        collect(study,config,one,two,raw_scenes,source,data/'teacher',status)
        dump(report/'teacher_replay.json',audit_teacher(data/'teacher',config,two))
        dump(report/'rrt_validation.json',evaluate_controller(None,config,two,two_cases,mode='rrt'))
        teacher_arrays(data/'teacher/teacher.npz',held)
        def evaluator(candidate,update):
            result,rows,one_rows=validation(candidate,config,two,one,two_cases,one_cases)
            dump(report/'bc_validation'/f'update_{update}.json',{'summary':result,'two_rows':rows,'one_rows':one_rows})
            status('bc_warmup',bc_updates=update,two_validation=result['two']['stage_target'],
                one_validation=result['one']['stage_target'],eligible=result['eligible'])
            return result
        spec=dict(study['bc'],seed=study['seed'])
        warm_path,metadata=warmup_actor(PPO.load(study['initial_model'],device='cpu'),data/'teacher/teacher.npz',
            held,spec,models/'bc',evaluator)
        selected=metadata['selected'];ppo_started=False;chosen_path=warm_path
        if selected['ppo_ready']:
            status('prepare_ppo',selected_bc_update=selected['update'])
            bundle=copy.deepcopy(parent)
            for key in ('training_settings','evaluation_settings'):
                bundle[key]=stage(bundle[key],2)
            bundle.update(model_sha256=sha(warm_path),warmup_metrics=selected)
            dump(data/'warm_source/manifest.json',bundle)
            ppo_study={'source_report':str(data/'warm_source'),'initial_model':str(warm_path),
                'train_scenes':study['raw_train'],'heldout_scenes':study['heldout_metadata'],
                'validation_cases':str(data/'two_validation.json'),'output_root':str(models/'ppo_run'),
                'report_root':str(report/'ppo_run'),'ppo_steps':study['ppo_steps'],'ppo_eval_every':study['ppo_eval_every'],
                'ppo_learning_rate':study['ppo_learning_rate'],'expert_probability':study['expert_probability'],
                'seed':study['seed'],'auxiliary_bc':study['auxiliary_bc']}
            (data/'ppo_study.yaml').write_text(yaml.safe_dump(ppo_study,sort_keys=False))
            chosen_path,_=train_ppo(ppo_study,one,one_cases,data/'teacher/teacher.npz',held,status)
            ppo_started=True
        else:status('bc_gate_failed',selected_bc_update=selected['update'])
        status('heldout_verification',ppo_started=ppo_started)
        chosen=PPO.load(chosen_path,device='cpu')
        before=evaluate_controller(source,config,two,two_test,mode='ppo')
        after=evaluate_controller(chosen,config,two,two_test,mode='ppo')
        old_cases=read(study['one_regression'])
        old_before=evaluate_controller(source,config,one,old_cases,mode='ppo')
        old_after=evaluate_controller(chosen,config,one,old_cases,mode='ppo')
        for name,result in [('two_test_before',before),('two_test_after',after),('one_before',old_before),('one_after',old_after)]:
            dump(report/f'{name}.json',result)
        import shutil
        shutil.copy2(chosen_path,report/'candidate_model.zip')
        final_check,_,_=validation(chosen,config,two,one,two_cases,one_cases)
        holdout_gate=gate_result(after['metrics'],two['gate'])
        if sha(study['initial_model'])!=source_hash:raise AssertionError('Source model modified')
        dump(report/'summary.json',{'baseline':base,'selected_bc':selected,'bc_updates_run':study['bc']['updates'],
            'ppo_started':ppo_started,'ppo_budget':study['ppo_steps'] if ppo_started else 0,
            'two_holdout_before':before['metrics'],'two_holdout_after':after['metrics'],
            'one_before':old_before['metrics'],'one_after':old_after['metrics'],
            'final_validation':final_check,'heldout_gate':holdout_gate,'candidate_path':str(chosen_path),
            'candidate_sha256':sha(chosen_path),'source_unchanged':True,'default_model_preserved':True,
            'fresh_test_used_for_selection':False})
        dump(report/'candidate_manifest.json',{'configuration':config,'evaluation_settings':two,
            'model_sha256':sha(chosen_path),'source_model':study['initial_model'],'validation_eligible':final_check['eligible']})
        status('complete',status='completed',ppo_started=ppo_started,two_holdout=after['metrics']['stage_target'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/two_static_warmup.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text())
    if study['bc']['updates']!=10000 or study['ppo_steps']!=10000:raise ValueError('One bounded warm-up and PPO pilot')
    print({'mode':'run' if args.run else 'check_only','study':study},flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
