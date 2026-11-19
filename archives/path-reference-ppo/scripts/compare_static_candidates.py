"""Evaluate retained PPO versus zero-collision candidate; no fitting or auto-selection."""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha,make_test_bank,score_only,contact_details,route_distance
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv,clearance_feedback
from wall_panel_rl.rl.wall_feedback import wall_feedback


def diagnose(model,config,s,case,horizon=240):
    physical=copy.deepcopy(config);physical['simulation']['max_steps']=horizon
    env=StagedAvoidanceEnv(physical,[],s);states=[]
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        while True:
            q,dq=env.env.joint_state();x=torch.as_tensor(obs[None])
            with torch.no_grad():
                prior=model.policy.reference_mean(x);strength,gate,residual=model.policy.correction_components(x)
                base=(1-strength+strength*gate)*prior;correction=strength*residual
            action=model.predict(obs,deterministic=True)[0]
            _,clearance=clearance_feedback(env.env,.3,0.)
            _,wall=wall_feedback(env.env,.1,0.)
            row={'step':env.env.steps,'q':q.tolist(),'dq':dq.tolist(),'reference':env.ref.tolist(),
                'reference_index':env.cursor.index,'last_reference_index':len(env.cursor.path)-1,
                'reference_recoveries':env.cursor.recoveries,
                'reference_distance_rad':float(np.linalg.norm(env.ref-q)),
                'route_distance_rad':route_distance(q,case['scene']['reference_joint_path']),
                'goal_max_joint_error_rad':float(np.max(np.abs(q-model.policy.goal_angles))),
                'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                'obstacle_clearance_capped_m':clearance,'wall_clearance_capped_m':wall,
                'action':action.tolist(),'action_norm':float(np.linalg.norm(action)),
                'prior_norm':float(torch.linalg.vector_norm(base)),
                'residual_norm':float(torch.linalg.vector_norm(correction)),
                'cancellation_ratio':float(np.linalg.norm(action)/(float(torch.linalg.vector_norm(base)+torch.linalg.vector_norm(correction))+1e-9)),
                'gate':float(gate.item()),'correction_strength':float(strength.item())}
            obs,_,done,truncated,info=env.step(action)
            row['executed_action']=info['executed_action'];states.append(row)
            if done or truncated:break
        return {'seed':case['scene']['seed'],'reason':info['termination_reason'],'steps':env.env.steps,
                'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                'contacts':contact_details(env.env),'states':states,'horizon':horizon}
    finally:env.close()


def stall_summary(trace):
    states=trace['states'];window=states[-40:];q=np.asarray([r['q'] for r in window])
    travel=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum());displacement=float(np.linalg.norm(q[-1]-q[0]))
    gain=window[0]['position_m']-window[-1]['position_m']
    angle_gain=window[0]['orientation_deg']-window[-1]['orientation_deg']
    ratio=float(np.mean([r['cancellation_ratio'] for r in window]))
    if travel<.01:kind='nearly_stationary'
    elif displacement<.25*travel:kind='oscillating_or_looping'
    elif gain>.001 or angle_gain>.1:kind='slow_task_progress'
    else:kind='moving_without_task_progress'
    final=window[-1]
    return {'seed':trace['seed'],'kind':kind,'position_m':trace['position_m'],'orientation_deg':trace['orientation_deg'],
        'joint_travel_last40_rad':travel,'joint_displacement_last40_rad':displacement,
        'position_progress_last40_m':float(gain),'orientation_progress_last40_deg':float(angle_gain),
        'mean_action_norm_last40':float(np.mean([r['action_norm'] for r in window])),
        'mean_cancellation_ratio_last40':ratio,'strong_action_cancellation':ratio<.1,
        'reference_index_change_last40':final['reference_index']-window[0]['reference_index'],
        'cursor_at_final_waypoint':final['reference_index']==final['last_reference_index'],
        'reference_distance_rad':final['reference_distance_rad'],'route_distance_rad':final['route_distance_rad'],
        'reference_recoveries':final['reference_recoveries'],'wall_clearance_capped_m':final['wall_clearance_capped_m'],
        'goal_max_joint_error_rad':final['goal_max_joint_error_rad']}


def paired_outcomes(a,b):
    x={r['seed']:r for r in a if r['suite']=='stage_target'};y={r['seed']:r for r in b if r['suite']=='stage_target'}
    if x.keys()!=y.keys():raise ValueError('Models did not receive the same cases')
    rows=[{'seed':seed,'retained':x[seed]['reason'],'candidate':y[seed]['reason']} for seed in sorted(x)]
    return {'episodes':len(rows),'transitions':dict(Counter(r['retained']+' -> '+r['candidate'] for r in rows)),'rows':rows}


def check(study):
    if study['independent_scenes']!=100 or study['diagnostic_horizon']!=480:raise ValueError('Fixed evaluation budget required')
    for key in ('retained_model','candidate_model','validation_cases','known_cases'):
        if not Path(study[key]).is_file():raise FileNotFoundError(study[key])


def run(study):
    check(study);report=Path(study['report_root']);data=Path(study['data_root'])
    if report.exists() or data.exists():raise FileExistsError('Refusing duplicate evaluation/overwrite')
    report.mkdir(parents=True);data.mkdir(parents=True)
    parent=Path(study['parent_report']);old=read(parent/'manifest.json');config=old['configuration'];s=old['evaluation_settings']
    models={name:PPO.load(study[key],device='cpu') for name,key in [('retained','retained_model'),('candidate','candidate_model')]}
    hashes={name:sha(study[key]) for name,key in [('retained','retained_model'),('candidate','candidate_model')]}
    manifest={'study':study,'model_hashes':hashes,'configuration':config,'evaluation_settings':s,
              'training_performed':False,'automatic_model_switch':False,
              'code_hashes':{p:sha(p) for p in ['scripts/compare_static_candidates.py','scripts/run_targeted_bc.py',
                   'src/wall_panel_rl/rl/avoidance_curriculum.py','src/wall_panel_rl/rl/autonomous_dynamic_ppo.py']}}
    dump(report/'manifest.json',manifest)
    state={'status':'running','runner_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),'training_performed':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('COMPARE',json.dumps(state),flush=True)
    try:
        validation=[c for c in read(study['validation_cases']) if c['suite']=='stage_target']
        trace_sets={}
        for name,model in models.items():
            status('validation_diagnosis',model=name)
            trace_sets[name]=[diagnose(model,config,s,case) for case in validation]
            dump(report/f'{name}_validation_traces.json',trace_sets[name])
        outcomes={name:{t['seed']:t['reason'] for t in traces} for name,traces in trace_sets.items()}
        timeouts=[t for t in trace_sets['candidate'] if t['reason']=='timeout']
        details=[]
        for t in timeouts:
            detail=stall_summary(t);seed=t['seed'];detail['retained_reason']=outcomes['retained'][seed]
            case=next(c for c in validation if c['scene']['seed']==seed)
            extended=diagnose(models['candidate'],config,s,case,study['diagnostic_horizon'])
            # Same physical prefix and observation/reference inputs at step 239.
            assert np.allclose(extended['states'][239]['q'],t['states'][239]['q'],atol=1e-8,rtol=0)
            detail.update(extended_reason=extended['reason'],extended_steps=extended['steps'],
                          extended_position_m=extended['position_m'],extended_orientation_deg=extended['orientation_deg'])
            details.append(detail);dump(report/f'extended_timeout_{seed}.json',extended)
        comparison={'validation_transitions':dict(Counter(outcomes['retained'][seed]+' -> '+outcomes['candidate'][seed]
            for seed in outcomes['retained'])),'candidate_timeouts':details,'model_switch_performed':False}
        dump(report/'timeout_comparison.json',comparison)
        known=read(study['known_cases'])
        forbidden={c['scene']['seed'] for c in read(study['validation_cases'])+known}
        for path in ['data/staged_avoidance_v3_seed61/one_static/train.json']:
            forbidden|={c['seed'] for c in read(path)}
        for path in ['data/targeted_bc_v3_seed61/independent_cases.json','data/stall_recovery_v3_seed61/independent_cases.json']:
            forbidden|={c['scene']['seed'] for c in read(path)}
        fresh=make_test_bank(config,s,study,data/'fresh_cases.json',forbidden,status)
        results={}
        for bank,cases in [('known',known),('fresh',fresh)]:
            results[bank]={}
            for name,model in models.items():
                status('expanded_evaluation',bank=bank,model=name)
                results[bank][name]=score_only(model,config,s,cases)
                results[bank][name].update(independent_test=bank=='fresh',used_for_selection=False)
                dump(report/f'{bank}_{name}.json',results[bank][name])
            dump(report/f'{bank}_paired.json',paired_outcomes(results[bank]['retained']['rows'],results[bank]['candidate']['rows']))
        combined={name:{key:sum(results[bank][name]['metrics']['stage_target'][key] for bank in results)
                       for key in ('episodes','success','collision','timeout','joint_limit')} for name in models}
        assert hashes['retained']==sha(study['retained_model']) and hashes['candidate']==sha(study['candidate_model'])
        dump(report/'summary.json',{'validation_timeout_comparison':comparison,
            'results':{bank:{name:r['metrics'] for name,r in groups.items()} for bank,groups in results.items()},
            'combined_200_case_descriptive_results':combined,'combined_is_pure_independent_test':False,
            'training_performed':False,'model_switch_performed':False,'model_hashes_unchanged':True})
        status('complete',status='completed',finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/static_candidate_comparison.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
