"""Fixed-weight original versus stagnation-triggered reference recovery.

No training/default adoption. Fresh 100-scene test is bounded and runs only if
the predeclared original validation + 200-case regression checks pass.
"""
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
from run_targeted_bc import read,dump,sha,make_test_bank
from evaluate_reference_handoff import tensor_fingerprint,summarize_rows,paired
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv


def score(model,config,settings,cases,factory,trace_seeds=()):
    env=factory(config,[],settings);rows=[];traces=[]
    try:
        for case in cases:
            obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            capture=case['suite']=='stage_target' and case['scene']['seed'] in trace_seeds;states=[]
            while True:
                action=model.predict(obs,deterministic=True)[0]
                if capture:
                    q,_=env.env.joint_state();x=torch.as_tensor(obs[None])
                    with torch.no_grad():
                        prior=model.policy.reference_mean(x);strength,gate,residual=model.policy.correction_components(x)
                        base=(1-strength+strength*gate)*prior;correction=strength*residual
                    state={'step':env.env.steps,'q':q.tolist(),'reference':env.ref.tolist(),
                        'reference_index':env.cursor.index,'reference_distance_rad':float(np.linalg.norm(env.ref-q)),
                        'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                        'action':action.tolist(),'action_norm':float(np.linalg.norm(action)),
                        'cancellation_ratio':float(np.linalg.norm(action)/(float(torch.linalg.vector_norm(base)+torch.linalg.vector_norm(correction))+1e-9))}
                obs,_,done,truncated,info=env.step(action)
                if capture:
                    state.update(recovered=bool(info.get('conditional_recovery_triggered',False)),
                        next_reference_index=env.cursor.index,executed_action=info['executed_action'])
                    states.append(state)
                if done or truncated:break
            row={'suite':case['suite'],'seed':case['scene']['seed'],'reason':info['termination_reason'],'steps':env.env.steps,
                 'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                 'conditional_attempts':int(info.get('conditional_recovery_attempts',0)),
                 'conditional_recoveries':int(info.get('conditional_recovery_count',0)),
                 'events':copy.deepcopy(getattr(env,'conditional_events',[])),'online_avoidance_planner':False}
            rows.append(row)
            if capture:traces.append({**row,'states':states})
    finally:env.close()
    return {'metrics':summarize_rows(rows),'rows':rows,'traces':traces,'training_performed':False}


def compare_unchanged_untriggered(baseline,treatment):
    original={(r['suite'],r['seed']):r for r in baseline}
    unchanged=0;triggered=0
    for row in treatment:
        base=original[(row['suite'],row['seed'])]
        if row['conditional_recoveries']:
            triggered+=1;continue
        if row['reason']!=base['reason'] or row['steps']!=base['steps'] or not np.isclose(row['position_m'],base['position_m'],atol=1e-8,rtol=0):
            raise AssertionError(f'Untriggered episode changed: {row["suite"]} {row["seed"]}')
        unchanged+=1
    return {'identical_untriggered_episodes':unchanged,'triggered_episodes':triggered}


def non_regression(results):
    base=results['baseline'];candidate=results['conditional'];checks={}
    for group in ('validation','combined_200'):
        a=base[group]['stage_target'] if group=='validation' else base[group]
        b=candidate[group]['stage_target'] if group=='validation' else candidate[group]
        checks[group]=b['success']>=a['success'] and b['collision']<=a['collision'] and b['timeout']<=a['timeout'] and b['joint_limit']==0
    checks['alignment_retained']=all(m[suite]['success']==m[suite]['episodes'] and m[suite]['collision']==0 and m[suite]['joint_limit']==0
        for m in [candidate['validation'],*candidate['regression'].values()] for suite in ('alignment_far','alignment_near'))
    return {'passed':all(checks.values()),'checks':checks}


def check(study):
    expected={'window':30,'min_steps':60,'position_epsilon':.001,'orientation_epsilon_deg':.1,
              'max_joint_displacement':.02,'min_reference_distance':.2,'extended_radius':.45,
              'max_attempts':3,'max_recoveries':1,'cooldown':20}
    if study['conditional_recovery']!=expected:raise ValueError('Single predeclared recovery experiment required')
    if study['independent_scenes']!=100:raise ValueError('Fresh test is bounded at 100 scenes')
    if sha(study['model'])!=read(Path(study['parent_report'])/'summary.json')['export_sha256']:
        raise ValueError('Not the retained PPO checkpoint')


def run(study):
    check(study);report=Path(study['report_root']);data=Path(study['data_root'])
    if report.exists() or data.exists():raise FileExistsError('Refusing duplicate evaluation/overwrite')
    report.mkdir(parents=True);data.mkdir(parents=True)
    parent=read(Path(study['parent_report'])/'manifest.json');config=parent['configuration'];settings=copy.deepcopy(parent['evaluation_settings'])
    settings['conditional_recovery']=study['conditional_recovery']
    assert settings['reference_recovery']['radius']==.3 and settings['terminal_radius']==.2 and config['simulation']['max_steps']==240
    cases=[c for c in read(study['validation_cases']) if c['suite']!='five_full_diagnostic']
    banks={name:read(path) for name,path in study['regression_banks'].items()}
    model=PPO.load(study['model'],device='cpu');file_hash=sha(study['model']);fingerprint=tensor_fingerprint(model.policy)
    assert model.policy.terminal_radius==.2
    manifest={'study':study,'configuration':config,'settings':settings,'model_sha256':file_hash,'tensor_fingerprint':fingerprint,
        'training_performed':False,'default_configuration_changed':False,'automatic_adoption':False,
        'fresh_test_only_after_regression_pass':True,
        'case_hashes':{p:sha(p) for p in [study['validation_cases'],*study['regression_banks'].values()]},
        'code_hashes':{p:sha(p) for p in ['scripts/evaluate_conditional_recovery.py',
            'src/wall_panel_rl/rl/conditional_recovery.py','src/wall_panel_rl/rl/avoidance_curriculum.py',
            'src/wall_panel_rl/rl/autonomous_dynamic_ppo.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    state={'status':'running','runner_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),
           'training_performed':False,'automatic_adoption':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('CONDITIONAL',json.dumps(state),flush=True)
    results={};scores={}
    try:
        for name,factory in [('baseline',StagedAvoidanceEnv),('conditional',ConditionalRecoveryEnv)]:
            status('validation',variant=name)
            evaluation=score(model,config,settings,cases,factory,trace_seeds={900002,900003,900005,900015})
            dump(report/name/'validation.json',evaluation)
            results[name]={'validation':evaluation['metrics'],'regression':{}}
            scores[name]={'validation':evaluation}
            for bank,bank_cases in banks.items():
                status('regression',variant=name,bank=bank)
                evaluation=score(model,config,settings,bank_cases,factory)
                dump(report/name/f'{bank}.json',evaluation)
                results[name]['regression'][bank]=evaluation['metrics'];scores[name][bank]=evaluation
                status('regression',variant=name,bank=bank,target=evaluation['metrics']['stage_target'])
            results[name]['combined_200']={k:sum(v['stage_target'][k] for v in results[name]['regression'].values())
                for k in ('episodes','success','collision','timeout','joint_limit')}
            if name=='baseline' and results[name]['combined_200']!={'episodes':200,'success':182,'collision':10,'timeout':8,'joint_limit':0}:
                raise RuntimeError('Retained baseline did not reproduce')
            assert tensor_fingerprint(model.policy)==fingerprint and model.policy.terminal_radius==.2
        comparisons={}
        for bank in ('validation',*banks):
            a=scores['baseline'][bank]['rows'];b=scores['conditional'][bank]['rows']
            comparisons[bank]={'paired_outcomes':paired(b,a),'untriggered_check':compare_unchanged_untriggered(a,b)}
        gate=non_regression(results);dump(report/'non_regression.json',gate)
        fresh_results=None
        if gate['passed']:
            status('freeze_fresh_test')
            forbidden={c['scene']['seed'] for c in cases}
            for bank in banks.values():forbidden|={c['scene']['seed'] for c in bank}
            for path in ['data/targeted_bc_v3_seed61/independent_cases.json','data/stall_recovery_v3_seed61/independent_cases.json']:
                forbidden|={c['scene']['seed'] for c in read(path)}
            forbidden|={c['seed'] for c in read('data/staged_avoidance_v3_seed61/one_static/train.json')}
            fresh=make_test_bank(config,settings,study,data/'fresh_cases.json',forbidden,status)
            fresh_results={}
            for name,factory in [('baseline',StagedAvoidanceEnv),('conditional',ConditionalRecoveryEnv)]:
                status('fresh_test',variant=name)
                fresh_results[name]=score(model,config,settings,fresh,factory)
                dump(report/name/'fresh_test.json',fresh_results[name])
            a=fresh_results['baseline']['rows'];b=fresh_results['conditional']['rows']
            comparisons['fresh_test']={'paired_outcomes':paired(b,a),'untriggered_check':compare_unchanged_untriggered(a,b)}
        assert sha(study['model'])==file_hash and tensor_fingerprint(model.policy)==fingerprint
        dump(report/'summary.json',{'variants':results,'comparisons':comparisons,'non_regression':gate,
            'fresh_test':None if fresh_results is None else {k:v['metrics'] for k,v in fresh_results.items()},
            'fresh_test_skipped_reason':None if gate['passed'] else 'Known-case non-regression failed; no extra test generation',
            'model_file_unchanged':True,'network_tensors_unchanged':True,'terminal_radius_unchanged':True,
            'training_performed':False,'default_configuration_changed':False,'automatic_adoption':False})
        status('complete',status='completed',non_regression_passed=gate['passed'],finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/conditional_recovery_study.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
