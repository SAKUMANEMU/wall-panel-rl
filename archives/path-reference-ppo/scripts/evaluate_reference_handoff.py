"""Fixed-weight inference ablation of cursor recovery and terminal servo range.

Changes controller behavior/inputs, not network tensors. No PPO/BC fitting,
new planner, model-file overwrite, combined-factor trial or automatic adoption.
"""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha,score_only
from compare_static_candidates import diagnose,stall_summary
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv


def tensor_fingerprint(policy):
    digest=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def apply_variant(model,settings,variant):
    s=copy.deepcopy(settings);s['reference_recovery']['radius']=float(variant['recovery_radius'])
    s['terminal_radius']=float(variant['terminal_radius'])
    # Both reference_mean and correction_components read this scalar. The
    # serialized network tensors and the on-disk checkpoint remain untouched.
    model.policy.terminal_radius=s['terminal_radius']
    return s


def summarize_rows(rows):
    metrics={}
    for suite in sorted({row['suite'] for row in rows}):
        group=[r for r in rows if r['suite']==suite]
        metrics[suite]={'episodes':len(group),**{k:sum(r['reason']==k for r in group)
            for k in ('success','collision','timeout','joint_limit')}}
    return metrics


def paired(rows,baseline):
    a={r['seed']:r for r in baseline if r['suite']=='stage_target'}
    b={r['seed']:r for r in rows if r['suite']=='stage_target'}
    if a.keys()!=b.keys():raise ValueError('Unmatched comparison cases')
    return dict(Counter(a[seed]['reason']+' -> '+b[seed]['reason'] for seed in a))


def goal_probes(model,config,s,case):
    env=StagedAvoidanceEnv(config,[],s);rows=[]
    try:
        for present in (False,True):
            scene=copy.deepcopy(case['scene']);goal=config['robot']['goal_q']
            scene.update(initial_q=goal,reference_joint_path=[goal,goal],obstacle_motion=[],
                         obstacles=[{'center':[5.,5.,5.],'dimensions':[.1,.1,.1]}] if present else [])
            obs,info=env.reset(options={'scenario':scene,'task':'static','strength':[.05,.02]})
            rows.append({'obstacle_present':present,'initial_success':bool(info['success']),
                         'max_action':float(np.max(np.abs(model.predict(obs,deterministic=True)[0])))})
    finally:env.close()
    return rows


def check(study):
    expected={'baseline':{'recovery_radius':.3,'terminal_radius':.2},
              'recovery_wider':{'recovery_radius':.45,'terminal_radius':.2},
              'terminal_earlier':{'recovery_radius':.3,'terminal_radius':.4}}
    if study['variants']!=expected:raise ValueError('Exactly baseline and two separate one-factor variants required')
    for key in ('model','validation_cases'):
        if not Path(study[key]).is_file():raise FileNotFoundError(study[key])
    for path in study['regression_banks'].values():
        if not Path(path).is_file():raise FileNotFoundError(path)
    if sha(study['model'])!=read(Path(study['parent_report'])/'summary.json')['export_sha256']:
        raise ValueError('Not the currently retained checkpoint')


def run(study):
    check(study);report=Path(study['report_root'])
    if report.exists():raise FileExistsError('Refusing repeat evaluation/overwrite')
    report.mkdir(parents=True)
    parent=read(Path(study['parent_report'])/'manifest.json');config=parent['configuration'];settings=parent['evaluation_settings']
    cases=[c for c in read(study['validation_cases']) if c['suite']!='five_full_diagnostic']
    targets=[c for c in cases if c['suite']=='stage_target']
    banks={name:read(path) for name,path in study['regression_banks'].items()}
    target_seeds=[{c['scene']['seed'] for c in bank if c['suite']=='stage_target'} for bank in banks.values()]
    if target_seeds[0]&target_seeds[1]:raise ValueError('Regression banks overlap')
    file_hash=sha(study['model']);base=PPO.load(study['model'],device='cpu');fingerprint=tensor_fingerprint(base.policy)
    manifest={'study':study,'configuration':config,'original_settings':settings,'model_file_sha256':file_hash,
        'policy_tensor_sha256':fingerprint,'training_performed':False,
        'behavior_and_observation_rules_changed_not_learned_weights':True,
        'regression_cases_are_already_seen':True,'automatic_adoption':False,
        'case_hashes':{p:sha(p) for p in [study['validation_cases'],*study['regression_banks'].values()]},
        'code_hashes':{p:sha(p) for p in ['scripts/evaluate_reference_handoff.py','scripts/compare_static_candidates.py',
            'scripts/run_targeted_bc.py','src/wall_panel_rl/rl/avoidance_curriculum.py',
            'src/wall_panel_rl/rl/autonomous_dynamic_ppo.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    state={'status':'running','runner_pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),
           'training_performed':False,'automatic_adoption':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('HANDOFF',json.dumps(state),flush=True)
    results={};all_rows={}
    try:
        focus={900002,900003,900005,900015}
        for name,variant in study['variants'].items():
            model=PPO.load(study['model'],device='cpu');s=apply_variant(model,settings,variant)
            assert tensor_fingerprint(model.policy)==fingerprint
            dump(report/name/'settings.json',s)
            status('validation',variant=name)
            evaluation=score_only(model,config,s,cases)
            dump(report/name/'validation.json',evaluation)
            # Failure-focused traces use the same retained weights in every arm.
            traces=[diagnose(model,config,s,c) for c in targets if c['scene']['seed'] in focus]
            dump(report/name/'focus_traces.json',traces)
            focus_rows=[{'seed':t['seed'],'reason':t['reason'],'steps':t['steps'],
                         'position_m':t['position_m'],'orientation_deg':t['orientation_deg'],
                         'last40':stall_summary(t)} for t in traces]
            dump(report/name/'focus_summary.json',focus_rows)
            probes=goal_probes(model,config,s,cases[0]);dump(report/name/'goal_probes.json',probes)
            results[name]={'parameters':variant,'validation':evaluation['metrics'],'regression':{},
                           'focus':focus_rows,'goal_probes':probes}
            all_rows[name]={'validation':evaluation['rows'],'regression':{}}
            for bank,bank_cases in banks.items():
                status('regression',variant=name,bank=bank)
                scores=score_only(model,config,s,bank_cases)
                scores.update(independent_test=False,already_seen_bank=True,used_for_training=False)
                dump(report/name/f'{bank}.json',scores)
                results[name]['regression'][bank]=scores['metrics']
                all_rows[name]['regression'][bank]=scores['rows']
                status('regression',variant=name,bank=bank,target=scores['metrics']['stage_target'])
            results[name]['combined_200']={key:sum(r['stage_target'][key] for r in results[name]['regression'].values())
                for key in ('episodes','success','collision','timeout','joint_limit')}
            assert tensor_fingerprint(model.policy)==fingerprint
            results[name]['policy_tensors_unchanged']=True
        # Reproduce the existing retained baseline before interpreting variants.
        baseline=results['baseline']
        if baseline['combined_200']!={'episodes':200,'success':182,'collision':10,'timeout':8,'joint_limit':0}:
            raise RuntimeError('Baseline failed to reproduce existing paired regression counts')
        for name in results:
            results[name]['paired_validation']=paired(all_rows[name]['validation'],all_rows['baseline']['validation'])
            results[name]['paired_regression']={bank:paired(all_rows[name]['regression'][bank],all_rows['baseline']['regression'][bank]) for bank in banks}
            eligible=True
            for group in ('validation','combined_200'):
                candidate=results[name][group]['stage_target'] if group=='validation' else results[name][group]
                original=baseline[group]['stage_target'] if group=='validation' else baseline[group]
                eligible &= candidate['success']>=original['success'] and candidate['collision']<=original['collision'] and candidate['timeout']<=original['timeout'] and candidate['joint_limit']==0
            for metrics in [results[name]['validation'],*results[name]['regression'].values()]:
                for suite in ('alignment_far','alignment_near'):
                    eligible &= metrics[suite]['success']==metrics[suite]['episodes'] and metrics[suite]['collision']==0 and metrics[suite]['joint_limit']==0
            eligible &= all(p['initial_success'] and p['max_action']<=settings['bc_gate']['maximum_goal_action'] for p in results[name]['goal_probes'])
            results[name]['non_regression_eligible']=bool(eligible)
        assert sha(study['model'])==file_hash
        dump(report/'summary.json',{'variants':results,'training_performed':False,'model_file_unchanged':True,
            'network_tensors_unchanged':True,'default_configuration_changed':False,
            'controller_hyperparameters_changed_during_inference_only':True,
            'combined_200_are_regression_not_new_independent_test':True,'automatic_adoption':False})
        status('complete',status='completed',finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/reference_handoff_ablation.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
