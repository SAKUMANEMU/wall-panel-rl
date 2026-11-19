"""Exactly 10k static PPO continuation with unchanged settings and 5k/10k gates.

The old 100-case bank is a regression check, NOT a new independent test. Only
--run learns. Never start the next stage or extend the budget automatically.
"""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha,score_only
from run_stall_recovery import run_ppo
from run_clearance_ppo import CollisionValidation,worker
from wall_panel_rl.rl.avoidance_curriculum import assess_history,gate_result


def inputs(study):
    if study['ppo_steps']!=10000 or study['ppo_eval_every']!=5000:
        raise ValueError('Stability budget must be exactly 10k with 5k checks')
    parent=Path(study['parent_report']);manifest=read(parent/'manifest.json')
    training=read(parent/'wall_feedback/training_settings.json')
    evaluation=manifest['evaluation_settings'];config=manifest['configuration']
    if study['ppo_learning_rate']!=training['learning_rate'] or training['eval_every']!=study['ppo_eval_every']:
        raise ValueError('Learning/evaluation settings changed')
    if config['simulation']['max_steps']!=240 or training['active_stage']['mode']!='static':
        raise ValueError('Expected unchanged static stage')
    if training['wall_reward']!={'distance':.02,'weight':1.}:
        raise ValueError('Expected unchanged wall feedback')
    if training['stall_reward']!=read(parent/'selected_model.json')['training_stall_reward']:
        raise ValueError('Stall feedback changed')
    if sha(study['initial_model'])!=read(parent/'summary.json')['export_sha256']:
        raise ValueError('Starting checkpoint differs from last selected PPO')
    return training,evaluation,config


def run(study):
    training,evaluation,config=inputs(study)
    output=Path(study['output_root']);report=Path(study['report_root']);parent=Path(study['parent_report'])
    if output.exists() or report.exists():raise FileExistsError('Refusing duplicate run/overwrite')
    report.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'runner_pid':os.getpid(),
           'additional_budget':10000,'automatic_stage_advance':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('STABILITY',json.dumps(state),flush=True)
    bank=read(Path(study['base_scenes'])/'train.json')
    held=set(read(parent/'collision_diagnosis.json')['heldout_scene_seeds'])
    train=[c for c in bank if c['seed'] not in held]
    cases=read(Path(study['base_scenes'])/'validation_staged.json')
    manifest={'study':study,'initial_model_sha256':sha(study['initial_model']),
        'training_settings':training,'evaluation_settings':evaluation,'configuration':config,
        'settings_unchanged_from_previous_selected_variant':True,
        'input_hashes':{str(p):sha(p) for p in [Path(study['base_scenes'])/'train.json',
            Path(study['base_scenes'])/'validation_staged.json',Path(study['regression_cases'])]},
        'regression_used_for_selection':False,'code_hashes':{p:sha(p) for p in
            ['scripts/run_static_stability.py','scripts/run_stall_recovery.py','scripts/run_clearance_ppo.py',
             'src/wall_panel_rl/rl/wall_feedback.py','src/wall_panel_rl/rl/stall_recovery.py',
             'src/wall_panel_rl/rl/avoidance_curriculum.py','scripts/staged_avoidance.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    dump(report/'training_settings.json',training)
    try:
        source=PPO.load(training['source_model'],device='cpu')
        trial=copy.deepcopy(study);trial['stall_reward']=training['stall_reward']
        status('ppo',ppo_additional_steps=0)
        path,selection=run_ppo(trial,config,training,train,cases,source,status,
            evaluation_settings=evaluation,callback_class=CollisionValidation,worker_factory=worker)
        dump(report/'selection.json',selection)
        history=[json.loads(line) for line in (output/'ppo/validation.jsonl').read_text().splitlines()]
        baseline=history[0]['metrics']['dynamic_ppo']['stage_target']
        expected=read(parent/'selected_model.json')['validation']['stage_target']
        if any(baseline[k]!=expected[k] for k in ('episodes','success','collision','timeout','joint_limit')):
            raise RuntimeError('Baseline did not reproduce the previous validation outcomes')
        assessment=assess_history(history,training['gate'])
        assessment['selected_model_gate']=gate_result(selection['metrics']['dynamic_ppo'],training['gate'])
        assessment['passed']=assessment['passed'] and assessment['selected_model_gate']['passed']
        assessment.update(automatic_stage_advance=False,selected_model_sha256=sha(path))
        dump(report/'promotion_assessment.json',assessment)
        # Lock validation selection before replaying the already-seen test bank.
        status('regression_check',selected_additional_steps=selection['additional_steps'])
        regression={}
        frozen=read(study['regression_cases'])
        for name,model_path in [('baseline',Path(study['initial_model'])),('selected',path)]:
            regression[name]=score_only(PPO.load(model_path,device='cpu'),config,evaluation,frozen)
            regression[name].update(independent_test=False,already_seen_bank=True)
            dump(report/f'regression_{name}.json',regression[name])
            status('regression_check',evaluated_model=name,regression_target=regression[name]['metrics']['stage_target'])
        shutil.copy2(path,report/'selected_model.zip')
        dump(report/'summary.json',{'completed_additional_steps':10000,
            'selected_additional_steps':selection['additional_steps'],
            'baseline_validation':history[0]['metrics']['dynamic_ppo'],
            'selected_validation':selection['metrics']['dynamic_ppo'],
            'checkpoint_history':[{'kind':e['kind'],'additional_steps':e['additional_steps'],
                'metrics':e['metrics']['dynamic_ppo']} for e in history],
            'promotion_assessment':assessment,'regression':{k:v['metrics'] for k,v in regression.items()},
            'regression_is_independent_test':False,'export_sha256':sha(report/'selected_model.zip'),
            'automatic_stage_advance':False})
        status('complete',status='completed',promotion_passed=assessment['passed'],finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/static_stability_study.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());inputs(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
