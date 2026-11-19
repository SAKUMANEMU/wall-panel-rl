"""Matched PPO continuations: original feedback versus wall-clearance feedback.

Only --run launches two exactly-10k pilots. Both start from the same latest PPO,
including its optimizer. Test is frozen first and scored after selections.
"""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from run_stall_recovery import PilotValidation,run_ppo
from run_targeted_bc import read,dump,sha,make_test_bank,score_only
from wall_panel_rl.rl.wall_feedback import WallAwareEnv
from wall_panel_rl.rl.avoidance_curriculum import assess_history,gate_result


def collision_score(result,baseline):
    """Require no regression in success/collisions/timeouts, then favor clearance."""
    target=result['metrics']['dynamic_ppo']['stage_target']
    checks=result['checks']
    retained=all(checks[k] for k in ('alignment_samples','alignment_retained','alignment_collision_free','goal_convergence'))
    eligible=(result['passed'] and retained and target['episodes']==baseline['episodes'] and
        target['success']>=baseline['success'] and target['collision']<=baseline['collision'] and
        target['timeout']<=baseline['timeout'] and target['joint_limit']==0)
    return (int(eligible),-target['collision'],target['success'],-target['timeout'],-target['joint_limit'])


class CollisionValidation(PilotValidation):
    def candidate_score(self,result):
        if not hasattr(self,'baseline_target'):
            self.baseline_target=copy.deepcopy(result['metrics']['dynamic_ppo']['stage_target'])
        return collision_score(result,self.baseline_target)


def worker(config,scenes,s,rank,output):
    torch.set_num_threads(1)
    return Monitor(WallAwareEnv(config,scenes,s,rank),str(output/f'worker_{rank}.monitor.csv'),
        info_keywords=('termination_reason','curriculum_task','reward_stall','reward_wall_clearance'))


def check(study):
    if study['ppo_steps']!=10000 or study['ppo_eval_every']!=5000:raise ValueError('Two bounded 10k PPO pilots required')
    if study['variants']!={'control':0.,'wall_feedback':1.}:raise ValueError('Only wall weight may differ: 0 versus 1')
    if not 0<study['wall_distance']<.025:raise ValueError('Wall distance must preserve nominal target gap')
    if not Path(study['initial_model']).is_file():raise FileNotFoundError(study['initial_model'])
    if study['ppo_learning_rate']!=1e-4:raise ValueError('Preserve learning rate')


def run(study):
    check(study);root=Path(study['output_root']);data=Path(study['data_root']);report=Path(study['report_root'])
    parent=Path(study['parent_report'])
    if root.exists() or data.exists() or (report/'status.json').exists():raise FileExistsError('Refusing duplicate run')
    diagnosis=read(report/'collision_diagnosis.json');config=diagnosis['configuration'];s=diagnosis['settings']
    s=copy.deepcopy(s);s['learning_rate']=study['ppo_learning_rate'];s['eval_every']=study['ppo_eval_every']
    initial=PPO.load(study['initial_model'],device='cpu');selected=PPO.load(parent/'selected_model.zip',device='cpu')
    if diagnosis['model_sha256']!=sha(parent/'selected_model.zip') or any(
        not torch.equal(v,initial.policy.state_dict()[k]) for k,v in selected.policy.state_dict().items()):
        raise ValueError('Diagnosis/initial model mismatch')
    if config['simulation']['max_steps']!=240 or s['clearance_weight']!=.5:raise ValueError('Physical/base-reward change')
    if config['target']['wall_clearance']<=study['wall_distance']:raise ValueError('Goal enters wall penalty band')
    root.mkdir(parents=True);data.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'runner_pid':os.getpid(),
           'ppo_budget_per_variant':10000,'variants':list(study['variants']),'automatic_stage_advance':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('CLEARANCE',json.dumps(state),flush=True)
    manifest={'study':study,'configuration':config,'evaluation_settings':s,
        'initial_checkpoint_sha256':sha(study['initial_model']),'diagnosis_sha256':sha(report/'collision_diagnosis.json'),
        'same_initial_actor_and_optimizer':True,'same_fresh_sampling_seed':s['seed'],
        'only_training_wall_weight_differs':True,'test_used_for_selection':False,
        'selection':'no regression in validation success/collision/timeout, then lower collision',
        'code_hashes':{p:sha(p) for p in ['scripts/run_clearance_ppo.py','scripts/run_stall_recovery.py',
            'src/wall_panel_rl/rl/wall_feedback.py','src/wall_panel_rl/rl/stall_recovery.py',
            'src/wall_panel_rl/rl/avoidance_curriculum.py','scripts/staged_avoidance.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    try:
        bank=read(Path(study['base_scenes'])/'train.json');held=set(diagnosis['heldout_scene_seeds'])
        train=[c for c in bank if c['seed'] not in held]
        if {t['seed'] for t in diagnosis['traces']['train']}!={c['seed'] for c in train}:raise ValueError('Training layout mismatch')
        cases=read(Path(study['base_scenes'])/'validation_staged.json')
        forbidden={c['seed'] for c in bank}|{c['scene']['seed'] for c in cases}
        for path in ['data/targeted_bc_v3_seed61/independent_cases.json','data/stall_recovery_v3_seed61/independent_cases.json']:
            forbidden|={c['scene']['seed'] for c in read(path)}
        tests=make_test_bank(config,s,study,data/'independent_cases.json',forbidden,status)
        source=PPO.load(s['source_model'],device='cpu');candidates={};assessments={}
        for name,weight in study['variants'].items():
            variant=copy.deepcopy(s);variant['wall_reward']={'distance':study['wall_distance'],'weight':weight}
            trial=copy.deepcopy(study);trial['output_root']=str(root/name);trial['report_root']=str(report/name)
            trial['stall_reward']=s['stall_reward']
            dump(report/name/'training_settings.json',variant)
            def variant_status(phase,**extra):status(phase,variant=name,**extra)
            status('train',variant=name,ppo_additional_steps=0)
            path,selection=run_ppo(trial,config,variant,train,cases,source,variant_status,
                evaluation_settings=s,callback_class=CollisionValidation,worker_factory=worker)
            candidates[name]=(path,selection['gate']);dump(report/f'{name}_selection.json',selection)
            history=[json.loads(line) for line in (root/name/'ppo/validation.jsonl').read_text().splitlines()]
            assessment=assess_history(history,s['gate'])
            assessment['selected_model_gate']=gate_result(selection['metrics']['dynamic_ppo'],s['gate'])
            assessment['passed']=assessment['passed'] and assessment['selected_model_gate']['passed']
            assessments[name]=assessment;dump(report/f'{name}_promotion_assessment.json',assessment)
        # The first control event is the original checkpoint's common evaluation.
        baseline_event=json.loads((root/'control/ppo/validation.jsonl').read_text().splitlines()[0])
        baseline=baseline_event['gate'];target=baseline['metrics']['dynamic_ppo']['stage_target']
        candidates={'baseline':(Path(study['initial_model']),baseline),**candidates}
        winner=max(candidates,key=lambda name:collision_score(candidates[name][1],target))
        status('independent_test',winner_locked=winner)
        independent={}
        for name,(path,result) in candidates.items():
            independent[name]=score_only(PPO.load(path,device='cpu'),config,s,tests)
            dump(report/f'independent_{name}.json',independent[name])
            status('independent_test',evaluated_model=name,independent_target=independent[name]['metrics']['stage_target'])
        shutil.copy2(candidates[winner][0],report/'selected_model.zip')
        dump(report/'summary.json',{'winner_by_validation':winner,'validation':{k:v[1] for k,v in candidates.items()},
            'independent':{k:v['metrics'] for k,v in independent.items()},'promotion_assessments':assessments,
            'ppo_steps_per_variant':10000,'selected_additional_steps':{k:read(report/f'{k}_selection.json')['additional_steps'] for k in study['variants']},
            'export_sha256':sha(report/'selected_model.zip'),'automatic_stage_advance':False})
        status('complete',status='completed',finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/clearance_ppo_study.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
