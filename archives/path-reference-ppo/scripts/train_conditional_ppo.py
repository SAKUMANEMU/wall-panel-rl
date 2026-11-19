"""One 10k static PPO adaptation with matched conditional-recovery evaluation."""
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
from run_targeted_bc import read,dump,sha,score_only
from run_stall_recovery import run_ppo
from run_clearance_ppo import CollisionValidation
from evaluate_conditional_recovery import score as conditional_rollouts
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv,ConditionalTrainingEnv
from wall_panel_rl.rl.avoidance_curriculum import gate_result,assess_history


def inputs(study):
    if study['ppo_steps']!=10000 or study['ppo_eval_every']!=5000 or study['ppo_learning_rate']!=1e-4:
        raise ValueError('Exactly one 10k adaptation with unchanged LR and 5k checks')
    report=Path(study['conditional_report']);manifest=read(report/'manifest.json')
    if not read(report/'non_regression.json')['passed']:raise ValueError('Conditional recovery was not validated')
    if sha(study['initial_model'])!=manifest['model_sha256']:raise ValueError('Wrong source checkpoint')
    training=copy.deepcopy(read(study['training_manifest'])['training_settings'])
    evaluation=copy.deepcopy(manifest['settings']);config=copy.deepcopy(manifest['configuration'])
    training['conditional_recovery']=copy.deepcopy(evaluation['conditional_recovery'])
    if training['wall_reward']!={'distance':.02,'weight':1.} or training['terminal_radius']!=.2 or training['reference_recovery']['radius']!=.3:
        raise ValueError('Existing reward/terminal/reference settings changed')
    if training['stall_reward']!=evaluation['stall_reward'] or config['simulation']['max_steps']!=240:
        raise ValueError('Stall feedback or horizon changed')
    return training,evaluation,config


def score_model(candidate,source,config,settings,cases):
    evaluated=conditional_rollouts(candidate,config,settings,cases,ConditionalRecoveryEnv)
    metrics=evaluated['metrics'];gate=gate_result(metrics,settings['bc_gate'])
    gate['checks']['alignment_collision_free']=all(metrics[k]['collision']==0 and metrics[k]['joint_limit']==0
        for k in ('alignment_far','alignment_near'))
    env=ConditionalRecoveryEnv(config,[],settings);probes=[]
    try:
        for present in (False,True):
            scene=copy.deepcopy(cases[0]['scene']);goal=config['robot']['goal_q']
            scene.update(initial_q=goal,reference_joint_path=[goal,goal],obstacle_motion=[],
                obstacles=[{'center':[5.,5.,5.],'dimensions':[.1,.1,.1]}] if present else [])
            obs,info=env.reset(options={'scenario':scene,'task':'static','strength':[.05,.02]})
            probes.append({'obstacle_present':present,'initial_success':bool(info['success']),
                           'max_action':float(np.max(np.abs(candidate.predict(obs,deterministic=True)[0])))})
    finally:env.close()
    gate['checks']['goal_convergence']=all(p['initial_success'] and p['max_action']<=settings['bc_gate']['maximum_goal_action'] for p in probes)
    gate['passed']=all(gate['checks'].values())
    gate.update(metrics={'dynamic_ppo':metrics},goal_probes=probes,training_performed=False,
                execution_wrapper='ConditionalRecoveryEnv',evaluated_with_training_shaping=False)
    return gate,evaluated['rows']


class ConditionalValidation(CollisionValidation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.sampling={'observed_transitions':0,'recovery_attempt_events':0,'recovery_success_events':0,
                       'stall_penalty_events':0,'stall_penalty_sum':0.,'wall_penalty_sum':0.}

    def score_model(self,candidate):
        return score_model(candidate,self.source,self.config,self.s,self.cases)

    def _on_step(self):
        for info in self.locals.get('infos',[]):
            self.sampling['observed_transitions']+=1
            self.sampling['recovery_attempt_events']+=int(info.get('conditional_recovery_attempted',False))
            self.sampling['recovery_success_events']+=int(info.get('conditional_recovery_triggered',False))
            self.sampling['stall_penalty_events']+=int(info.get('reward_stall',0.)<0)
            self.sampling['stall_penalty_sum']+=float(info.get('reward_stall',0.))
            self.sampling['wall_penalty_sum']+=float(info.get('reward_wall_clearance',0.))
        return True

    def _on_rollout_end(self):
        super()._on_rollout_end();dump(self.output/'sampling_events.json',self.sampling)


def worker(config,scenes,settings,rank,output):
    torch.set_num_threads(1)
    return Monitor(ConditionalTrainingEnv(config,scenes,settings,rank),str(output/f'worker_{rank}.monitor.csv'),
        info_keywords=('termination_reason','curriculum_task','conditional_recovery_attempts',
                       'conditional_recovery_count','reward_stall','reward_wall_clearance'))


def run(study):
    training,evaluation,config=inputs(study);root=Path(study['output_root']);report=Path(study['report_root'])
    if root.exists() or report.exists():raise FileExistsError('Refusing repeated adaptation/overwrite')
    report.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'runner_pid':os.getpid(),
           'additional_budget':10000,'automatic_stage_advance':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('ADAPT',json.dumps(state),flush=True)
    bank=read(Path(study['base_scenes'])/'train.json')
    held=set(read('/home/johnsonchen/wall-panel-rl/results/diagnostics/clearance_ppo_v3_seed61/collision_diagnosis.json')['heldout_scene_seeds'])
    scenes=[c for c in bank if c['seed'] not in held]
    case_path=Path(study['base_scenes'])/'validation_staged.json'
    cases=[c for c in read(case_path) if c['suite']!='five_full_diagnostic']
    manifest={'study':study,'configuration':config,'training_settings':training,'evaluation_settings':evaluation,
        'initial_model_sha256':sha(study['initial_model']),'training_wrapper':'ConditionalTrainingEnv',
        'evaluation_wrapper':'ConditionalRecoveryEnv','extra_training_rewards_excluded_from_evaluation':True,
        'training_contract':'reference_preserving_v3_plus_conditional_recovery_v1',
        'regression_used_for_selection':False,'automatic_stage_advance':False,
        'input_hashes':{str(p):sha(p) for p in [Path(study['base_scenes'])/'train.json',case_path,*map(Path,study['regression_banks'].values())]},
        'code_hashes':{p:sha(p) for p in ['scripts/train_conditional_ppo.py','scripts/run_stall_recovery.py',
            'scripts/run_clearance_ppo.py','scripts/evaluate_conditional_recovery.py',
            'src/wall_panel_rl/rl/conditional_recovery.py','src/wall_panel_rl/rl/wall_feedback.py',
            'src/wall_panel_rl/rl/stall_recovery.py','src/wall_panel_rl/rl/avoidance_curriculum.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    dump(report/'training_settings.json',training)
    try:
        base=PPO.load(study['initial_model'],device='cpu');source=PPO.load(training['source_model'],device='cpu')
        status('preflight')
        original=score_only(base,config,evaluation,cases);conditional,_=score_model(base,source,config,evaluation,cases)
        expected=read(Path(study['conditional_report'])/'summary.json')['variants']['conditional']['validation']['stage_target']
        actual=conditional['metrics']['dynamic_ppo']['stage_target']
        if actual!=expected:raise RuntimeError('Conditional controller baseline did not reproduce')
        dump(report/'preflight_comparison.json',{'same_weights_original_controller':original['metrics'],
            'same_weights_conditional_controller':conditional['metrics']['dynamic_ppo']})
        trial=copy.deepcopy(study);trial['stall_reward']=training['stall_reward']
        status('ppo',ppo_additional_steps=0)
        path,selection=run_ppo(trial,config,training,scenes,cases,source,status,
            evaluation_settings=evaluation,callback_class=ConditionalValidation,worker_factory=worker,initial_scorer=score_model)
        dump(report/'selection.json',selection)
        sampling=read(root/'ppo/sampling_events.json')
        if sampling['observed_transitions']!=10000:raise AssertionError('Unexpected rollout accounting')
        history=[json.loads(line) for line in (root/'ppo/validation.jsonl').read_text().splitlines()]
        assessment=assess_history(history,training['gate'])
        assessment['selected_model_gate']=gate_result(selection['metrics']['dynamic_ppo'],training['gate'])
        assessment['passed']=assessment['passed'] and assessment['selected_model_gate']['passed']
        assessment.update(execution_wrapper='ConditionalRecoveryEnv',selected_model_sha256=sha(path),automatic_stage_advance=False)
        dump(report/'promotion_assessment.json',assessment)
        regression={}
        for name,model_path in [('conditional_before',Path(study['initial_model'])),('conditional_selected',path)]:
            model=PPO.load(model_path,device='cpu');regression[name]={}
            for key,case_file in study['regression_banks'].items():
                status('regression',model=name,bank=key)
                result=conditional_rollouts(model,config,evaluation,read(case_file),ConditionalRecoveryEnv)
                result.update(independent_test=False,already_seen_bank=True,used_for_selection=False)
                dump(report/name/f'{key}.json',result);regression[name][key]=result['metrics']
        combined={name:{key:sum(bank['stage_target'][key] for bank in banks.values()) for key in
                  ('episodes','success','collision','timeout','joint_limit')} for name,banks in regression.items()}
        shutil.copy2(path,report/'selected_model.zip')
        shutil.copy2(root/'ppo/final_model.zip',report/'final_candidate.zip')
        execution={'model_file':'selected_model.zip','model_sha256':sha(report/'selected_model.zip'),
            'wrapper_module':'wall_panel_rl.rl.conditional_recovery','wrapper_class':'ConditionalRecoveryEnv',
            'configuration':config,'evaluation_settings':evaluation,'training_wrapper_class':'ConditionalTrainingEnv'}
        dump(report/'controller_config.json',execution)
        dump(report/'summary.json',{'completed_additional_steps':10000,'selected_additional_steps':selection['additional_steps'],
            'preflight':read(report/'preflight_comparison.json'),
            'history':[{'kind':e['kind'],'additional_steps':e['additional_steps'],'metrics':e['metrics']['dynamic_ppo']} for e in history],
            'selected_validation':selection['metrics']['dynamic_ppo'],'promotion_assessment':assessment,
            'regression':regression,'combined_300_regression':combined,'regression_is_independent_test':False,
            'sampling_events':sampling,'export_sha256':sha(report/'selected_model.zip'),
            'execution_wrapper':'ConditionalRecoveryEnv','automatic_stage_advance':False})
        status('complete',status='completed',promotion_passed=assessment['passed'],finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/conditional_ppo_adaptation.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());inputs(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
