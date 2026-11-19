"""50/50 RRT/PPO episodes, separate BC loss and on-policy PPO updates.

Default checks inputs only. --train starts the configured bounded experiment;
--evaluate replays frozen validation using PPO, RRT and the episode mixture.
"""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from run_targeted_bc import read,dump,sha
from select_safety_first_ppo import SafetyFirstValidation
from wall_panel_rl.rl.conditional_recovery import ConditionalTrainingEnv
from wall_panel_rl.rl.episode_imitation import ExpertEpisodeEnv,AuxiliaryImitation,evaluate_controller


def inputs(study):
    parent=read(Path(study['source_report'])/'manifest.json')
    if sha(study['initial_model'])!=parent['model_sha256']:raise ValueError('Wrong PPO source model')
    training=copy.deepcopy(parent['training_settings']);evaluation=copy.deepcopy(parent['evaluation_settings'])
    config=copy.deepcopy(parent['configuration'])
    if training['active_stage']['mode']!='static':raise ValueError('Validate this change in the static course first')
    training.update(eval_every=study['ppo_eval_every'])
    evaluation.update(eval_every=study['ppo_eval_every'])
    rollout=training['workers']*training['n_steps']
    if (study['ppo_steps']<=0 or study['ppo_steps']%rollout or study['ppo_eval_every']%rollout or
        not 0<=study['expert_probability']<1):raise ValueError('Invalid PPO budget, checkpoints or episode probability')
    held=set(read(study['heldout_scenes'])['heldout_scene_seeds'])
    scenes=[s for s in read(study['train_scenes']) if s['seed'] not in held]
    cases=[c for c in read(study['validation_cases']) if c['suite']!='five_full_diagnostic']
    if not scenes or not cases:raise ValueError('Empty training or validation bank')
    if {s['seed'] for s in scenes}&{c['scene']['seed'] for c in cases}:raise ValueError('Scene leakage')
    return training,evaluation,config,scenes,cases


class MixedValidation(SafetyFirstValidation):
    def __init__(self,*args,probability=.5,selection_seed=61,**kwargs):
        super().__init__(*args,**kwargs);self.probability=probability;self.selection_seed=selection_seed

    def score_model(self,candidate):
        # Model selection uses pure PPO results; mixture performance is separate.
        result,rows=super().score_model(candidate)
        added=candidate.num_timesteps-self.start
        mixed=evaluate_controller(candidate,self.config,self.s,self.cases,
            probability=self.probability,seed=self.selection_seed)
        dump(self.output/'mixed_evaluations'/f'episodes_{added}.json',mixed)
        if added==0:
            dump(self.output/'rrt_validation.json',evaluate_controller(candidate,self.config,self.s,self.cases,mode='rrt'))
        return result,rows


def evaluate(study,model_path=None,output=None):
    _,settings,config,_,cases=inputs(study)
    output=Path(output or study['evaluation_output'])
    if output.exists():raise FileExistsError('Refusing evaluation overwrite')
    model=PPO.load(model_path or study['initial_model'],device='cpu');output.mkdir(parents=True)
    summaries={}
    for mode in ('ppo','rrt','mixture'):
        result=evaluate_controller(model,config,settings,cases,mode,study['expert_probability'],study['seed'])
        dump(output/f'{mode}.json',result);summaries[mode]={'metrics':result['metrics'],'by_controller':result['by_controller']}
    dump(output/'summary.json',{'model_sha256':sha(model_path or study['initial_model']),
        'training_performed':False,'independent_test':False,'controllers':summaries})
    return summaries


def train(study):
    training,evaluation,config,scenes,cases=inputs(study)
    root=Path(study['output_root']);report=Path(study['report_root'])
    if root.exists() or report.exists():raise FileExistsError('Refusing experiment overwrite')
    root.mkdir(parents=True);report.mkdir(parents=True);ppo=root/'ppo';ppo.mkdir()
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'training_performed':True,
           'ppo_budget':study['ppo_steps'],'expert_probability':study['expert_probability']}
    def status(phase,**extra):
        state.update(phase=phase,**extra);dump(report/'status.json',state)
        print('EPISODE_MIX',json.dumps(state),flush=True)
    dump(report/'manifest.json',{'study':study,'configuration':config,'training_settings':training,
        'evaluation_settings':evaluation,'source_sha256':sha(study['initial_model']),
        'training_data_contract':'PPO buffer contains PPO actions only; successful RRT episodes supervise actor separately',
        'selection_rule':'minimum_ability_then_collision_rate_then_success_rate',
        'regression_is_independent_test':False,'heldout_excluded':sorted(set(read(study['heldout_scenes'])['heldout_scene_seeds'])),
        'train_scene_seeds':[s['seed'] for s in scenes],
        'code_hashes':{p:sha(p) for p in ('scripts/train_episode_rrt_ppo.py',
            'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/rl/episode_policy.py')}})
    def worker(rank):
        return ExpertEpisodeEnv(ConditionalTrainingEnv(copy.deepcopy(config),scenes,training,rank),
            probability=study['expert_probability'],seed=study['seed']+rank*1000,
            journal=root/f'episodes_worker_{rank}.jsonl')
    env=DummyVecEnv([lambda rank=i:worker(rank) for i in range(training['workers'])])
    try:
        model=PPO.load(study['initial_model'],device='cpu',env=env,seed=study['seed'],
            n_steps=training['n_steps'],batch_size=training['batch_size'],n_epochs=training['n_epochs'],
            learning_rate=study['ppo_learning_rate'],clip_range=training['clip_range'],target_kl=training['target_kl'])
        start=model.num_timesteps
        validation=MixedValidation(ppo,None,config,evaluation,cases,status,
            probability=study['expert_probability'],selection_seed=study['seed'])
        auxiliary=AuxiliaryImitation(root/'auxiliary_bc',study['auxiliary_bc'],study['seed'])
        # Validation runs first, then BC updates, then a new on-policy rollout.
        # Never override actions inside env.step or teach from validation cases.
        model.learn(total_timesteps=study['ppo_steps'],reset_num_timesteps=False,callback=[validation,auxiliary])
        if model.num_timesteps-start!=study['ppo_steps']:raise AssertionError('Unexpected PPO sampling budget')
        rows=[row for e in env.envs for row in e.records]
        counts={name:dict(Counter(r['reason'] for r in rows if r['controller']==name)) for name in ('rrt','ppo')}
        started={name:sum(e.started[name] for e in env.envs) for name in ('rrt','ppo')}
        rrt_steps=sum(r['steps'] for r in rows if r['controller']=='rrt')
        selection=read(ppo/'selection.json')
        shutil.copy2(ppo/'best_model.zip',report/'selected_model.zip')
        shutil.copy2(ppo/'final_model.zip',report/'final_candidate.zip')
        exported=read(report/'manifest.json');exported['model_sha256']=sha(report/'selected_model.zip')
        dump(report/'manifest.json',exported)
        dump(report/'controller_config.json',{'model_file':'selected_model.zip',
            'model_sha256':exported['model_sha256'],'configuration':config,'evaluation_settings':evaluation,
            'ppo_wrapper':'ConditionalRecoveryEnv','expert':'ClassicalController / RRT-Connect',
            'expert_probability':study['expert_probability'],'selection_unit':'episode'})
        dump(report/'summary.json',{'ppo_transitions':model.num_timesteps-start,'rrt_transitions':rrt_steps,
            'total_physics_transitions':model.num_timesteps-start+rrt_steps,'episode_starts':started,
            'completed_by_controller':counts,'imitation_accepted_updates':auxiliary.accepted,
            'imitation_rejected_updates':auxiliary.rejected,'imitation_pool_size':len(auxiliary.pool),
            'selection':selection,'model_sha256':sha(report/'selected_model.zip'),
            'episode_ratio_is_probability_not_exact_quota':True})
        evaluate(study,report/'selected_model.zip',report/'final_evaluation')
        status('complete',status='completed',selected_additional_steps=selection['additional_steps'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise
    finally:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/episode_rrt_ppo.yaml')
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--train',action='store_true');mode.add_argument('--evaluate',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());_,_,_,scenes,cases=inputs(study)
    print({'mode':'train' if args.train else 'evaluate' if args.evaluate else 'check_only',
        'train_layouts':len(scenes),'validation_cases':len(cases),'ppo_steps':study['ppo_steps'],
        'additional_rrt_steps':'counted separately','expert_probability':study['expert_probability']},flush=True)
    if args.train:train(study)
    if args.evaluate:print(evaluate(study),flush=True)


if __name__=='__main__':main()
