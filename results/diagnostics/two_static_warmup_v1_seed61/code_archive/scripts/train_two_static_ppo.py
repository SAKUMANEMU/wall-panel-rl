"""Conservative PPO after qualified two-static BC; retain one-static validation."""
import _bootstrap
import copy
import json
from pathlib import Path
import shutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_episode_rrt_ppo import inputs,MixedValidation
from train_conditional_ppo import score_model
from run_targeted_bc import read,dump,sha
from wall_panel_rl.rl.conditional_recovery import ConditionalTrainingEnv
from wall_panel_rl.rl.episode_imitation import ExpertEpisodeEnv,AuxiliaryImitation
from wall_panel_rl.rl.avoidance_curriculum import gate_result
from wall_panel_rl.rl.multi_obstacle_warmup import teacher_arrays


class RetainedOneValidation(MixedValidation):
    def __init__(self,*args,one_settings,one_cases,**kwargs):
        super().__init__(*args,**kwargs);self.one_settings=one_settings;self.one_cases=one_cases

    def score_model(self,candidate):
        result,rows=super().score_model(candidate)
        one,one_rows=score_model(candidate,None,self.config,self.one_settings,self.one_cases)
        retained=gate_result(one['metrics']['dynamic_ppo'],self.one_settings['gate'])['passed']
        retained=retained and one['metrics']['dynamic_ppo']['stage_target']['collision']==0 and one['checks']['goal_convergence']
        result['checks']['one_static_retained']=retained
        result['passed']=all(result['checks'].values())
        result['one_static_metrics']=one['metrics']['dynamic_ppo']
        dump(self.output/'one_regression'/f'episodes_{candidate.num_timesteps-self.start}.json',
             {'gate':one,'rows':one_rows,'retained':retained})
        return result,rows

    def candidate_score(self,result):
        base=super().candidate_score(result)
        return (int(base[0] and result['checks']['one_static_retained']),*base[1:])


class SeededImitation(AuxiliaryImitation):
    def __init__(self,*args,teacher_path,heldout,**kwargs):
        super().__init__(*args,**kwargs);self.teacher_path=teacher_path;self.heldout=heldout

    def _on_training_start(self):
        super()._on_training_start()
        data,parts=teacher_arrays(self.teacher_path,self.heldout)
        episodes=set(int(i) for group in parts['train'].values() for i in group)
        for i in sorted(episodes):
            for k in range(data['offsets'][i],data['offsets'][i+1]):self.pool.append((data['obs'][k],data['actions'][k]))
        dump(self.output/'initial_pool.json',{'transitions':len(self.pool),'heldout_excluded':sorted(self.heldout),
            'teacher_sha256':sha(self.teacher_path),'optimizer':'fresh auxiliary Adam'})


def train(study,one_settings,one_cases,teacher_path,heldout,status):
    training,evaluation,config,scenes,cases=inputs(study)
    root=Path(study['output_root']);report=Path(study['report_root']);ppo=root/'ppo'
    if root.exists() or report.exists():raise FileExistsError('Refusing PPO overwrite')
    ppo.mkdir(parents=True);report.mkdir(parents=True)
    dump(report/'manifest.json',{'study':study,'configuration':config,'training_settings':training,
        'evaluation_settings':evaluation,'one_evaluation_settings':one_settings,'source_sha256':sha(study['initial_model']),
        'ppo_optimizer':'fresh Adam after BC warm-up; original PPO actor moments are not reused',
        'auxiliary_optimizer':'fresh Adam; successful warm-up training labels seed replay pool',
        'train_scene_seeds':[s['seed'] for s in scenes],'heldout_excluded':sorted(heldout),
        'training_data_contract':'PPO samples only in rollout buffer; actor imitation separate',
        'code_hashes':{p:sha(p) for p in ('scripts/train_two_static_ppo.py','src/wall_panel_rl/rl/multi_obstacle_warmup.py',
            'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/rl/avoidance_curriculum.py')}})
    def worker(rank):
        return ExpertEpisodeEnv(ConditionalTrainingEnv(copy.deepcopy(config),scenes,training,rank),
            probability=study['expert_probability'],seed=study['seed']+rank*1000,journal=root/f'episodes_worker_{rank}.jsonl')
    env=DummyVecEnv([lambda rank=i:worker(rank) for i in range(training['workers'])])
    try:
        model=PPO.load(study['initial_model'],env=env,device='cpu',seed=study['seed'],n_steps=training['n_steps'],
            batch_size=training['batch_size'],n_epochs=training['n_epochs'],learning_rate=study['ppo_learning_rate'],
            clip_range=training['clip_range'],target_kl=training['target_kl'])
        model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),
            lr=study['ppo_learning_rate'],**model.policy.optimizer_kwargs)
        start=model.num_timesteps
        validation=RetainedOneValidation(ppo,None,config,evaluation,cases,status,
            one_settings=one_settings,one_cases=one_cases,probability=study['expert_probability'],selection_seed=study['seed'])
        aux=SeededImitation(root/'auxiliary_bc',study['auxiliary_bc'],study['seed'],teacher_path=teacher_path,heldout=heldout)
        model.learn(study['ppo_steps'],reset_num_timesteps=False,callback=[validation,aux])
        if model.num_timesteps-start!=study['ppo_steps']:raise AssertionError('PPO budget mismatch')
        selected=read(ppo/'selection.json')
        shutil.copy2(ppo/'best_model.zip',report/'selected_model.zip');shutil.copy2(ppo/'final_model.zip',report/'final_candidate.zip')
        meta=read(report/'manifest.json');meta['model_sha256']=sha(report/'selected_model.zip');dump(report/'manifest.json',meta)
        dump(report/'controller_config.json',{'model_file':'selected_model.zip','model_sha256':meta['model_sha256'],
            'configuration':config,'evaluation_settings':evaluation,'ppo_wrapper':'ConditionalRecoveryEnv',
            'expert_probability':study['expert_probability']})
        logs=[r for e in env.envs for r in e.records]
        dump(report/'summary.json',{'ppo_transitions':model.num_timesteps-start,
            'rrt_transitions':sum(r['steps'] for r in logs if r['controller']=='rrt'),
            'episode_starts':{k:sum(e.started[k] for e in env.envs) for k in ('rrt','ppo')},
            'auxiliary_updates':aux.accepted,'auxiliary_rejected':aux.rejected,'selection':selected,
            'model_sha256':meta['model_sha256']})
        status('ppo_complete',selected_ppo_steps=selected['additional_steps'])
        return report/'selected_model.zip',selected
    finally:env.close()
