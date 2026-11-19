"""Four-static50/50 RRT/PPO entry; explicit train/evaluate flags only."""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from run_targeted_bc import read,dump,sha
from wall_panel_rl.rl.four_episode_mix import FourCaseBankEnv,FourEpisodeMixture,evaluate_mix
from wall_panel_rl.rl.four_ppo_continuation import load_continuation
from wall_panel_rl.rl.episode_imitation import AuxiliaryImitation
from wall_panel_rl.rl.four_priority_bc import score
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES


def inputs(spec):
    source=Path(spec['source_report']);p=read(source/'protocol.json');m=read(source/'candidate_manifest.json')
    if sha(spec['source_model'])!=m['model_sha256']:raise ValueError('Wrong source checkpoint')
    data=Path(spec['data_root']);splits=read(data/'split.json')
    train=read(data/'train_cases.json');validation=read(data/'validation_cases.json')
    held=set(splits['validation']+splits['test']+splits['old_validation']+splits['old_test_regression'])
    if set(c['scene']['seed'] for c in train)&held:raise ValueError('Held-out scene in train bank')
    if spec['expert_probability']!=.5:raise ValueError('This profile requires50% expert episode probability')
    rollout=spec['workers']*spec['n_steps']
    if spec['steps']%rollout or spec['evaluate_every']%rollout:raise ValueError('Whole rollout budgets required')
    if spec['reward_scale']!=p['spec']['reward_scale']:raise ValueError('Continuation reward units changed')
    probs=p.get('sampling_probabilities')
    if probs is not None:
        original_train=str(data/'train_cases.json')
        if sha(original_train)!=p['input_hashes'][original_train]:raise ValueError('Source train ordering changed')
    return p['configuration'],p['evaluation_settings'],train,validation,held,probs


def evaluate(spec,*,limit=0,mode='mixture',model_path=None,output=None):
    config,settings,_,cases,_,_=inputs(spec)
    if limit:cases=cases[:limit]
    root=Path(output or spec['evaluation_root'])
    if root.exists():raise FileExistsError('Refusing evaluation overwrite')
    root.mkdir(parents=True);torch.set_num_threads(1);model_file=Path(model_path or spec['source_model'])
    model=PPO.load(model_file,device='cpu')
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):raise ValueError('Expected77 observations and7 actions')
    print('MIX_EVALUATION',json.dumps({'cases':len(cases),'mode':mode,'formal_training':False}),flush=True)
    modes=['ppo','rrt','mixture'] if mode=='all' else [mode]
    summaries={}
    for name in modes:
        result=evaluate_mix(model,config,settings,cases,mode=name,probability=.5,seed=spec['seed'])
        dump(root/f'{name}.json',result);summaries[name]={'metrics':result['metrics'],'by_controller':result['by_controller']}
        print('MIX_RESULT',name,json.dumps(summaries[name]),flush=True)
    # Real training-wrapper probe: deliberate functional seed gives expert then
    # PPO ownership. No optimizer step or checkpoint write occurs in this probe.
    train=read(Path(spec['data_root'])/'train_cases.json')[:2]
    env=FourEpisodeMixture(FourCaseBankEnv(config,settings,train,seed=8),probability=.5,seed=8,reward_scale=spec['reward_scale'])
    try:
        obs,_=env.reset()
        while True:
            obs,_,done,truncated,_=env.step(model.predict(obs,deterministic=True)[0])
            if done or truncated:break
        probe={'records':env.records,'verified_bc_pairs':len(env.pending),'ppo_updates':0,
               'note':'functional training-wrapper execution only; source weights unchanged'}
        assert any(r['controller']=='rrt' and r['accepted_for_imitation'] for r in env.records)
        assert any(r['controller']=='ppo' for r in env.records)
        dump(root/'training_wrapper_probe.json',probe)
    finally:env.close()
    dump(root/'summary.json',{'source_model_sha256':sha(model_file),'model_file':str(model_file.resolve()),'cases':len(cases),
        'expert_probability':.5,'episode_branch_unit':True,'controllers':summaries,
        'training_performed':False,'independent_test':False,'limited_functional_preflight':bool(limit),
        'verified_training_wrapper_pairs':probe['verified_bc_pairs']})
    return summaries


def train(spec):
    config,settings,cases,validation,held,probs=inputs(spec)
    root,report=Path(spec['model_root']),Path(spec['report_root'])
    if root.exists() or report.exists():raise FileExistsError('Refusing mixture training overwrite')
    root.mkdir(parents=True);report.mkdir(parents=True);torch.set_num_threads(1)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'expert_probability':.5}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('FOUR_EPISODE_MIX',json.dumps(state),flush=True)
    status('initializing',ppo_steps=0,ppo_budget=spec['steps'],train_layouts=len(cases))
    workers=[]
    def worker(rank):
        physical=FourCaseBankEnv(config,settings,cases,excluded=held,seed=spec['seed']+rank*1000,probabilities=probs)
        wrapper=FourEpisodeMixture(physical,probability=.5,seed=spec['seed']+rank*1000,
            reward_scale=spec['reward_scale'],journal=root/f'episodes_worker_{rank}.jsonl')
        workers.append(wrapper);return wrapper
    env=DummyVecEnv([lambda i=i:worker(i) for i in range(spec['workers'])])
    try:
        model,continued=load_continuation(spec['source_model'],env,spec)
        original=copy.deepcopy(model.policy.state_dict());start=model.num_timesteps
        model.set_logger(configure(str(root/'training'),['csv']))
        code=['scripts/train_four_episode_mix.py','configs/four_episode_mix.yaml','src/wall_panel_rl/rl/four_episode_mix.py',
              'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/rl/four_ppo_continuation.py',
              'src/wall_panel_rl/rl/episode_policy.py','src/wall_panel_rl/baselines/classical.py',
              'src/wall_panel_rl/planning/rrt_connect.py','src/wall_panel_rl/planning/static_validation.py']
        dump(report/'protocol.json',{'spec':spec,'configuration':config,'evaluation_settings':settings,
            'source_sha256':sha(spec['source_model']),'continuation':continued,'train_seeds':[c['scene']['seed'] for c in cases],
            'excluded_seeds':sorted(held),'validation_case_sha256':sha(Path(spec['data_root'])/'validation_cases.json'),
            'code_hashes':{p:sha(p) for p in code},'expert_is_bc_network':False,
            'data_rule':'PPO buffer uses PPO actions/logp only; successful verified RRT labels use separate BC Adam at next rollout start',
            'selection':'full pure-PPO validation; mixture score reported separately','test_used':False})
        history=[];baseline=None;best=None;selected=None
        def checkpoint(added):
            nonlocal baseline,best,selected
            rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
            try:
                status('evaluating_pure_ppo',ppo_steps=added)
                pure=evaluate_mix(model,config,settings,validation,mode='ppo',seed=spec['seed'])
                status('evaluating_mixture',ppo_steps=added,pure_ppo=pure['metrics']['stage_target'])
                mixed=evaluate_mix(model,config,settings,validation,mode='mixture',seed=spec['seed'])
            finally:torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
            metrics=pure['metrics']['stage_target']
            if baseline is None:baseline=metrics
            event={'ppo_steps':added,'ppo':metrics,'mixture':mixed['metrics']['stage_target'],
                   'mixture_by_controller':mixed['by_controller'],'score':score(metrics,baseline['collision'])}
            model.save(root/'checkpoints'/f'ppo_{added}.zip')
            if best is None or tuple(event['score'])>best:
                best=tuple(event['score']);selected=copy.deepcopy(event);model.save(root/'selected_model.zip')
            dump(report/'validation'/f'ppo_{added}.json',pure);dump(report/'validation'/f'mixture_{added}.json',mixed)
            history.append(event);dump(report/'history.json',history);status('validation',**event)
        class Validation(BaseCallback):
            def _on_training_start(self):checkpoint(0)
            def _on_rollout_start(self):
                added=self.model.num_timesteps-start
                if added and added%spec['evaluate_every']==0:checkpoint(added)
                status('ppo_rollout',ppo_steps=added,completed={name:sum(r['controller']==name for w in workers for r in w.records) for name in ('rrt','ppo')})
            def _on_step(self):return True
            def _on_training_end(self):checkpoint(self.model.num_timesteps-start)
        auxiliary=AuxiliaryImitation(root/'auxiliary_bc',spec['auxiliary_bc'],spec['seed'])
        # Validation snapshots first; BC update next; only then sample a new PPO rollout.
        status('initial_episode_resets',ppo_steps=0,ppo_budget=spec['steps'])
        model.learn(spec['steps'],reset_num_timesteps=False,callback=[Validation(),auxiliary])
        assert model.num_timesteps-start==spec['steps']
        model.save(root/'final_model.zip');chosen=PPO.load(root/'selected_model.zip',device='cpu')
        assert all(torch.equal(v,chosen.policy.state_dict()[k]) for k,v in original.items() if not k.startswith(('multi_adapter.',*CRITIC_PREFIXES)))
        records=[r for w in workers for r in w.records];rrt_steps=sum(r['steps'] for r in records if r['controller']=='rrt')
        dump(report/'summary.json',{'ppo_transitions':spec['steps'],'rrt_transitions':rrt_steps,'total_physics_transitions':spec['steps']+rrt_steps,
            'episode_starts':{name:sum(w.started[name] for w in workers) for name in ('rrt','ppo')},
            'completed_by_controller':{name:dict(Counter(r['reason'] for r in records if r['controller']==name)) for name in ('rrt','ppo')},
            'selected':selected,'bc_auxiliary_updates':auxiliary.accepted,'bc_rejected_updates':auxiliary.rejected,
            'on_policy_data_separated':True,'probability_not_quota':True,'source_unchanged':sha(spec['source_model'])==read(report/'protocol.json')['source_sha256'],
            'independent_test':False,'default_model_preserved':True})
        shutil.copy2(root/'selected_model.zip',report/'selected_model.zip');shutil.copy2(root/'final_model.zip',report/'final_candidate.zip')
        dump(report/'candidate_manifest.json',{'configuration':config,'evaluation_settings':settings,'model_sha256':sha(report/'selected_model.zip'),
            'expert_probability':.5,'expert_controller':'RRT-Connect','ownership_unit':'episode'})
        for path in code:
            dest=report/'code_archive'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        status('complete',status='completed',selected=selected,ppo_steps=spec['steps'],rrt_steps=rrt_steps)
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_episode_mix.yaml')
    group=p.add_mutually_exclusive_group();group.add_argument('--train',action='store_true');group.add_argument('--evaluate',action='store_true')
    p.add_argument('--limit',type=int,default=0,help='Evaluation case limit only;0 means full validation')
    p.add_argument('--mode',choices=['mixture','ppo','rrt','all'],default='mixture',help='Evaluation mode; training always uses the50/50 episode mixture')
    p.add_argument('--model',type=Path,help='Evaluation checkpoint override only')
    p.add_argument('--output',type=Path,help='New evaluation output directory; existing results are never overwritten')
    args=p.parse_args();spec=yaml.safe_load(Path(args.config).read_text());_,_,cases,validation,_,_=inputs(spec)
    if args.limit<0:p.error('Nonnegative evaluation limit required')
    if args.train and (args.limit or args.mode!='mixture' or args.model or args.output):p.error('--limit/--mode/--model/--output only apply to evaluation')
    print({'mode':'train' if args.train else 'evaluate' if args.evaluate else 'check_only','train_layouts':len(cases),
           'validation_layouts':len(validation),'expert_probability':.5,'ppo_steps':spec['steps'],'extra_rrt_steps':'counted separately'},flush=True)
    if args.train:train(spec)
    if args.evaluate:print(evaluate(spec,limit=args.limit,mode=args.mode,model_path=args.model,output=args.output),flush=True)


if __name__=='__main__':main()
