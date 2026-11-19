"""Conservative v5 continuation: 50/50 RRT/PPO, 30% static retention, gated speeds."""
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
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.four_ppo_continuation import load_continuation
from wall_panel_rl.rl.episode_imitation import AuxiliaryImitation
from wall_panel_rl.rl.imitation_continuation import ContinuedImitation
from wall_panel_rl.rl.dynamic_four_training import (DynamicFourBankEnv,DynamicFourEpisodeMixture,
    evaluate_ppo,expert_cache,compose_mixture,curriculum_gate)
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES
from run_targeted_bc import read,dump,sha


def speed_key(speed):return 'dynamic_'+format(speed,'.3f')


def inputs(spec):
    preparation=read(ROOT/spec['preparation_report']/'protocol.json');data=ROOT/spec['data_root']
    for f,expected in preparation['input_hashes'].items():
        if sha(ROOT/f)!=expected:raise ValueError('Prepared bank changed: '+f)
    expected=(read(ROOT/spec['source_report']/'candidate_manifest.json')['model_sha256']
              if spec.get('resume_auxiliary_state') else preparation['source_sha256'])
    if sha(ROOT/spec['source_model'])!=expected:raise ValueError('Source model changed')
    split=read(data/'split.json');train=read(data/'train_cases.json');val=read(data/'validation_cases.json')
    held=set(split['validation']+split['test']+split['previously_excluded'])
    if set(split['train'])&held:raise ValueError('Held-out cases in train bank')
    if spec['expert_probability']!=.5 or not np.isfinite(spec['static_probability']) or not 0<=spec['static_probability']<=1:
        raise ValueError('Expected 50/50 owners and a valid static scene probability')
    if spec['steps']%(spec['workers']*spec['n_steps']) or spec['evaluate_every']%(spec['workers']*spec['n_steps']):raise ValueError('Whole rollouts required')
    speeds=np.asarray(spec['curriculum_speeds'])
    if not len(speeds):raise ValueError('At least one training speed is required')
    target_speed=spec.get('validation_target_speed',float(speeds[-1]))
    if not len(speeds) or not np.isfinite(speeds).all() or np.any(speeds<=0) or np.any(np.diff(speeds)<=0) or target_speed!=.02:
        raise ValueError('Invalid curriculum speeds or target evaluation speed')
    old=read(ROOT/spec['source_report']/'protocol.json')
    if spec['reward_scale']!=old['spec']['reward_scale']:raise ValueError('Reward units changed')
    legacy=read(ROOT/spec['source_data']/'validation_cases.json')
    return preparation,train,val,legacy,held,read(data/'sampling.json')['probabilities']


def select_score(target,baseline,static,static0,legacy,legacy0):
    eligible=(target['joint_limit']==0 and target['collision']<=baseline['collision'] and
        static['success']>=static0['success'] and static['collision']<=static0['collision'] and static['joint_limit']==0 and
        legacy['success']>=legacy0['success'] and legacy['collision']<=legacy0['collision'] and legacy['joint_limit']==0)
    return [int(eligible),target['success'],-target['collision'],-target['timeout']]


def train(spec,config_file):
    p,cases,validation,legacy,held,probs=inputs(spec);config=p['configuration'];settings=p['evaluation_settings']
    root=ROOT/spec['model_root'];report=ROOT/spec['report_root']
    if root.exists() or report.exists():raise FileExistsError('Refusing dynamic training overwrite')
    root.mkdir(parents=True);report.mkdir(parents=True);torch.set_num_threads(1)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'ppo_budget':spec['steps'],'expert_probability':.5,'static_probability':spec['static_probability']}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('DYNAMIC_PPO',json.dumps(state),flush=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():
        if visible.exists() or visible.is_symlink():raise FileExistsError('Visible report alias exists')
        visible.symlink_to(report,target_is_directory=True)
    status('initializing',ppo_steps=0,train_layouts=len(cases),validation_layouts=len(validation),reserved_test_layouts=spec['test_count'])
    workers=[]
    def worker(rank):
        env=DynamicFourBankEnv(config,settings,cases,spec,seed=spec['seed']+rank*1000,excluded=held,probabilities=probs)
        wrapped=DynamicFourEpisodeMixture(env,probability=.5,seed=spec['seed']+rank*1000,reward_scale=spec['reward_scale'],journal=root/f'episodes_worker_{rank}.jsonl')
        workers.append(wrapped);return wrapped
    env=DummyVecEnv([lambda i=i:worker(i) for i in range(spec['workers'])])
    try:
        model,continued=load_continuation(ROOT/spec['source_model'],env,spec)
        original=copy.deepcopy(model.policy.state_dict());start=model.num_timesteps
        model.set_logger(configure(str(root/'training'),['csv']))
        files=[str(Path(config_file).relative_to(ROOT)),spec['motion_profile'],'scripts/train_four_dynamic_ppo.py',
            'src/wall_panel_rl/rl/dynamic_four_training.py','src/wall_panel_rl/rl/slow_dynamic_four.py',
            'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/rl/conditional_recovery.py',
            'src/wall_panel_rl/rl/avoidance_curriculum.py','src/wall_panel_rl/rl/autonomous_dynamic_ppo.py',
            'src/wall_panel_rl/rl/four_ppo_continuation.py','src/wall_panel_rl/rl/imitation_continuation.py','src/wall_panel_rl/rl/multi_obstacle_adapter.py',
            'src/wall_panel_rl/baselines/classical.py','src/wall_panel_rl/planning/timed_rrt.py',
            'src/wall_panel_rl/planning/rrt_connect.py','src/wall_panel_rl/planning/static_validation.py',
            'src/wall_panel_rl/envs/wall_panel_env.py']
        hashes={f:sha(ROOT/f) for f in files}
        runtime_inputs=dict(p['input_hashes'])
        runtime_inputs[spec['source_data']+'/validation_cases.json']=sha(ROOT/spec['source_data']/'validation_cases.json')
        for key in ('resume_auxiliary_state','validation_expert_cache'):
            if spec.get(key):runtime_inputs[spec[key]]=sha(ROOT/spec[key])
        protocol={'spec':spec,'configuration':config,'evaluation_settings':settings,'source_sha256':sha(ROOT/spec['source_model']),
            'continuation':continued,'input_hashes':runtime_inputs,'code_hashes':hashes,
            'train_seeds':[c['scene']['seed'] for c in cases],'excluded_seeds':sorted(held),
            'validation_expert_rule':'Physically execute the common RRT assignments once and cache rows; paired mixture uses matching pure-PPO rows. No evaluation labels enter BC.',
            'selection':'Fixed 0.02m/s pure-PPO validation, with frozen-layout and original-static non-regression guards; earlier checkpoint wins ties',
            'curriculum':('Fixed training speed; no automatic progression' if len(spec['curriculum_speeds'])==1 else
                'Advance one level only after a complete rollout and non-regression at current speed, target speed, and both static suites'),
            'test_rule':('Validation-only speed screening; previously used test is not reevaluated' if not spec.get('run_independent_test',True) else
                'One source-vs-selected pure-PPO 100-case paired test only after a new checkpoint is selected; never tune afterwards'),
            'episode_resume_rule':'Start fresh physical episodes with paired seed streams; preserve checkpoint policy, PPO Adam/value state, and optional BC Adam/pool/RNG'}
        dump(report/'protocol.json',protocol)
        for f in files:
            dest=report/'code_archive'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
        history=[];baseline={};cache=None;level=0;best=None;selected=None;target_speed=spec.get('validation_target_speed',spec['curriculum_speeds'][-1])
        def checkpoint(added):
            nonlocal cache,level,best,selected
            rng=(torch.get_rng_state(),np.random.get_state(),random.getstate());results={}
            try:
                speeds=sorted(set(spec['curriculum_speeds']+[target_speed])) if added==0 else sorted(set([spec['curriculum_speeds'][level],target_speed]))
                for speed in speeds:
                    key=speed_key(speed);status('validating_'+key,ppo_steps=added,curriculum_level=level)
                    results[key]=evaluate_ppo(model,config,settings,validation,speed)
                status('validating_frozen_layouts',ppo_steps=added)
                results['static_frozen']=evaluate_ppo(model,config,settings,validation,target_speed,static=True)
                status('validating_original_static',ppo_steps=added)
                results['legacy_static']=evaluate_ppo(model,config,settings,legacy,target_speed,static=True)
                if added==0:
                    baseline.update({key:value['metrics'] for key,value in results.items()})
                    status('evaluating_common_rrt_branch',ppo_steps=0)
                    if spec.get('validation_expert_cache'):
                        from wall_panel_rl.rl.episode_policy import EpisodeChoice
                        cache=read(ROOT/spec['validation_expert_cache']);choice=EpisodeChoice(.5,spec.get('evaluation_seed',spec['seed']))
                        expected=[]
                        for case in validation:
                            expected.append({'seed':case['scene']['seed'],'controller':choice.begin_episode()});choice.end_episode()
                        if expected!=cache['owners']:raise ValueError('Expert cache case order or coin assignments changed')
                    else:cache=expert_cache(config,settings,validation,target_speed,spec.get('evaluation_seed',spec['seed']))
                    dump(report/'validation_expert_cache.json',cache)
            finally:torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
            target=results[speed_key(target_speed)]['metrics'];static=results['static_frozen']['metrics'];old_static=results['legacy_static']['metrics']
            score=select_score(target,baseline[speed_key(target_speed)],static,baseline['static_frozen'],old_static,baseline['legacy_static'])
            mixture=compose_mixture(results[speed_key(target_speed)],cache)
            event={'ppo_steps':added,'curriculum_level':level,'curriculum_speed':spec['curriculum_speeds'][level],
                'target':target,'static_frozen':static,'legacy_static':old_static,'mixture':mixture['metrics'],
                'mixture_by_controller':mixture['by_controller'],'score':score,'all_metrics':{k:v['metrics'] for k,v in results.items()}}
            model.save(root/'checkpoints'/f'ppo_{added}.zip')
            if best is None or tuple(score)>best:
                best=tuple(score);selected=copy.deepcopy(event);model.save(root/'selected_model.zip')
            for key,value in results.items():dump(report/'validation'/f'{key}_{added}.json',value)
            dump(report/'validation'/f'mixture_{added}.json',mixture)
            if added and added<spec['steps'] and level<len(spec['curriculum_speeds'])-1:
                gate=curriculum_gate(results[speed_key(spec['curriculum_speeds'][level])]['metrics'],baseline[speed_key(spec['curriculum_speeds'][level])],
                    static,baseline['static_frozen'],target,baseline[speed_key(target_speed)])
                gate['checks']['legacy_static']=bool(old_static['joint_limit']==0 and old_static['success']>=baseline['legacy_static']['success'] and old_static['collision']<=baseline['legacy_static']['collision'])
                gate['passed']=all(gate['checks'].values());event['advance_gate']=gate
                if gate['passed']:
                    level+=1
                    for w in workers:w.env.set_curriculum(level)
            event['next_curriculum_level']=level;history.append(event);dump(report/'history.json',history)
            status('validation',**event,selected_ppo_steps=selected['ppo_steps'])
        class Validation(BaseCallback):
            def _on_training_start(self):checkpoint(0)
            def _on_rollout_start(self):
                added=self.model.num_timesteps-start
                if added and added%spec['evaluate_every']==0:checkpoint(added)
                status('ppo_rollout',ppo_steps=added,curriculum_level=level,curriculum_speed=spec['curriculum_speeds'][level],
                    completed={owner:sum(r['controller']==owner for w in workers for r in w.records) for owner in ('rrt','ppo')})
            def _on_step(self):return True
            def _on_training_end(self):checkpoint(self.model.num_timesteps-start)
        auxiliary=(ContinuedImitation(root/'auxiliary_bc',spec['auxiliary_bc'],spec['seed'],resume_path=ROOT/spec['resume_auxiliary_state'])
            if spec.get('resume_auxiliary_state') else AuxiliaryImitation(root/'auxiliary_bc',spec['auxiliary_bc'],spec['seed']))
        status('initial_training_resets',ppo_steps=0)
        model.learn(spec['steps'],reset_num_timesteps=False,callback=[Validation(),auxiliary])
        dump(report/'auxiliary_restoration.json',getattr(auxiliary,'restoration',{'restored':False}))
        assert model.num_timesteps-start==spec['steps']
        model.save(root/'final_model.zip');chosen=PPO.load(root/'selected_model.zip',device='cpu')
        assert all(torch.equal(v,chosen.policy.state_dict()[k]) for k,v in original.items() if not k.startswith(('multi_adapter.',*CRITIC_PREFIXES)))
        independent={'performed':False,'reason':'No validation-improving checkpoint selected; reserved test remains unused'}
        if not spec.get('run_independent_test',True):
            independent={'performed':False,'reason':'Validation-only speed screening; previously used test bank remains excluded'}
        elif selected['ppo_steps']>0:
            status('independent_paired_test',ppo_steps=spec['steps'])
            test=read(ROOT/spec['data_root']/'test_cases.json');source=PPO.load(ROOT/spec['source_model'],device='cpu')
            a=evaluate_ppo(source,config,settings,test,target_speed);b=evaluate_ppo(chosen,config,settings,test,target_speed)
            a['independent_test']=True;b['independent_test']=True
            independent={'performed':True,'source':a['metrics'],'selected':b['metrics'],'selected_ppo_steps':selected['ppo_steps'],
                'paired':dict(Counter(x['reason']+' -> '+y['reason'] for x,y in zip(a['rows'],b['rows']))),'no_tuning_after_test':True}
            dump(report/'independent_source.json',a);dump(report/'independent_selected.json',b)
        dump(report/'independent_test.json',independent)
        records=[r for w in workers for r in w.records];rrt_steps=sum(r['steps'] for r in records if r['controller']=='rrt')
        replay_steps=sum(r.get('replay_steps',0) for r in records)
        same=all(torch.equal(v,chosen.policy.state_dict()[k]) for k,v in original.items())
        summary={'ppo_transitions':spec['steps'],'rrt_transitions':rrt_steps,'verification_replay_transitions':replay_steps,
            'rollout_physics_transitions':spec['steps']+rrt_steps,'total_training_and_replay_physics':spec['steps']+rrt_steps+replay_steps,
            'episode_starts':{owner:sum(w.started[owner] for w in workers) for owner in ('rrt','ppo')},
            'completed_by_controller':{owner:dict(Counter(r['reason'] for r in records if r['controller']==owner)) for owner in ('rrt','ppo')},
            'completed_by_task':dict(Counter(r['task'] for r in records)),
            'verified_expert_episodes':sum(r.get('accepted_for_imitation',False) for r in records),
            'bc_auxiliary_updates':auxiliary.accepted-getattr(auxiliary,'initial_accepted',0),
            'bc_rejected_updates':auxiliary.rejected-getattr(auxiliary,'initial_rejected',0),
            'bc_total_accepted_updates':auxiliary.accepted,'bc_total_rejected_updates':auxiliary.rejected,
            'bc_restoration':getattr(auxiliary,'restoration',{'restored':False}),'selected':selected,
            'selected_equals_source':same,'source_unchanged':sha(ROOT/spec['source_model'])==protocol['source_sha256'],
            'independent_test':independent,'default_model_preserved':True,'final_curriculum_level':level}
        for f,expected in hashes.items():assert sha(ROOT/f)==expected,'Runtime code changed during training'
        for f,expected in runtime_inputs.items():assert sha(ROOT/f)==expected,'Data changed during training'
        dump(report/'summary.json',summary)
        shutil.copy2(root/'selected_model.zip',report/'selected_model.zip');shutil.copy2(root/'final_model.zip',report/'final_candidate.zip')
        dump(report/'candidate_manifest.json',{'configuration':config,'evaluation_settings':settings,'model_sha256':sha(report/'selected_model.zip'),
            'expert_probability':.5,'expert_controller':'TimedRRT','ownership_unit':'episode','motion_profile':spec['motion_profile']})
        status('complete',status='completed',ppo_steps=spec['steps'],selected=selected,summary=summary,
            completed={owner:sum(summary['completed_by_controller'][owner].values()) for owner in ('rrt','ppo')})
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_dynamic_ppo.yaml');p.add_argument('--train',action='store_true')
    args=p.parse_args();spec=yaml.safe_load(args.config.read_text());_,cases,validation,_,_,_=inputs(spec)
    print({'mode':'train' if args.train else 'check_only','ppo_steps':spec['steps'],'train':len(cases),'validation':len(validation),'expert_probability':.5,'static_probability':spec['static_probability'],'speeds':spec['curriculum_speeds']},flush=True)
    if args.train:train(spec,args.config.resolve())


if __name__=='__main__':main()
