"""Bounded stall-recovery BC and PPO comparison; default checks only."""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import functools
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
import staged_avoidance as stage
from run_targeted_bc import (read,dump,sha,collect_corrections,merge_teacher,make_test_bank,score_only)
from wall_panel_rl.rl.avoidance_imitation import pretrain_actor,closed_loop_score,teacher_reuse_contract
from wall_panel_rl.rl.stall_recovery import StallAwareEnv


def stall_handoffs(t,maximum=160):
    """Cover onset, stationary states and approach; leave >=80 expert execution steps."""
    if t['reason']!='timeout':return []
    states=t['states'];onset=None
    for i in range(20,len(states)):
        window=states[i-20:i+1]
        travel=np.linalg.norm(np.diff(np.asarray([r['q'] for r in window]),axis=0),axis=1).sum()
        if travel<.01:
            onset=i-20;break
    points=set(range(25,maximum+1,12))|{maximum}
    if onset is not None:points.update(max(0,min(maximum,onset+v)) for v in (-30,-10,0,10,30))
    # First collect states near identified stagnation, then earlier approaches.
    pivot=min(maximum,(onset+10) if onset is not None else maximum)
    return sorted((k for k in points if k<len(states)),key=lambda k:(abs(k-pivot),-k))


def worker(config,scenes,s,rank,output):
    torch.set_num_threads(1)
    return Monitor(StallAwareEnv(config,scenes,s,rank),str(output/f'worker_{rank}.monitor.csv'),
        info_keywords=('termination_reason','curriculum_task','reward_stall'))


class PilotValidation(BaseCallback):
    """Evaluate original reward/metrics; include the initial BC as a candidate."""
    def __init__(self,output,source,config,s,cases,status):
        super().__init__();self.output=output;self.source=source;self.config=config;self.s=s
        self.cases=cases;self.status=status;self.start=None;self.last=None;self.best=None

    def _on_step(self):return True
    def _on_training_start(self):
        self.start=self.model.num_timesteps;self.score('baseline')
    def _on_rollout_start(self):
        added=self.model.num_timesteps-self.start
        if added and added%self.s['eval_every']==0 and added!=self.last:self.score('post_update')
    def _on_rollout_end(self):
        dump(self.output/'sampling_progress.json',{'additional_steps':self.model.num_timesteps-self.start})
    def _on_training_end(self):
        self.model.save(self.output/'final_model.zip')
        self.score('final_saved_reload',reload_path=self.output/'final_model.zip')

    def candidate_score(self,result):
        return closed_loop_score(result,0.)

    def score_model(self,candidate):
        return stage.score_bc_model(candidate,self.source,self.config,self.s,self.cases)

    def score(self,kind,reload_path=None):
        states=(torch.get_rng_state(),np.random.get_state(),random.getstate())
        try:
            candidate=PPO.load(reload_path,device='cpu') if reload_path else self.model
            result,rows=self.score_model(candidate)
            added=candidate.num_timesteps-self.start
            score=self.candidate_score(result)
            event={'kind':kind,'additional_steps':added,'gate':result,
                   'metrics':result['metrics'],'evaluated_with_original_reward':True,
                   'selection_score':score}
            path=self.output/'checkpoints'/f'ppo_{added}.zip';path.parent.mkdir(exist_ok=True)
            candidate.save(path);dump(self.output/'evaluations'/f'episodes_{added}.json',rows)
            with (self.output/'validation.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            if self.best is None or score>self.best:
                self.best=score;candidate.save(self.output/'best_model.zip');dump(self.output/'selection.json',event)
            dump(self.output/'progress.json',event);self.last=added
            self.status('ppo',ppo_additional_steps=added,validation_target=result['metrics']['dynamic_ppo']['stage_target'])
        finally:
            torch.set_rng_state(states[0]);np.random.set_state(states[1]);random.setstate(states[2])


def run_ppo(study,config,s,scenes,cases,source,status,*,evaluation_settings=None,
            callback_class=PilotValidation,worker_factory=worker,initial_scorer=None):
    output=Path(study['output_root'])/'ppo'
    if output.exists():raise FileExistsError(output)
    base=PPO.load(study['initial_model'],device='cpu')
    eval_s=s if evaluation_settings is None else evaluation_settings
    scorer=stage.score_bc_model if initial_scorer is None else initial_scorer
    gate,_=scorer(base,source,config,eval_s,cases)
    dump(Path(study['report_root'])/'ppo_initial_gate.json',gate)
    if not gate['passed']:raise RuntimeError('Initial BC gate failed; no PPO workers or training')
    output.mkdir(parents=True)
    model=PPO.load(study['initial_model'],device='cpu',n_envs=s['workers'],n_steps=s['n_steps'],
        batch_size=s['batch_size'],learning_rate=study['ppo_learning_rate'],n_epochs=s['n_epochs'],
        clip_range=s['clip_range'],target_kl=s['target_kl'],seed=s['seed'],ent_coef=0.)
    # Preserve the starting actor, including its fixed exploration std.
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in base.policy.state_dict().items())
    start=model.num_timesteps;started=time.time()
    env=SubprocVecEnv([functools.partial(worker_factory,config,scenes,s,i,output) for i in range(s['workers'])],start_method='spawn')
    try:
        model.set_env(env)
        model.learn(total_timesteps=study['ppo_steps'],reset_num_timesteps=False,
            callback=callback_class(output,source,config,eval_s,cases,status))
        assert model.num_timesteps-start==study['ppo_steps']
        dump(output/'training.json',{'additional_steps':model.num_timesteps-start,'starting_steps':start,
            'elapsed_seconds':time.time()-started,'initial_model_sha256':sha(study['initial_model']),
            'stall_reward':study['stall_reward'],'automatic_stage_advance':False,
            'training_wall_reward':s.get('wall_reward'),
            'bc_optimizer_used_for_ppo':False,'ppo_optimizer':'loaded original PPO state; BC Adam is separate'})
    finally:env.close()
    return output/'best_model.zip',read(output/'selection.json')


def check(study):
    if study['bc_updates']!=10000 or study['ppo_steps']!=10000:raise ValueError('Budget must be exactly 10k per variant')
    if study['bc_checkpoints']!=[2000,4000,6000,8000,10000]:raise ValueError('Invalid BC checkpoints')
    if study['ppo_eval_every']!=5000:raise ValueError('Invalid PPO checks')
    for key in ('initial_model','initial_bc_state'):
        if not Path(study[key]).is_file():raise FileNotFoundError(study[key])
    if torch.load(study['initial_bc_state'],map_location='cpu',weights_only=False)['updates']!=20000:
        raise ValueError('Expected the 20k BC optimizer state')
    if not 0<study['bc_learning_rate']<=1e-4:raise ValueError('Expected low BC learning rate')


def run(study):
    check(study);root=Path(study['output_root']);data=Path(study['data_root']);report=Path(study['report_root'])
    if root.exists() or data.exists():raise FileExistsError('Refusing output overwrite/restart')
    diagnosis=read(report/'train_diagnosis.json');parent=Path(study['parent_report'])
    s=copy.deepcopy(read(parent/'targeted_mix.json')['settings']);config=read(parent/'manifest.json')['configuration']
    selected=PPO.load(parent/'selected_bc_model.zip',device='cpu');initial=PPO.load(study['initial_model'],device='cpu')
    if any(not torch.equal(v,initial.policy.state_dict()[k]) for k,v in selected.policy.state_dict().items()):
        raise ValueError('Starting checkpoint differs from selected current model')
    if diagnosis['model_sha256']!=sha(parent/'selected_bc_model.zip') or diagnosis['environment_config']!=config:
        raise ValueError('Cached training diagnosis has a different model/environment')
    if (report/'status.json').exists():raise FileExistsError('Study already started')
    root.mkdir(parents=True);data.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'runner_pid':os.getpid(),
           'bc_budget':study['bc_updates'],'ppo_budget':study['ppo_steps']}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('STALL',json.dumps(state),flush=True)
    manifest={'study':study,'configuration':config,'same_initial_actor':True,
        'bc_initial_hash':sha(study['initial_model']),'bc_state_hash':sha(study['initial_bc_state']),
        'diagnosis_sha256':sha(report/'train_diagnosis.json'),'test_used_for_selection':False,
        'code_hashes':{p:sha(p) for p in ['scripts/run_stall_recovery.py','scripts/run_targeted_bc.py',
            'src/wall_panel_rl/rl/stall_recovery.py','src/wall_panel_rl/rl/avoidance_imitation.py',
            'src/wall_panel_rl/rl/avoidance_curriculum.py','scripts/staged_avoidance.py']}}
    dump(report/'manifest.json',manifest);(report/'study.yaml').write_text(yaml.safe_dump(study))
    try:
        held=set(read(parent/'targeted_mix.json')['heldout_scene_seeds'])
        bank=read(Path(study['base_scenes'])/'train.json');train_bank=[c for c in bank if c['seed'] not in held]
        if {t['seed'] for t in diagnosis['traces']}!={c['seed'] for c in train_bank}:
            raise ValueError('Diagnosis bank or held-out membership changed')
        cases=read(Path(study['base_scenes'])/'validation_staged.json')
        forbidden={c['seed'] for c in bank}|{c['scene']['seed'] for c in cases}
        forbidden|={c['scene']['seed'] for c in read('data/targeted_bc_v3_seed61/independent_cases.json')}
        tests=make_test_bank(config,s,study,data/'independent_cases.json',forbidden,status)
        status('collect_stall_recoveries')
        collect_corrections(config,s,initial,diagnosis['traces'],data/'corrections',study,status,
            handoff_selector=lambda t:stall_handoffs(t,study['maximum_handoff_step']))
        status('audit_recoveries')
        subprocess.run([sys.executable,'scripts/audit_correction_data.py','--data',str(data/'corrections'),
                        '--output',str(report/'correction_replay.json')],check=True)
        audit=read(report/'correction_replay.json')
        if audit['success']!=audit['sampled_corrections'] or audit['max_observation_error']>1e-6:
            raise RuntimeError('Recovery data failed physical replay')
        merge_teacher(Path(study['base_teacher']),data/'corrections',data/'mixed_teacher',s,config)
        with np.load(data/'corrections/teacher.npz') as z:
            if held & set(z['seeds'].tolist()):raise ValueError('Held-out data used for recovery')
        teacher_reuse_contract(read(data/'mixed_teacher/metadata.json'),s,config)
        source=PPO.load(s['source_model'],device='cpu')
        bc_settings=copy.deepcopy(s);bc_settings['stage_output']=str(root/'bc_low_lr')
        bc_settings['teacher'].update(updates=study['bc_updates'],learning_rate=study['bc_learning_rate'],closed_loop_updates=study['bc_checkpoints'])
        output=root/'bc_low_lr';status('bc_low_lr',bc_update=0)
        def evaluator(candidate,update,losses):
            result,rows=stage.score_bc_model(candidate,source,config,bc_settings,cases)
            dump(output/'evaluations'/f'update_{update}.json',result)
            dump(output/'evaluations'/f'episodes_{update}.json',rows)
            status('bc_low_lr',bc_update=update,validation_target=result['metrics']['dynamic_ppo']['stage_target'])
            return result
        pretrain_actor(PPO.load(study['initial_model'],device='cpu'),data/'mixed_teacher',bc_settings,output,
            reuse_environment=config,evaluator=evaluator,resume_state=study['initial_bc_state'],heldout_seeds=sorted(held),include_initial=True)
        bc_meta=read(output/'metadata.json');dump(report/'bc_low_lr.json',bc_meta)
        ppo_settings=copy.deepcopy(s);ppo_settings['stall_reward']=study['stall_reward']
        ppo_settings['eval_every']=study['ppo_eval_every'];ppo_settings['learning_rate']=study['ppo_learning_rate']
        status('ppo',ppo_additional_steps=0)
        ppo_path,ppo_selection=run_ppo(study,config,ppo_settings,train_bank,cases,source,status)
        dump(report/'ppo_selection.json',ppo_selection)
        # Compare with the SAME original validation score; test cannot choose.
        candidates={'baseline':(Path(study['initial_model']),read(root/'bc_low_lr/evaluations/update_0.json')),
                    'bc_low_lr':(root/'bc_low_lr/model.zip',bc_meta['selected_closed_loop']),
                    'ppo_stall':(ppo_path,ppo_selection['gate'])}
        winner=max(candidates,key=lambda k:closed_loop_score(candidates[k][1],0.))
        status('independent_test',winner_locked=winner)
        independent={}
        for name,(path,metrics) in candidates.items():
            independent[name]=score_only(PPO.load(path,device='cpu'),config,s,tests)
            dump(report/f'independent_{name}.json',independent[name])
            status('independent_test',evaluated_model=name,independent_target=independent[name]['metrics']['stage_target'])
        shutil.copy2(candidates[winner][0],report/'selected_model.zip')
        summary={'winner_by_validation':winner,'validation':{k:v[1] for k,v in candidates.items()},
            'independent':{k:v['metrics'] for k,v in independent.items()},'bc_updates_completed':study['bc_updates'],
            'ppo_steps_completed':study['ppo_steps'],'selected_bc_additional_updates':bc_meta['selected_update'],
            'selected_ppo_additional_steps':ppo_selection['additional_steps'],
            'export_sha256':sha(report/'selected_model.zip'),'automatic_stage_advance':False}
        dump(report/'summary.json',summary);status('complete',status='completed',finished_at=datetime.now().astimezone().isoformat())
    except Exception as exc:
        status(state.get('phase','initialization'),status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/stall_recovery_study.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
