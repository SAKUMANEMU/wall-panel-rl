"""Frozen-scene 70-input PPO pilot, exact-budget curriculum, and paired evaluation.

Run subcommands in order: prepare, pilot, train --condition clean/noise, evaluate.
All runs are isolated; reward/tolerances and final scene settings remain frozen.
"""
import _bootstrap
import argparse
import copy
import functools
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import capture_scenario, rollout

ROOT=Path('models/dynamic70_study')
BANK=ROOT/'scenes'


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def prepare():
    ROOT.mkdir(parents=True,exist_ok=False)
    BANK.mkdir()
    config=load_config('configs/ppo_dynamic_clean.yaml')
    (ROOT/'frozen_config.yaml').write_text(yaml.safe_dump(config))
    dump(ROOT/'protocol.json',{'training_seed':42,'pilot_steps':50000,'steps_per_condition':1000000,
        'phases':{'alignment':200000,'static':200000,'dynamic':600000},
        'train_scenes':128,'validation_scenes':10,'test_scenes':100,
        'noise':{'position_std':.005,'orientation_std_deg':.5},
        'selection':'Validation success rate, then mean reward; only final dynamic-phase checkpoints',
        'observation':'70; each obstacle: relative position(3), size(3), world velocity(3), presence(1)',
        'rollout_steps_per_worker':250,'workers':4,'batch_size':100,
        'device':'cpu','torch_threads':1})
    e=WallPanelEnv(config)
    try:
        for split,count,start in [('train',128,30000),('validation',10,40000),('test',100,50000)]:
            scenes=[];failed=[];seed=start
            while len(scenes)<count and seed<start+1000:
                try:
                    e.reset(seed=seed)
                    scenes.append(capture_scenario(e,seed))
                except RuntimeError as exc:
                    failed.append({'seed':seed,'error':str(exc)})
                seed+=1
                dump(BANK/f'{split}.json',scenes)
                dump(BANK/f'{split}_generation_failures.json',failed)
                if len(scenes)%10==0:print(split,len(scenes),'rejected',len(failed),flush=True)
            if len(scenes)!=count:raise RuntimeError(f'Could not fill {split}')
    finally:e.close()
    dump(ROOT/'scene_hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in BANK.glob('*.json')})


class CurriculumEnv(gym.Wrapper):
    def __init__(self,config,scenes,phase,seed,phase_steps):
        c=copy.deepcopy(config)
        if phase=='alignment':
            c['obstacles']['stage']='A'
            c['obstacles']['dynamic']={'enabled':False}
            c['robot']['start_position_distance_range']=None
        elif phase=='static':c['obstacles']['freeze_motion']=True
        super().__init__(WallPanelEnv(c))
        self.scenes=scenes;self.phase=phase;self.rng=np.random.default_rng(seed)
        self.used=0;self.budget=phase_steps;self.offset=np.array(config['robot']['start_offset'])

    def reset(self,*,seed=None,options=None):
        if self.phase=='alignment':
            # Begin with terminal alignment, then progressively restore far starts.
            progress=min(1.,self.used/max(1,self.budget))
            scale=.05+.95*progress
            scene=self.scenes[int(self.rng.integers(len(self.scenes)))]
            path=scene['reference_joint_path']
            # Follow a collision-checked reference rather than linearly blending
            # joint angles through potentially self-colliding configurations.
            start=np.array(path[int((1-scale)*(len(path)-1))])
            self.env.config['robot']['start_offset']=(start-self.env.goal_q).tolist()
            self.env.config['robot']['start_randomization']=.03*scale
            scene_seed=int(self.rng.integers(200000,2000000000))
            return self.env.reset(seed=scene_seed)
        scene=self.scenes[int(self.rng.integers(len(self.scenes)))]
        return self.env.reset(seed=scene['seed'],options={'scenario':scene,'noise_seed':int(self.rng.integers(200000,2000000000))})

    def step(self,action):
        self.used+=1
        return self.env.step(action)


def make_worker(config,scenes,phase,seed,budget,log):
    torch.set_num_threads(1)
    return Monitor(CurriculumEnv(config,scenes,phase,seed,budget),log)


class Progress(BaseCallback):
    def __init__(self,directory,validation,config,phase,target,best):
        super().__init__();self.directory=directory;self.validation=validation;self.config=config
        self.phase=phase;self.target=target;self.best=best;self.started=time.time();self.phase_start=None
        self.last_saved_step=None

    def _on_training_start(self):self.phase_start=self.model.num_timesteps

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        # SB3 updates AFTER rollout_end. The next rollout_start sees updated weights.
        self._save_updated_weights()

    def _on_training_end(self):
        # No next rollout exists for the final update: validate it explicitly.
        self._save_updated_weights(force=True)

    def _save_updated_weights(self,force=False):
        step=self.model.num_timesteps
        if step==self.phase_start or step==self.last_saved_step:
            return
        if force or step%50000==0:
            self.last_saved_step=step
            self.model.save(self.directory/'checkpoints'/f'checkpoint_{self.model.num_timesteps}')
            if self.phase=='dynamic':
                env=WallPanelEnv(self.config);rows=[]
                try:
                    for scene in self.validation:
                        r,_=rollout(env,'ppo',scene['seed'],self.model,scenario=scene);rows.append(r)
                finally:env.close()
                score=(sum(r['success'] for r in rows)/len(rows),float(np.mean([r['cumulative_reward'] for r in rows])))
                record={'steps':self.model.num_timesteps,'success_rate':score[0],'mean_reward':score[1],
                        'validation_timing':'post_update'}
                with (self.directory/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if score>tuple(self.best['score']):
                    self.best.update(score=score,steps=self.model.num_timesteps)
                    self.model.save(self.directory/'best_model');dump(self.directory/'selection.json',self.best)
                print('validation',record,flush=True)
        return True

    def _on_rollout_end(self):
        step=self.model.num_timesteps
        elapsed=time.time()-self.started
        dump(self.directory/'progress.json',{'phase':self.phase,'steps':step,'target_steps':self.target,
             'phase_elapsed_seconds':elapsed,'phase_fps':(step-self.phase_start)/max(elapsed,.001),'updated_unix':time.time()})
        if step%10000==0:print(self.directory.name,self.phase,step,'fps',round((step-self.phase_start)/max(elapsed,.001),1),flush=True)


def train(condition,pilot=False):
    torch.set_num_threads(1)
    config=yaml.safe_load((ROOT/'frozen_config.yaml').read_text())
    config['ppo']['n_steps']=250
    if condition=='noise':config['noise']={'position_std':.005,'orientation_std_deg':.5}
    scenes=json.loads((BANK/'train.json').read_text());validation=json.loads((BANK/'validation.json').read_text())
    directory=ROOT/('pilot' if pilot else 'ppo_'+condition);directory.mkdir(exist_ok=False)
    (directory/'config.yaml').write_text(yaml.safe_dump(config))
    phases=[('alignment',40000),('static',5000),('dynamic',5000)] if pilot else [('alignment',200000),('static',200000),('dynamic',600000)]
    workers=4;model=None;best={'score':[-1.,-1e30],'steps':None};started=time.time()
    for phase,budget in phases:
        env=SubprocVecEnv([functools.partial(make_worker,config,scenes,phase,42+i*1000,budget//workers,str(directory/f'{phase}_{i}.monitor.csv')) for i in range(workers)],start_method='spawn')
        try:
            if model is None:
                pc=config['ppo']
                model=PPO('MlpPolicy',env,device='cpu',seed=42,n_steps=250,batch_size=100,verbose=0,
                    tensorboard_log=str(directory/'tensorboard'),
                    policy_kwargs={'net_arch':pc['net_arch'],'activation_fn':torch.nn.ReLU,'log_std_init':pc['log_std_init']},
                    **{k:pc[k] for k in ['learning_rate','n_epochs','gamma','gae_lambda','clip_range','ent_coef']})
            else:model.set_env(env)
            target=model.num_timesteps+budget
            model.learn(total_timesteps=budget,reset_num_timesteps=False,
                callback=Progress(directory,validation,config,phase,target,best),tb_log_name=phase)
            assert model.num_timesteps==target
            model.save(directory/f'{phase}_final_model')
        finally:env.close()
    model.save(directory/'final_model')
    dump(directory/'training.json',{'steps':model.num_timesteps,'elapsed_seconds':time.time()-started,'seed':42,'phases':phases,'pilot':pilot})
    print('FINISHED',directory,model.num_timesteps,flush=True)


def evaluate():
    torch.set_num_threads(1)
    folder=Path('results/raw/dynamic70_main');folder.mkdir(parents=True,exist_ok=False)
    scenes=json.loads((BANK/'test.json').read_text());rows=[]
    models={name:PPO.load(ROOT/f'ppo_{name}'/'best_model.zip',device='cpu') for name in ['clean','noise']}
    for condition in ['clean','noise']:
        config=yaml.safe_load((ROOT/'frozen_config.yaml').read_text())
        if condition=='noise':config['noise']={'position_std':.005,'orientation_std_deg':.5}
        (folder/f'{condition}_config.yaml').write_text(yaml.safe_dump(config))
        for method in ['rrt','ppo_clean','ppo_noise']:
            env=WallPanelEnv(config)
            try:
                for scene in scenes:
                    model=None if method=='rrt' else models[method.removeprefix('ppo_')]
                    r,t=rollout(env,'rrt' if method=='rrt' else 'ppo',scene['seed'],model,scenario=scene)
                    r.update(method=method,evaluation_noise=condition,run_id='dynamic70_main')
                    rows.append(r);pd.DataFrame(rows).to_csv(folder/'episodes.csv',index=False)
                    if scene is scenes[0]:dump(folder/f'{condition}_{method}_trajectory.json',t)
                    if len(rows)%10==0:print('evaluation',len(rows),condition,method,flush=True)
            finally:env.close()
    dump(folder/'test_scenes.json',scenes)
    frame=pd.DataFrame(rows)
    summary=[]
    for (condition,method),group in frame.groupby(['evaluation_noise','method']):
        summary.append({'condition':condition,'method':method,'episodes':len(group),'success_rate':float(group.success.mean()),
          'collision_rate':float(group.collision.mean()),'mean_position_error_m':float(group.position_error_m.mean()),
          'mean_orientation_error_deg':float(group.orientation_error_deg.mean()),'mean_steps':float(group.steps.mean())})
    dump(folder/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','pilot','train','evaluate'])
    p.add_argument('--condition',choices=['clean','noise'],default='clean');a=p.parse_args()
    if a.command=='prepare':prepare()
    elif a.command=='evaluate':evaluate()
    else:train(a.condition,a.command=='pilot')
