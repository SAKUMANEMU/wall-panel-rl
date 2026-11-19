"""Autonomous dynamic avoidance with a frozen BC/PPO tracking prior.

The obstacle-free RRT route is a reference, never an online avoidance controller.
Only the residual actor chooses waiting, deviations and obstacle avoidance.
"""
import copy
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.waypoint_tracking import reference_observation
from wall_panel_rl.rl.waypoint_ppo import RawWaypointPolicy, WaypointPPOEnv
from wall_panel_rl.utils.metrics import capture_scenario


class DynamicResidualPolicy(RawWaypointPolicy):
    def __init__(self,*args,residual_scale=1.,**kwargs):
        self.residual_scale=float(residual_scale)
        super().__init__(*args,**kwargs)
        self.residual=nn.Sequential(nn.Linear(77,64),nn.Tanh(),nn.Linear(64,64),
                                    nn.Tanh(),nn.Linear(64,7))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        for name,param in self.named_parameters():
            param.requires_grad_(name.startswith('residual.') or
                name.startswith('mlp_extractor.value_net.') or name.startswith('value_net.'))
        schedule=kwargs.get('lr_schedule',args[2] if len(args)>2 else None)
        self.optimizer=self.optimizer_class(self.parameters(),lr=schedule(1),**self.optimizer_kwargs)

    def tracking_mean(self,obs):
        base=super().tracking_mean(obs)
        return (base+self.residual_scale*torch.tanh(self.residual(obs))).clamp(-1,1)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),residual_scale=self.residual_scale)


class SpacesOnly(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)
    def reset(self,*,seed=None,options=None):return np.zeros(77,np.float32),{}
    def step(self,action):raise RuntimeError('Attach a real environment before learning')


def initialize_from_tracking(source,settings):
    old=PPO.load(source,device='cpu')
    policy=old.policy
    kwargs=dict(policy.policy_kwargs) if hasattr(policy,'policy_kwargs') else dict(old.policy_kwargs)
    kwargs.update(joint_lower=policy.joint_lower,joint_upper=policy.joint_upper,
                  joint_increment=policy.joint_increment,joint_margin=policy.joint_margin,
                  initial_gain=policy.initial_gain,residual_scale=settings['residual_scale'])
    model=PPO(DynamicResidualPolicy,DummyVecEnv([SpacesOnly for _ in range(settings['workers'])]),device='cpu',seed=settings['seed'],
        n_steps=settings['n_steps'],batch_size=settings['batch_size'],n_epochs=settings['n_epochs'],
        learning_rate=settings['learning_rate'],clip_range=settings['clip_range'],
        target_kl=settings['target_kl'],ent_coef=0.,policy_kwargs=kwargs)
    missing,unexpected=model.policy.load_state_dict(policy.state_dict(),strict=False)
    if unexpected or any(not key.startswith('residual.') for key in missing):
        raise ValueError('Unexpected checkpoint architecture mismatch')
    model.num_timesteps=old.num_timesteps
    return model


def difficulty(progress):
    # Hold easy difficulty for 20%, ramp to full at 80%, then retain full difficulty.
    fraction=float(np.clip((progress-.2)/.6,0,1))
    return .10+.30*fraction, .05+.35*fraction


def thin_reference(path,spacing=.055):
    """Preserve corners while thinning collinear dense RRT reference segments."""
    path=np.asarray(path,dtype=float)
    if len(path)<2:raise ValueError('A nonempty reference route is required')
    result=[path[0]]
    for i in range(1,len(path)-1):
        before=path[i]-path[i-1];after=path[i+1]-path[i]
        denom=np.linalg.norm(before)*np.linalg.norm(after)
        corner=denom>1e-12 and np.dot(before,after)/denom<.9999
        if corner or np.linalg.norm(path[i]-result[-1])>=spacing:result.append(path[i])
    result.append(path[-1])
    return np.asarray(result)


class ReferenceCursor:
    def __init__(self,path):self.path=thin_reference(path);self.index=1
    def reference(self,q):
        while self.index<len(self.path)-1:
            point=self.path[self.index];tangent=point-self.path[self.index-1]
            length=np.linalg.norm(tangent)
            distance=np.linalg.norm(q-point)
            passed=False
            if length>1e-9:
                along=np.dot(q-point,tangent)/length
                lateral=np.linalg.norm(q-point-along*tangent/length)
                passed=along>=0 and lateral<.20
            if distance<.008 or passed:self.index+=1
            else:break
        return self.path[self.index].copy()


def scenario_at_strength(original,amplitude,speed,phase_rng=None):
    scene=copy.deepcopy(original)
    for obstacle,spec in zip(scene['obstacles'],scene['obstacle_motion']):
        old_extent=np.asarray(spec['extent']);old_amplitude=np.linalg.norm(old_extent)
        ratio=np.asarray(spec['omega'])*old_amplitude/.40
        spec['extent']=np.full(3,amplitude/np.sqrt(3)).tolist()
        spec['omega']=(ratio*speed/amplitude).tolist()
        if phase_rng is not None:spec['phase']=phase_rng.uniform(0,2*np.pi,3).tolist()
        obstacle['center']=(np.asarray(spec['anchor'])+np.asarray(spec['extent'])*np.sin(spec['phase'])).tolist()
    return scene


def without_obstacles(original,remaining=None):
    scene=copy.deepcopy(original);scene['obstacles']=[];scene['obstacle_motion']=[]
    if remaining is not None:
        path=np.asarray(scene['reference_joint_path']);index=(1-remaining)*(len(path)-1)
        left=min(int(index),len(path)-2);f=index-left
        q=(1-f)*path[left]+f*path[left+1]
        scene['initial_q']=q.tolist();scene['reference_joint_path']=[q.tolist()]+path[left+1:].tolist()
    return scene


class AutonomousDynamicEnv(WaypointPPOEnv):
    def __init__(self,config,scenes,settings,rank=0):
        # Do not call static wrapper reset/planner: execution never invokes a planner.
        self.env=WallPanelEnv(config)
        self.action_space=self.env.action_space
        self.observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
        self.settings=settings;self.scenes=scenes
        self.rng=np.random.default_rng(settings['seed']+rank*1000)
        self.margin=settings['joint_margin'];self.progress_weight=settings['progress_weight']
        self.used=0;self.finished=True

    def create_reference_cursor(self,path):
        return ReferenceCursor(path)

    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        options=options or {}
        if 'scenario' in options:
            scene=copy.deepcopy(options['scenario']);task=options.get('task','dynamic')
            amplitude,speed=options.get('strength',(.4,.4))
        else:
            task=str(self.rng.choice(['alignment','static','dynamic'],
                       p=[self.settings['mixture'][k] for k in ('alignment','static','dynamic')]))
            progress=self.used/(self.settings['additional_timesteps']/self.settings['workers'])
            amplitude,speed=difficulty(progress)
            original=self.scenes[int(self.rng.integers(len(self.scenes)))]
            scene=scenario_at_strength(original,amplitude,speed,self.rng)
            if task=='alignment':
                remaining=self.rng.uniform(.02,.15) if self.rng.random()<.5 else None
                scene=without_obstacles(scene,remaining)
        self.task=task;self.amplitude=amplitude;self.speed=speed
        self.env.config['obstacles']['freeze_motion']=task=='static'
        self.env.config['obstacles']['dynamic'].update(amplitude=amplitude,max_speed=speed)
        obs,info=self.env.reset(seed=scene['seed'],options={'scenario':scene,
            'noise_seed':int(options.get('noise_seed',scene['seed']+1000000))})
        self.cursor=self.create_reference_cursor(scene['reference_joint_path'])
        self.ref=self.cursor.reference(self.env.joint_state()[0])
        self.finished=False;self.planning_success=True
        self.last_scene=scene
        info.update(curriculum_task=task,amplitude=amplitude,max_speed=speed,
                    online_avoidance_planner=False)
        return reference_observation(obs,self.ref,self.env.lower,self.env.upper),info

    def step(self,action):
        self.used+=1
        obs,reward,done,truncated,info=super().step(action)
        info.update(curriculum_task=self.task,amplitude=self.amplitude,max_speed=self.speed,
                    online_avoidance_planner=False)
        return obs,reward,done,truncated,info


def evaluate(candidate,source,config,settings,validation,env_factory=None):
    rows=[]
    env=(env_factory or AutonomousDynamicEnv)(config,[],settings)
    try:
        for case in validation:
            for method,model in [('reference_servo',None),('source_ppo',source),('dynamic_ppo',candidate)]:
                obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],
                    'strength':case['strength']})
                total=0.
                while True:
                    q,_=env.env.joint_state()
                    action=np.clip((env.ref-q)/env.env.max_increment,-1,1) if model is None else model.predict(obs,deterministic=True)[0]
                    obs,reward,done,truncated,info=env.step(action);total+=reward
                    if done or truncated:break
                rows.append({'method':method,'suite':case['suite'],'seed':case['scene']['seed'],
                    'reason':info['termination_reason'],'steps':env.env.steps,
                    'position_m':float(info['position_error']),
                    'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                    'reward':total,'online_avoidance_planner':False})
    finally:env.close()
    summaries={}
    for method in ('reference_servo','source_ppo','dynamic_ppo'):
        summaries[method]={}
        for suite in sorted({c['suite'] for c in validation}):
            group=[r for r in rows if r['method']==method and r['suite']==suite]
            timeouts=[r for r in group if r['reason']=='timeout']
            successes=[r for r in group if r['reason']=='success']
            summaries[method][suite]={'episodes':len(group),
                **{reason:sum(r['reason']==reason for r in group) for reason in ('success','collision','joint_limit','timeout')},
                'timeout_mean_position_m':float(np.mean([r['position_m'] for r in timeouts])) if timeouts else None,
                'timeout_mean_orientation_deg':float(np.mean([r['orientation_deg'] for r in timeouts])) if timeouts else None,
                'successful_mean_steps':float(np.mean([r['steps'] for r in successes])) if successes else None,
                'mean_reward':float(np.mean([r['reward'] for r in group]))}
    return summaries,rows


class DynamicValidation(BaseCallback):
    def __init__(self,output,source,config,settings,validation,source_steps):
        super().__init__();self.output=Path(output);self.source=source;self.config=config
        self.settings=settings;self.validation=validation;self.source_steps=source_steps
        self.best=None;self.last=None
    def _on_step(self):return True
    def _on_training_start(self):self.score('baseline')
    def _on_rollout_start(self):
        added=self.model.num_timesteps-self.source_steps
        if added>0 and added%self.settings['eval_every']==0 and added!=self.last:self.score('post_update')
    def _on_rollout_end(self):
        (self.output/'sampling_progress.json').write_text(json.dumps({
            'additional_steps':self.model.num_timesteps-self.source_steps,
            'total_steps':self.model.num_timesteps}))
    def _on_training_end(self):
        self.model.save(self.output/'final_model.zip')
        self.score('final_saved_reload',PPO.load(self.output/'final_model.zip',device='cpu'))
    def score(self,kind,model=None):
        model=self.model if model is None else model
        added=model.num_timesteps-self.source_steps
        metrics,rows=evaluate(model,self.source,self.config,self.settings,self.validation)
        record={'kind':kind,'additional_steps':added,'total_steps':model.num_timesteps,'metrics':metrics}
        with (self.output/'validation.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        with (self.output/'validation_episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps({'kind':kind,'additional_steps':added,**row})+'\n')
        checkpoint=self.output/'checkpoints'/f'checkpoint_{added}.zip'
        checkpoint.parent.mkdir(exist_ok=True);model.save(checkpoint)
        current=metrics['dynamic_ppo']
        score=(current['dynamic_full']['success'],current['dynamic_easy']['success'],
               -current['dynamic_full']['collision'],
               current['alignment_far']['success']+current['alignment_near']['success'],
               current['dynamic_full']['mean_reward'])
        if self.best is None or score>self.best:
            self.best=score;model.save(self.output/'best_model.zip')
            (self.output/'selection.json').write_text(json.dumps(record,indent=2))
        (self.output/'progress.json').write_text(json.dumps(record,indent=2));self.last=added
        print(json.dumps(record),flush=True)
