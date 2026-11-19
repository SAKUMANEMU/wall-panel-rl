"""Gated avoidance policy, explicit stage environments, and promotion criteria."""
import copy
import json
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl.autonomous_dynamic_ppo import (
    DynamicResidualPolicy, SpacesOnly, AutonomousDynamicEnv,
    scenario_at_strength, without_obstacles, evaluate, ReferenceCursor)
from wall_panel_rl.rl.waypoint_tracking import WaypointTrackingPolicy
from wall_panel_rl.utils.config import load_config


INTERFACE_CONTRACT='reference_preserving_v3'


class GatedAvoidancePolicy(DynamicResidualPolicy):
    """Learn when to slow the prior as well as how to deviate from it."""
    def __init__(self,*args,initial_prior_gate=.5,**kwargs):
        if not 0<initial_prior_gate<1:raise ValueError('Initial gate must be in (0,1)')
        self.initial_prior_gate=float(initial_prior_gate)
        super().__init__(*args,**kwargs)
        self.prior_gate=nn.Sequential(nn.Linear(77,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.prior_gate[-1].weight)
        nn.init.constant_(self.prior_gate[-1].bias,float(np.log(initial_prior_gate/(1-initial_prior_gate))))
        for name,param in self.named_parameters():
            param.requires_grad_(name.startswith(('residual.','prior_gate.','mlp_extractor.value_net.','value_net.')))
        schedule=kwargs.get('lr_schedule',args[2] if len(args)>2 else None)
        self.optimizer=self.optimizer_class(self.parameters(),lr=schedule(1),**self.optimizer_kwargs)

    def tracking_mean(self,obs):
        prior=WaypointTrackingPolicy.tracking_mean(self,obs)
        gate=torch.sigmoid(self.prior_gate(obs))
        return (gate*prior+self.residual_scale*torch.tanh(self.residual(obs))).clamp(-1,1)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),initial_prior_gate=self.initial_prior_gate)


class ReferencePreservingPolicy(GatedAvoidancePolicy):
    """Obstacle correction preserves free-space servo and vanishes at the goal.

    All branches use the same mean for BC, PPO likelihoods and predict. The mask
    only denotes obstacle presence; it is not a collision-based action filter.
    Squared goal taper makes learned correction second order near the endpoint.
    Old GatedAvoidancePolicy checkpoints keep their original semantics.
    """
    def __init__(self,*args,goal_angles,terminal_radius=.20,**kwargs):
        goal=np.asarray(goal_angles,dtype=float)
        if goal.shape!=(7,) or not np.isfinite(goal).all() or terminal_radius<=0:
            raise ValueError('Invalid terminal servo settings')
        self.goal_angles=goal.tolist();self.terminal_radius=float(terminal_radius)
        super().__init__(*args,**kwargs)

    def reference_mean(self,obs):
        prior=WaypointTrackingPolicy.tracking_mean(self,obs)
        lower=obs.new_tensor(self.joint_lower);span=obs.new_tensor(self.joint_upper)-lower
        q=lower+(obs[:,:7]+1)*.5*span
        error=obs.new_tensor(self.goal_angles)-q
        taper=(error.abs().amax(dim=1,keepdim=True)/self.terminal_radius).clamp(0,1)
        # The fixed endpoint servo has no bias floor and no dependence on a
        # potentially stale local waypoint. It replaces the prior smoothly.
        endpoint=(F.softplus(self.tracking_gain_raw)*error/self.joint_increment).clamp(-1,1)
        return taper*prior+(1-taper)*endpoint

    def correction_components(self,obs):
        lower=obs.new_tensor(self.joint_lower);span=obs.new_tensor(self.joint_upper)-lower
        q=lower+(obs[:,:7]+1)*.5*span
        taper=((q-obs.new_tensor(self.goal_angles)).abs().amax(dim=1,keepdim=True)
               /self.terminal_radius).clamp(0,1).square()
        present=(obs[:,20:70].reshape(-1,5,10)[:,:,9]>.5).any(dim=1,keepdim=True).to(obs.dtype)
        strength=present*taper
        gate=torch.sigmoid(self.prior_gate(obs))
        residual=self.residual_scale*torch.tanh(self.residual(obs))
        return strength,gate,residual

    def tracking_mean(self,obs):
        prior=self.reference_mean(obs)
        strength,gate,residual=self.correction_components(obs)
        return ((1-strength+strength*gate)*prior+strength*residual).clamp(-1,1)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),goal_angles=self.goal_angles,
                    terminal_radius=self.terminal_radius)


def initialize_gated(source,s):
    old=PPO.load(source,device='cpu');p=old.policy
    kwargs=dict(old.policy_kwargs)
    kwargs.update(joint_lower=p.joint_lower,joint_upper=p.joint_upper,
        joint_increment=p.joint_increment,joint_margin=p.joint_margin,initial_gain=p.initial_gain,
        residual_scale=s['residual_scale'],initial_prior_gate=s['initial_prior_gate'])
    policy_class=GatedAvoidancePolicy
    if s.get('interface_contract')==INTERFACE_CONTRACT:
        policy_class=ReferencePreservingPolicy
        kwargs.update(goal_angles=load_config(s['environment'])['robot']['goal_q'],
                      terminal_radius=s['terminal_radius'])
    model=PPO(policy_class,DummyVecEnv([SpacesOnly for _ in range(s['workers'])]),
        device='cpu',seed=s['seed'],n_steps=s['n_steps'],batch_size=s['batch_size'],
        n_epochs=s['n_epochs'],learning_rate=s['learning_rate'],clip_range=s['clip_range'],
        target_kl=s['target_kl'],ent_coef=0.,policy_kwargs=kwargs)
    missing,unexpected=model.policy.load_state_dict(old.policy.state_dict(),strict=False)
    if unexpected or any(not k.startswith(('residual.','prior_gate.')) for k in missing):
        raise ValueError('Source architecture cannot be transferred')
    with torch.no_grad():model.policy.log_std.fill_(np.log(s['action_std']))
    model.num_timesteps=old.num_timesteps
    return model


class RecoveringReferenceCursor(ReferenceCursor):
    """Monotone nearest-forward-segment recovery after a detour, no planning.

    The polyline is still the original obstacle-free reference. Recovery changes
    the observed local waypoint, never a submitted action or collision verdict.
    Earliest equal-distance segment wins to avoid gratuitous jumps at crossings.
    """
    def __init__(self,path,radius=.30,min_distance=.08,improvement=.02):
        if radius<=0 or min_distance<0 or improvement<=0:
            raise ValueError('Invalid reference recovery settings')
        super().__init__(path)
        self.radius=radius;self.min_distance=min_distance;self.improvement=improvement
        self.recoveries=0

    def reference(self,q):
        q=np.asarray(q,dtype=float)
        if self.index<len(self.path)-1 and np.linalg.norm(q-self.path[self.index])>self.min_distance:
            starts=self.path[self.index-1:-1];vectors=np.diff(self.path[self.index-1:],axis=0)
            lengths=np.sum(vectors*vectors,axis=1)
            fractions=np.clip(np.sum((q-starts)*vectors,axis=1)/np.maximum(lengths,1e-12),0,1)
            distances=np.linalg.norm(q-(starts+fractions[:,None]*vectors),axis=1)
            candidate=int(np.argmin(distances))
            if (candidate>0 and distances[candidate]<=self.radius and
                distances[candidate]+self.improvement<distances[0]):
                self.index+=candidate;self.recoveries+=1
        return super().reference(q)


def stage_scene(original,stage,rng=None):
    scene=scenario_at_strength(original,stage['amplitude'],stage['speed'],rng)
    # Two-box bridge retains the trained middle obstacle in observation slot 0.
    indices={1:[2],2:[2,3],3:[1,2,3],5:[0,1,2,3,4]}[stage['count']]
    if len(scene['obstacles'])!=5 or len(scene['obstacle_motion'])!=5:
        raise ValueError('Expected a full five-obstacle source scene')
    scene['obstacles']=[scene['obstacles'][i] for i in indices]
    scene['obstacle_motion']=[scene['obstacle_motion'][i] for i in indices]
    scene['obstacle_original_indices']=indices
    if stage['mode']=='static':scene['obstacle_motion']=[]
    return scene


def clearance_feedback(env,safe_distance,weight):
    """Only obstacles, all arm links + payload; never penalize intended wall clearance."""
    if safe_distance<=0 or weight<0:raise ValueError('Invalid clearance settings')
    if not env.obstacle_ids:return 0.,None
    closest=safe_distance
    for obstacle in env.obstacle_ids:
        for body in (env.robot,env.panel):
            for contact in p.getClosestPoints(body,obstacle,safe_distance,physicsClientId=env.client):
                closest=min(closest,float(contact[8]))
    proximity=float(np.clip(1-closest/safe_distance,0,1))
    return -weight*proximity**2,closest


class StagedAvoidanceEnv(AutonomousDynamicEnv):
    def create_reference_cursor(self,path):
        if self.settings.get('interface_contract')==INTERFACE_CONTRACT:
            return RecoveringReferenceCursor(path,**self.settings['reference_recovery'])
        return super().create_reference_cursor(path)

    def reset(self,*,seed=None,options=None):
        if seed is not None:self.rng=np.random.default_rng(seed)
        if options and 'scenario' in options:
            return super().reset(seed=seed,options=options)
        stage=self.settings['active_stage']
        task=str(self.rng.choice(['alignment','static','dynamic'],
             p=[self.settings['mixture'][k] for k in ('alignment','static','dynamic')]))
        scene=self.scenes[int(self.rng.integers(len(self.scenes)))]
        selected=stage_scene(scene,stage,self.rng)
        if task=='alignment':
            remaining=self.rng.uniform(.02,.15) if self.rng.random()<.5 else None
            selected=without_obstacles(selected,remaining)
        elif stage['mode']=='static' or task=='static':
            selected['obstacle_motion']=[];task='static'
        return super().reset(options={'scenario':selected,'task':task,
            'strength':[stage['amplitude'],stage['speed']]})

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        penalty,distance=clearance_feedback(self.env,self.settings['clearance_distance'],
                                            self.settings['clearance_weight'])
        info.update(reward_obstacle_clearance=penalty,obstacle_clearance_capped_m=distance,
                    reward_total=float(reward+penalty),
                    reference_recoveries=getattr(self.cursor,'recoveries',0))
        return obs,float(reward+penalty),done,truncated,info


def gate_result(metrics,gate):
    """Promotion checks counts and errors explicitly; it is not an independent test."""
    target=metrics['stage_target']
    rates={reason:target[reason]/max(1,target['episodes']) for reason in
           ('success','collision','joint_limit','timeout')}
    alignment={key:metrics[key]['success']/max(1,metrics[key]['episodes']) for key in
               ('alignment_far','alignment_near')}
    checks={'valid_exclusive_counts':all(target[k]>=0 for k in rates) and sum(target[k] for k in rates)==target['episodes'],
        'enough_target_episodes':target['episodes']>=gate['minimum_episodes'],
        'success':rates['success']>=gate['minimum_success_rate'],
        'collision':rates['collision']<=gate['maximum_collision_rate'],
        'no_joint_limits':target['joint_limit']==0,
        'alignment_samples':all(metrics[k]['episodes']>=gate['minimum_alignment_episodes'] for k in alignment),
        'alignment_retained':min(alignment.values())>=gate['minimum_alignment_success_rate']}
    return {'passed':all(checks.values()),'checks':checks,'rates':rates,'alignment':alignment}


def assess_history(history,gate):
    # Baseline/imitation scores cannot alone unlock a longer or harder PPO stage.
    updates=[r for r in history if r['kind'] in ('post_update','final_saved_reload')]
    required=gate['consecutive_checks']
    latest=updates[-required:]
    # A duplicate callback event at the same step does not count as extra evidence.
    unique=len({r['additional_steps'] for r in latest})==required
    results=[gate_result(r['metrics']['dynamic_ppo'],gate) for r in latest]
    return {'passed':len(results)==required and unique and all(r['passed'] for r in results),
        'enough_distinct_checks':len(results)==required and unique,'checks':results,
        'additional_steps':[r['additional_steps'] for r in latest],'thresholds':gate,
        'automatic_stage_advance':False}


class StagedValidation(BaseCallback):
    def __init__(self,output,source,config,settings,cases,source_steps):
        super().__init__();self.output=Path(output);self.source=source;self.config=config
        self.settings=settings;self.cases=cases;self.source_steps=source_steps
        self.best=None;self.last=None;self.history=[]
    def _on_step(self):return True
    def _on_training_start(self):self.score('baseline')
    def _on_rollout_start(self):
        added=self.model.num_timesteps-self.source_steps
        if added>0 and added%self.settings['eval_every']==0 and added!=self.last:self.score('post_update')
    def _on_rollout_end(self):
        (self.output/'sampling_progress.json').write_text(json.dumps({
            'additional_steps':self.model.num_timesteps-self.source_steps}))
    def _on_training_end(self):
        self.model.save(self.output/'final_model.zip')
        self.score('final_saved_reload',PPO.load(self.output/'final_model.zip',device='cpu'))
    def score(self,kind,model=None):
        model=self.model if model is None else model;added=model.num_timesteps-self.source_steps
        metrics,rows=evaluate(model,self.source,self.config,self.settings,self.cases,
                              env_factory=StagedAvoidanceEnv)
        event={'kind':kind,'additional_steps':added,'total_steps':model.num_timesteps,'metrics':metrics,
               'gate':gate_result(metrics['dynamic_ppo'],self.settings['gate'])}
        self.history.append(event)
        with (self.output/'validation.jsonl').open('a') as f:f.write(json.dumps(event,allow_nan=False)+'\n')
        with (self.output/'validation_episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps({'kind':kind,'additional_steps':added,**row})+'\n')
        path=self.output/'checkpoints'/f'checkpoint_{added}.zip';path.parent.mkdir(exist_ok=True);model.save(path)
        current=metrics['dynamic_ppo'];target=current['stage_target']
        score=(int(event['gate']['passed']),target['success'],-target['collision'],
            current['alignment_far']['success']+current['alignment_near']['success'],target['mean_reward'])
        if self.best is None or score>self.best:
            self.best=score;model.save(self.output/'best_model.zip')
            (self.output/'selection.json').write_text(json.dumps(event,indent=2))
        (self.output/'progress.json').write_text(json.dumps(event,indent=2));self.last=added
        print(json.dumps(event),flush=True)
