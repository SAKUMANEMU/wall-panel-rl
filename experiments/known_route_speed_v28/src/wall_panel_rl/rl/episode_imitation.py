"""Episode-level RRT/PPO training with strictly separated data streams.

Successful RRT episodes run in reset(), before returning a fresh PPO episode.
SB3 therefore receives ONLY PPO-generated transitions and their original action
log probabilities. Supervised updates run at the NEXT rollout start, before any
new PPO sampling; never while collecting an on-policy rollout.
"""
from collections import Counter,deque
import copy
import json
from pathlib import Path
import time
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.rl.episode_policy import EpisodeChoice
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv


def make_rrt(env,seed):
    """Use the existing planner, targeting the policy reference's SAME endpoint."""
    return ClassicalController(env.env,int(seed),goal_q=env.cursor.path[-1])


def episode_record(env,info,controller,reward,planning_seconds=0.):
    return {'controller':controller,'seed':int(env.last_scene['seed']),'task':env.task,
        'reason':info['termination_reason'],'steps':int(env.env.steps),'reward':float(reward),
        'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
        'planning_seconds':float(planning_seconds),'online_avoidance_planner':controller=='rrt',
        'conditional_recoveries':int(info.get('conditional_recovery_count',0))}


class ExpertEpisodeEnv(gym.Wrapper):
    """Expose PPO episodes to SB3; journal every selected RRT/PPO episode.

    The underlying scene/task RNG is shared across both episode types. Teacher
    actions never overwrite the saved obstacle-free observation reference. A
    failed plan counts as a failed RRT episode, without relabelling or fallback.
    """
    def __init__(self,env,*,probability=.5,seed=61,journal=None,max_expert_streak=100):
        super().__init__(env)
        if not 0<=probability<1:raise ValueError('Training must allow PPO episodes')
        self.choice=EpisodeChoice(probability,seed)
        self.planner_rng=np.random.default_rng(seed+100000)
        self.journal=Path(journal) if journal else None
        self.max_expert_streak=max_expert_streak
        self.pending=[];self.records=[];self.counts=Counter();self.started=Counter()
        self.episode_reward=0.;self.start_time=time.perf_counter();self.ppo_active=False

    def _record(self,row):
        self.records.append(row);self.counts[(row['controller'],row['reason'])]+=1
        if self.journal:
            self.journal.parent.mkdir(parents=True,exist_ok=True)
            with self.journal.open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def reset(self,*,seed=None,options=None):
        self.choice.end_episode()
        self.ppo_active=False
        for _ in range(self.max_expert_streak):
            obs,info=self.env.reset(seed=seed,options=options);seed=None
            controller=self.choice.begin_episode();self.started[controller]+=1
            if controller=='ppo':
                self.ppo_active=True;self.episode_reward=0.
                info['episode_controller']='ppo'
                return obs,info
            expert=make_rrt(self.env,self.planner_rng.integers(0,2**31-1))
            trajectory=[];total=0.
            if expert.planning_success:
                while True:
                    previous=obs.copy();action=expert.action(obs)
                    obs,reward,done,truncated,info=self.env.step(action);total+=reward
                    trajectory.append((previous,np.asarray(info['executed_action'],dtype=np.float32)))
                    if done or truncated:break
            else:info=dict(info,termination_reason='planning_failure')
            row=episode_record(self.env,info,'rrt',total,expert.planning_time)
            row['accepted_for_imitation']=row['reason']=='success'
            self._record(row)
            if row['accepted_for_imitation']:self.pending.extend(trajectory)
            self.choice.end_episode()
        raise RuntimeError('Expert streak limit reached; all attempts are journalled, no silent fallback')

    def step(self,action):
        if not self.ppo_active:raise RuntimeError('Reset required for a PPO episode')
        obs,reward,done,truncated,info=self.env.step(action);self.episode_reward+=reward
        info['episode_controller']='ppo'
        if done or truncated:
            row=episode_record(self.env,info,'ppo',self.episode_reward);self._record(row)
            info['episode']={'r':self.episode_reward,'l':row['steps'],'t':time.perf_counter()-self.start_time}
            self.ppo_active=False;self.choice.end_episode()
        return obs,reward,done,truncated,info

    def drain_demonstrations(self):
        result=self.pending;self.pending=[];return result


def evaluate_controller(model,config,settings,cases,mode='mixture',probability=.5,seed=61):
    """Paired case order + common Bernoulli assignments across checkpoints."""
    if mode not in ('ppo','rrt','mixture'):raise ValueError('Unknown controller mode')
    choice=EpisodeChoice(probability,seed);rows=[]
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings)
    try:
        for case in cases:
            obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            controller=choice.begin_episode() if mode=='mixture' else mode
            expert=make_rrt(env,int(case['scene']['seed'])+100000) if controller=='rrt' else None
            total=0.
            if expert is not None and not expert.planning_success:
                info=dict(info,termination_reason='planning_failure')
            else:
                while True:
                    action=expert.action(obs) if expert else model.predict(obs,deterministic=True)[0]
                    obs,reward,done,truncated,info=env.step(action);total+=reward
                    if done or truncated:break
            row=episode_record(env,info,controller,total,expert.planning_time if expert else 0.)
            row['suite']=case['suite'];rows.append(row);choice.end_episode()
    finally:env.close()
    def summarize(group):
        return {'episodes':len(group),**{k:sum(r['reason']==k for r in group) for k in
            ('success','collision','timeout','joint_limit','planning_failure')}}
    return {'mode':mode,'probability':probability,'selection_seed':seed,'training_performed':False,
        'independent_test':False,'rows':rows,
        'metrics':{suite:summarize([r for r in rows if r['suite']==suite]) for suite in sorted({r['suite'] for r in rows})},
        'by_controller':{name:summarize([r for r in rows if r['controller']==name]) for name in ('rrt','ppo')}}


class AuxiliaryImitation(BaseCallback):
    """Separate actor-only Adam updates, bounded by a sampled Gaussian KL check."""
    def __init__(self,output,spec,seed=61):
        super().__init__();self.output=Path(output);self.spec=spec;self.seed=seed
        if (min(spec['capacity'],spec['batch_size'],spec['checkpoint_every'])<=0 or
            spec['updates_per_rollout']<0 or spec['weight']<0 or
            min(spec['learning_rate'],spec['maximum_kl'])<=0):
            raise ValueError('Invalid auxiliary imitation settings')
        self.pool=deque(maxlen=int(spec['capacity']));self.rng=np.random.default_rng(seed)
        self.accepted=0;self.rejected=0;self.events=[]

    def _on_step(self):return True

    def _on_training_start(self):
        self.start=self.model.num_timesteps
        prefixes=getattr(self.model.policy,'imitation_prefixes',('residual.','prior_gate.'))
        self.params=[p for name,p in self.model.policy.named_parameters()
                     if name.startswith(prefixes) and p.requires_grad]
        if not self.params:raise ValueError('Imitation requires the reference-preserving residual actor')
        self.optimizer=torch.optim.Adam(self.params,lr=self.spec['learning_rate'])
        self.output.mkdir(parents=True,exist_ok=True)

    def save_state(self,name):
        # Separate from PPO's optimizer. The pool is saved for audit as well.
        torch.save({'optimizer':self.optimizer.state_dict(),'rng':self.rng.bit_generator.state,
            'accepted_updates':self.accepted,'rejected_updates':self.rejected,'pool':list(self.pool),
            'ppo_num_timesteps':self.model.num_timesteps},self.output/name)

    def _on_rollout_start(self):
        for pairs in self.training_env.env_method('drain_demonstrations'):self.pool.extend(pairs)
        added=self.model.num_timesteps-self.start
        if added%self.spec['checkpoint_every']==0:self.save_state(f'aux_{added}.pt')
        if not self.pool:return
        policy=self.model.policy
        policy.set_training_mode(True)
        try:
            for _ in range(self.spec['updates_per_rollout']):
                batch=[self.pool[int(k)] for k in self.rng.integers(0,len(self.pool),self.spec['batch_size'])]
                obs=torch.as_tensor(np.stack([v[0] for v in batch]),device=self.model.device)
                target=torch.as_tensor(np.stack([v[1] for v in batch]),device=self.model.device)
                guard=torch.cat([obs,torch.as_tensor(self.model._last_obs,device=self.model.device)])
                with torch.no_grad():
                    dist=policy.get_distribution(guard).distribution
                    old=torch.distributions.Normal(dist.loc.detach().clone(),dist.scale.detach().clone())
                weights=[p.detach().clone() for p in self.params]
                optimizer_state=copy.deepcopy(self.optimizer.state_dict())
                loss=self.spec['weight']*torch.nn.functional.mse_loss(policy.tracking_mean(obs),target)
                self.optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params,.5);self.optimizer.step()
                with torch.no_grad():
                    new=policy.get_distribution(guard).distribution
                    kl=float(torch.distributions.kl_divergence(old,new).sum(-1).mean())
                accepted=np.isfinite(kl) and kl<=self.spec['maximum_kl']
                if not accepted:
                    with torch.no_grad():
                        for p,w in zip(self.params,weights):p.copy_(w)
                    self.optimizer.load_state_dict(optimizer_state);self.rejected+=1
                else:self.accepted+=1
                event={'ppo_additional_steps':added,'pool_size':len(self.pool),'loss':float(loss.detach()),
                       'sampled_kl':kl,'accepted':bool(accepted)}
                self.events.append(event)
                with (self.output/'updates.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                if not accepted:break
        finally:policy.set_training_mode(False)

    def _on_training_end(self):
        for pairs in self.training_env.env_method('drain_demonstrations'):self.pool.extend(pairs)
        self.save_state('final_aux_state.pt')
