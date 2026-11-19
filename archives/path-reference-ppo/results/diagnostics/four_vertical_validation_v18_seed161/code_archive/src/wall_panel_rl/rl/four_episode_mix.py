"""Episode-owned RRT/PPO execution with isolated supervised expert labels."""
import copy
import numpy as np
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import ExpertEpisodeEnv, make_rrt, episode_record, evaluate_controller
from wall_panel_rl.planning.static_validation import dense_route_valid


class FourCaseBankEnv(ConditionalRecoveryEnv):
    """Keep physical .env/.cursor interfaces required by the existing RRT."""
    def __init__(self,config,settings,cases,*,excluded=(),seed=61,probabilities=None):
        ids=[c['scene']['seed'] for c in cases]
        if not cases or len(set(ids))!=len(ids) or set(ids)&set(excluded):raise ValueError('Invalid train/heldout bank')
        if any(len(c['scene']['obstacles'])!=4 or c['scene']['obstacle_motion'] for c in cases):raise ValueError('Four static cases required')
        self.case_probabilities=None if probabilities is None else np.asarray(probabilities,dtype=float).copy()
        if self.case_probabilities is not None:
            p=self.case_probabilities
            if p.shape!=(len(cases),) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):raise ValueError('Invalid case probabilities')
        super().__init__(copy.deepcopy(config),[],copy.deepcopy(settings))
        self.cases=copy.deepcopy(cases);self.case_rng=np.random.default_rng(seed+3000000)

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('Training cannot override the frozen bank')
        if seed is not None:self.case_rng=np.random.default_rng(seed+3000000)
        i=self.case_rng.integers(len(self.cases)) if self.case_probabilities is None else self.case_rng.choice(len(self.cases),p=self.case_probabilities)
        case=self.cases[int(i)]
        return super().reset(options={'scenario':case['scene'],'task':'static','strength':case['strength']})


class FourEpisodeMixture(ExpertEpisodeEnv):
    """RRT episodes run fully in reset; SB3 receives only PPO transitions.

    Expert success alone is insufficient for BC labels: require the same fixed
    observation reference and pre/post verified static route. Failure stays an
    RRT failure. A new episode gets a fresh coin, never a within-episode fallback.
    """
    def __init__(self,*args,reward_scale=.01,**kwargs):
        if not np.isfinite(reward_scale) or reward_scale<=0:raise ValueError('Positive reward scale required')
        super().__init__(*args,**kwargs);self.reward_scale=reward_scale

    def reset(self,*,seed=None,options=None):
        self.choice.end_episode();self.ppo_active=False
        for _ in range(self.max_expert_streak):
            obs,info=self.env.reset(seed=seed,options=options);seed=None
            owner=self.choice.begin_episode();self.started[owner]+=1
            if owner=='ppo':
                self.ppo_active=True;self.episode_reward=0.;info['episode_controller']='ppo';return obs,info
            reference=self.env.cursor.path.copy();planner_seed=int(self.planner_rng.integers(0,2**31-1))
            expert=make_rrt(self.env,planner_seed);pairs=[];total=0.
            pre=bool(expert.planning_success and getattr(expert,'static_path_verified',False))
            if pre:
                while True:
                    previous=obs.copy();command=expert.action(obs)
                    obs,reward,done,truncated,info=self.env.step(command);total+=reward
                    pairs.append((previous,np.asarray(info['executed_action'],dtype=np.float32)))
                    if done or truncated:break
                post=dense_route_valid(self.env.env.collision,expert.path,expert.static_validation_resolution)
            else:info=dict(info,termination_reason='planning_failure');post=False
            preserved=np.array_equal(reference,self.env.cursor.path)
            if not preserved:raise AssertionError('Expert replaced the policy reference')
            row=episode_record(self.env,info,'rrt',total,expert.planning_time)
            row.update(planner_seed=planner_seed,pre_verified=pre,post_verified=bool(post),reference_preserved=preserved,
                       accepted_for_imitation=bool(pre and post and row['reason']=='success'))
            self._record(row)
            if row['accepted_for_imitation']:self.pending.extend(pairs)
            self.choice.end_episode()
        raise RuntimeError('Expert streak limit reached; no controller fallback was performed')

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        info['unscaled_reward']=float(reward)
        if 'episode' in info:info['episode']=dict(info['episode'],r=info['episode']['r']*self.reward_scale)
        return obs,float(reward*self.reward_scale),done,truncated,info


def evaluate_mix(model,config,settings,cases,*,probability=.5,seed=61,mode='mixture'):
    result=evaluate_controller(model,config,settings,cases,mode=mode,probability=probability,seed=seed)
    result.update(expert_branch='RRT-Connect',expert_is_bc_network=False,ownership_unit='episode',
                  probability_is_not_an_exact_quota=True)
    return result
