"""Dynamic curriculum and verified RRT teacher, isolated from on-policy PPO data."""
import copy
import time
import numpy as np
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv,planning_snapshot,assert_planning_paused
from wall_panel_rl.rl.episode_imitation import ExpertEpisodeEnv,make_rrt,episode_record
from wall_panel_rl.rl.episode_policy import EpisodeChoice


def case_at_speed(original,speed,*,static=False,phase_rng=None,amplitude=None):
    """Keep anchor, size, robot and route; change only frequency and/or phase."""
    case=copy.deepcopy(original);scene=case['scene'];old_speed=float(case['strength'][1])
    old_amplitude=float(case['strength'][0]);new_amplitude=old_amplitude if amplitude is None else float(amplitude)
    if old_speed<=0 or speed<=0 or new_amplitude<=0 or new_amplitude>old_amplitude+1e-12:
        raise ValueError('Positive speed and amplitude within canonical motion bounds required')
    for box,motion in zip(scene['obstacles'],scene['obstacle_motion']):
        ratio=new_amplitude/old_amplitude
        if ratio!=1.:motion['extent']=(np.asarray(motion['extent'])*ratio).tolist()
        motion['omega']=(np.asarray(motion['omega'])*(speed/old_speed)/ratio).tolist()
        if phase_rng is not None:motion['phase']=phase_rng.uniform(0,2*np.pi,3).tolist()
        box['center']=(np.asarray(motion['anchor'])+np.asarray(motion['extent'])*np.sin(motion['phase'])).tolist()
    if static:scene['obstacle_motion']=[]
    else:scene['slow_dynamic_motion'].update(speed_bound_m_s=float(speed),amplitude_bound_m=new_amplitude)
    case.update(task='static' if static else 'dynamic',strength=[new_amplitude,0. if static else float(speed)])
    return case


class HybridFourEnv(SlowDynamicFourEnv):
    """Both static retention episodes and the validated slow-dynamic environment."""
    def reset(self,*,seed=None,options=None):
        if options and options.get('task')=='static':
            scene=options['scenario']
            if len(scene['obstacles'])!=4 or scene.get('obstacle_motion'):
                raise ValueError('Retention episode must have four static boxes')
            return ConditionalRecoveryEnv.reset(self,seed=seed,options=options)
        return super().reset(seed=seed,options=options)


class DynamicFourBankEnv(HybridFourEnv):
    """A speed change applies only to the next episode, never a running one."""
    def __init__(self,config,settings,cases,spec,*,seed,excluded=(),probabilities=None):
        ids=[c['scene']['seed'] for c in cases]
        if not ids or len(set(ids))!=len(ids) or set(ids)&set(excluded):raise ValueError('Invalid training split')
        if any(c['task']!='dynamic' or len(c['scene']['obstacles'])!=4 or len(c['scene']['obstacle_motion'])!=4 for c in cases):
            raise ValueError('Expected canonical four-dynamic cases')
        super().__init__(copy.deepcopy(config),[],copy.deepcopy(settings))
        self.cases=copy.deepcopy(cases);self.spec=spec;self.level=0;self.episode_level=0
        self.probabilities=None if probabilities is None else np.array(probabilities,dtype=float)
        if self.probabilities is not None and (self.probabilities.shape!=(len(cases),) or
            not np.isfinite(self.probabilities).all() or np.any(self.probabilities<0) or not np.isclose(self.probabilities.sum(),1)):
            raise ValueError('Invalid scene probabilities')
        self._seed_streams(seed)

    def _seed_streams(self,seed):
        self.case_rng=np.random.default_rng(seed+3000000)
        self.mode_rng=np.random.default_rng(seed+4000000)
        self.phase_rng=np.random.default_rng(seed+5000000)

    def set_curriculum(self,level):
        if not 0<=int(level)<len(self.spec['curriculum_speeds']):raise ValueError('Invalid level')
        self.level=int(level)

    def reset(self,*,seed=None,options=None):
        if options:raise ValueError('Training cannot override the case bank')
        if seed is not None:self._seed_streams(seed)
        i=self.case_rng.choice(len(self.cases),p=self.probabilities)
        self.episode_level=self.level;speed=self.spec['curriculum_speeds'][self.episode_level]
        amplitude=self.spec.get('curriculum_amplitudes',[self.cases[0]['strength'][0]]*len(self.spec['curriculum_speeds']))[self.episode_level]
        static=self.mode_rng.random()<self.spec['static_probability']
        case=case_at_speed(self.cases[int(i)],speed,static=static,phase_rng=self.phase_rng,amplitude=amplitude)
        return super().reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})


def replay_expert(config,settings,case,pairs,expected_q,expected_times):
    """Fresh physics replay; compare original observations, executed labels and time."""
    env=HybridFourEnv(copy.deepcopy(config),[],copy.deepcopy(settings));steps=0;error=0.
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        for index,(original_obs,action) in enumerate(pairs):
            error=max(error,float(np.max(np.abs(obs-original_obs))))
            obs,_,done,truncated,info=env.step(action);steps+=1
            error=max(error,float(np.max(np.abs(env.env.joint_state()[0]-expected_q[index+1]))))
            if not np.isclose(env.env.sim_time,expected_times[index+1],atol=1e-12,rtol=0):return False,steps,error
            if (done or truncated) and index!=len(pairs)-1:return False,steps,error
        return bool(pairs and info['termination_reason']=='success' and error<=1e-6),steps,error
    finally:env.close()


def verified_expert_episode(env,planner_seed,initial_observation=None):
    """Timed precheck + actual execution + timestamp check + fresh action replay."""
    before=planning_snapshot(env);reference=env.cursor.path.copy();case={'scene':copy.deepcopy(env.last_scene),
        'task':env.task,'strength':[env.amplitude,env.speed]}
    expert=make_rrt(env,planner_seed);assert_planning_paused(env,before)
    pre=bool(expert.planning_success);verification_seconds=0.;replay_steps=0;replay_error=None
    if pre and expert.timed:
        begin=time.perf_counter();dt=expert.planner.dt
        pre=all(expert.planner.edge_valid(a,b,env.env.sim_time+i*dt) for i,(a,b) in enumerate(zip(expert.path[:-1],expert.path[1:])))
        verification_seconds+=time.perf_counter()-begin;assert_planning_paused(env,before)
    elif pre:pre=bool(expert.static_path_verified)
    pairs=[];total=0.;q=[env.env.joint_state()[0].copy()];times=[env.env.sim_time]
    if initial_observation is None:
        from wall_panel_rl.rl.waypoint_tracking import reference_observation
        obs=reference_observation(env.env.observation(),env.ref,env.env.lower,env.env.upper)
    else:obs=np.asarray(initial_observation,dtype=np.float32).copy()
    if obs.shape!=(77,):raise ValueError('Expected original 77-feature observation')
    info=env.env.info();post=False;replayed=False
    if pre:
        while True:
            saved=planning_snapshot(env);command=expert.action(obs);assert_planning_paused(env,saved)
            previous=obs.copy();obs,reward,done,truncated,info=env.step(command);total+=reward
            pairs.append((previous,np.asarray(info['executed_action'],dtype=np.float32)))
            q.append(env.env.joint_state()[0].copy());times.append(env.env.sim_time)
            if done or truncated:break
        if info['termination_reason']=='success':
            begin=time.perf_counter();after=planning_snapshot(env)
            post=not expert.timed or all(expert.planner.edge_valid(a,b,t) for a,b,t in zip(q[:-1],q[1:],times[:-1]))
            assert_planning_paused(env,after)
            replayed,replay_steps,replay_error=replay_expert(env.env.config,env.settings,case,pairs,q,times)
            verification_seconds+=time.perf_counter()-begin
    else:info=dict(info,termination_reason='planning_failure')
    preserved=np.array_equal(reference,env.cursor.path)
    if not preserved:raise AssertionError('Expert modified the immutable reference')
    row=episode_record(env,info,'rrt',total,expert.planning_time)
    row.update(planner_seed=int(planner_seed),pre_verified=bool(pre),post_time_verified=bool(post),
        physical_replay_verified=bool(replayed),replay_steps=replay_steps,replay_max_error=replay_error,
        verification_seconds=verification_seconds,reference_preserved=preserved,
        accepted_for_imitation=bool(pre and post and replayed and row['reason']=='success'),
        dynamic_planning_mode='timed' if expert.timed else 'static_verified',
        replans=expert.replans,hold_steps=expert.hold_steps,scheduled_waits=expert.scheduled_waits)
    return row,pairs


class DynamicFourEpisodeMixture(ExpertEpisodeEnv):
    def __init__(self,*args,reward_scale=.01,**kwargs):
        super().__init__(*args,**kwargs)
        if reward_scale<=0:raise ValueError('Positive reward scale required')
        self.reward_scale=reward_scale

    def reset(self,*,seed=None,options=None):
        self.choice.end_episode();self.ppo_active=False
        for _ in range(self.max_expert_streak):
            obs,info=self.env.reset(seed=seed,options=options);seed=None
            owner=self.choice.begin_episode();self.started[owner]+=1
            if owner=='ppo':
                self.ppo_active=True;self.episode_reward=0.
                return obs,dict(info,episode_controller='ppo')
            row,pairs=verified_expert_episode(self.env,int(self.planner_rng.integers(0,2**31-1)),obs)
            row.update(speed_m_s=self.env.speed,curriculum_level=self.env.episode_level)
            self._record(row)
            if row['accepted_for_imitation']:self.pending.extend(pairs)
            self.choice.end_episode()
        raise RuntimeError('Expert streak limit reached; no branch fallback')

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        # Journal metadata is added in _record, before writing the line.
        info['unscaled_reward']=float(reward)
        if 'episode' in info:info['episode']=dict(info['episode'],r=info['episode']['r']*self.reward_scale)
        return obs,float(reward*self.reward_scale),done,truncated,info

    def _record(self,row):
        row.update(speed_m_s=self.env.speed,amplitude_m=self.env.amplitude,curriculum_level=self.env.episode_level)
        super()._record(row)


def curriculum_gate(current,baseline,static,static_baseline,target,target_baseline):
    """Advance at most one level after a whole rollout, with static/target retention."""
    checks={}
    for name,a,b in (('current',current,baseline),('static',static,static_baseline),('target',target,target_baseline)):
        checks[name]=bool(a['joint_limit']==0 and a['success']>=b['success'] and a['collision']<=b['collision'])
    return {'passed':all(checks.values()),'checks':checks}


def counts(rows):
    return {'episodes':len(rows),**{reason:sum(r['reason']==reason for r in rows) for reason in
        ('success','collision','timeout','joint_limit','planning_failure')}}


def evaluate_ppo(model,config,settings,cases,speed,*,static=False,amplitude=None):
    env=HybridFourEnv(copy.deepcopy(config),[],copy.deepcopy(settings));rows=[]
    try:
        for original in cases:
            case=case_at_speed(original,speed,static=static,amplitude=amplitude)
            obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            reference=env.cursor.path.copy();total=0.;last=env.cursor.index;stale=0;max_stale=0
            while True:
                obs,reward,done,truncated,info=env.step(model.predict(obs,deterministic=True)[0]);total+=reward
                stale=stale+1 if last==env.cursor.index else 0;last=env.cursor.index;max_stale=max(max_stale,stale)
                if done or truncated:break
            assert np.array_equal(reference,env.cursor.path)
            row=episode_record(env,info,'ppo',total)
            row.update(speed_m_s=case['strength'][1],amplitude_m=case['strength'][0],reference_index=int(env.cursor.index),reference_last=len(reference)-1,
                reference_distance_rad=float(np.linalg.norm(env.ref-env.env.joint_state()[0])),
                max_unchanged_reference_steps=max_stale,conditional_recovery_attempts=info.get('conditional_recovery_attempts',0))
            rows.append(row)
    finally:env.close()
    return {'rows':rows,'metrics':counts(rows),'independent_test':False,'training_performed':False}


def expert_cache(config,settings,cases,speed,seed):
    """Actual RRT execution once for the common episode assignments, no BC labels."""
    env=HybridFourEnv(copy.deepcopy(config),[],copy.deepcopy(settings));choice=EpisodeChoice(.5,seed);rows=[];owners=[]
    try:
        for original in cases:
            owner=choice.begin_episode();owners.append({'seed':original['scene']['seed'],'controller':owner});choice.end_episode()
            if owner!='rrt':continue
            case=case_at_speed(original,speed)
            obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
            before=planning_snapshot(env);expert=make_rrt(env,int(case['scene']['seed'])+100000);assert_planning_paused(env,before)
            assert expert.timed;total=0.
            if not expert.planning_success:info=dict(info,termination_reason='planning_failure')
            else:
                while True:
                    before=planning_snapshot(env);action=expert.action(obs);assert_planning_paused(env,before)
                    obs,reward,done,truncated,info=env.step(action);total+=reward
                    if done or truncated:break
            rows.append(episode_record(env,info,'rrt',total,expert.planning_time))
    finally:env.close()
    return {'owners':owners,'rrt_rows':rows,'note':'Independent expert branch physically executed once; reused unchanged in paired mixture comparisons. Never a training demonstration.'}


def compose_mixture(pure,cache):
    ppo={r['seed']:r for r in pure['rows']};rrt={r['seed']:r for r in cache['rrt_rows']}
    rows=[copy.deepcopy((rrt if r['controller']=='rrt' else ppo)[r['seed']]) for r in cache['owners']]
    return {'rows':rows,'metrics':counts(rows),'by_controller':{name:counts([r for r in rows if r['controller']==name]) for name in ('rrt','ppo')},
        'expert_probability':.5,'cached_expert_execution':True,'no_mid_episode_switch':True}
