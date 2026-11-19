"""Dynamic corrective imitation from exact PPO prefixes, with full replay.

Prefix actions are reproduced from the original reset. Time, obstacle phase,
robot velocities and reference/recovery state are never reset at takeover.
Only the successful expert suffix supplies supervised labels.
"""
import copy,time
import numpy as np
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv
from wall_panel_rl.rl.episode_imitation import make_rrt,episode_record
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot,assert_planning_paused


def reset_case(env,case):
    if case['task']!='dynamic':raise ValueError('Only dynamic cases allowed')
    return env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})


def trace_policy(config,settings,case,model):
    env=HybridFourEnv(copy.deepcopy(config),[],copy.deepcopy(settings));states=[]
    try:
        obs,_=reset_case(env,case)
        while True:
            action=np.asarray(model.predict(obs,deterministic=True)[0],dtype=np.float32)
            states.append({'observation':obs.copy(),'action':action.copy(),'q':env.env.joint_state()[0].copy(),
                'time':env.env.sim_time,'reference_index':int(env.cursor.index)})
            obs,_,done,truncated,info=env.step(action)
            if done or truncated:break
        return {'seed':int(case['scene']['seed']),'reason':info['termination_reason'],'states':states,'steps':env.env.steps}
    finally:env.close()


def restore_prefix(env,case,prefix,expected=None):
    obs,info=reset_case(env,case)
    for action in prefix:
        obs,_,done,truncated,info=env.step(action)
        if done or truncated:raise ValueError('PPO prefix terminates before takeover')
    error=0.
    if expected is not None:
        error=max(float(np.max(np.abs(obs-expected['observation']))),float(np.max(np.abs(env.env.joint_state()[0]-expected['q']))))
        if error>1e-6 or not np.isclose(env.env.sim_time,expected['time'],atol=1e-12,rtol=0) or env.cursor.index!=expected['reference_index']:
            raise AssertionError('Prefix did not reproduce the original policy state')
    return obs,info,error


def replay_correction(config,settings,case,prefix,pairs,q,times):
    env=HybridFourEnv(copy.deepcopy(config),[],copy.deepcopy(settings));error=0.;steps=0
    try:
        obs,info,_=restore_prefix(env,case,prefix)
        error=max(error,float(np.max(np.abs(env.env.joint_state()[0]-q[0]))))
        if not np.isclose(env.env.sim_time,times[0],atol=1e-12,rtol=0):return False,steps,error
        for i,(expected,action) in enumerate(pairs):
            error=max(error,float(np.max(np.abs(obs-expected))))
            obs,_,done,truncated,info=env.step(action);steps+=1
            error=max(error,float(np.max(np.abs(env.env.joint_state()[0]-q[i+1]))),
                      float(np.max(np.abs(np.asarray(info['executed_action'])-action))))
            if not np.isclose(env.env.sim_time,times[i+1],atol=1e-12,rtol=0):return False,steps,error
            if (done or truncated) and i!=len(pairs)-1:return False,steps,error
        return bool(pairs and info['termination_reason']=='success' and error<=1e-6),steps,error
    finally:env.close()


def corrective_episode(config,settings,case,trace,handoff,planner_seed):
    if not 0<=handoff<len(trace['states']):raise ValueError('Invalid takeover index')
    prefix=[s['action'].copy() for s in trace['states'][:handoff]]
    env=HybridFourEnv(copy.deepcopy(config),[],copy.deepcopy(settings));pairs=[];replay_error=None;replay_steps=0;post=False;replayed=False
    try:
        obs,info,prefix_error=restore_prefix(env,case,prefix,trace['states'][handoff])
        reference=env.cursor.path.copy();before=planning_snapshot(env);start_time=env.env.sim_time
        expert=make_rrt(env,int(planner_seed));assert_planning_paused(env,before)
        if not expert.timed:raise AssertionError('Corrective expert must use time-aware planning')
        pre=bool(expert.planning_success)
        if pre:
            pre=all(expert.planner.edge_valid(a,b,start_time+i*expert.planner.dt) for i,(a,b) in enumerate(zip(expert.path[:-1],expert.path[1:])))
            assert_planning_paused(env,before)
        total=0.;q=[env.env.joint_state()[0].copy()];times=[env.env.sim_time]
        if pre:
            while True:
                state=planning_snapshot(env);action=expert.action(obs);assert_planning_paused(env,state)
                previous=obs.copy();obs,reward,done,truncated,info=env.step(action);total+=reward
                pairs.append((previous,np.asarray(info['executed_action'],dtype=np.float32)))
                q.append(env.env.joint_state()[0].copy());times.append(env.env.sim_time)
                if done or truncated:break
            if info['termination_reason']=='success':
                after=planning_snapshot(env)
                post=all(expert.planner.edge_valid(a,b,t) for a,b,t in zip(q[:-1],q[1:],times[:-1]))
                assert_planning_paused(env,after)
                replayed,replay_steps,replay_error=replay_correction(config,settings,case,prefix,pairs,q,times)
        else:info=dict(info,termination_reason='planning_failure')
        preserved=np.array_equal(reference,env.cursor.path)
        if not preserved:raise AssertionError('Corrective expert modified the immutable reference')
        row=episode_record(env,info,'rrt_correction',total,expert.planning_time)
        row.update(online_avoidance_planner=True,handoff=int(handoff),handoff_time_s=float(start_time),prefix_steps=len(prefix),suffix_steps=len(pairs),
            prefix_error=prefix_error,source_reason=trace['reason'],planner_seed=int(planner_seed),
            pre_verified=bool(pre),post_time_verified=bool(post),physical_replay_verified=bool(replayed),
            replay_steps=replay_steps,replay_max_error=replay_error,reference_preserved=preserved,
            accepted_for_imitation=bool(pre and post and replayed and info['termination_reason']=='success'),
            dynamic_planning_mode='timed')
        return {'row':row,'case':copy.deepcopy(case),'prefix':prefix,'pairs':pairs if row['accepted_for_imitation'] else [],'q':q,'times':times}
    finally:env.close()
