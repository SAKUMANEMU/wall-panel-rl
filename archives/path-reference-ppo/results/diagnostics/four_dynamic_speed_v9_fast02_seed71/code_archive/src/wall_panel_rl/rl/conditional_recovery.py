"""Stagnation-triggered, bounded widening of reference recovery; inference experiment.

Only updates the local reference observation, never the action, physical state,
collision verdict, goal, reward definition or terminal servo. No route planner.
"""
from collections import deque
import numpy as np
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv
from wall_panel_rl.rl.waypoint_tracking import reference_observation
from wall_panel_rl.rl.wall_feedback import WallAwareEnv


class RecoveryGate:
    def __init__(self,window=30,min_steps=60,position_epsilon=.001,orientation_epsilon_deg=.1,
                 max_joint_displacement=.02,min_reference_distance=.2,extended_radius=.45,
                 max_attempts=3,max_recoveries=1,cooldown=20):
        if min(window,min_steps,max_attempts,max_recoveries,cooldown)<1 or min(
                position_epsilon,orientation_epsilon_deg,max_joint_displacement,min_reference_distance,extended_radius)<=0:
            raise ValueError('Invalid conditional recovery settings')
        self.window=window;self.min_steps=min_steps;self.position_epsilon=position_epsilon
        self.orientation_epsilon=np.deg2rad(orientation_epsilon_deg)
        self.max_joint_displacement=max_joint_displacement;self.min_reference_distance=min_reference_distance
        self.extended_radius=extended_radius;self.max_attempts=max_attempts;self.max_recoveries=max_recoveries
        self.cooldown=cooldown;self.history=deque(maxlen=window+1)
        self.attempts=0;self.recoveries=0;self.last_attempt=-cooldown

    def observe(self,step,q,position,orientation,index,last_index,reference_distance,*,obstacles_present=True,terminal=False):
        self.history.append((np.asarray(q,dtype=float).copy(),float(position),float(orientation),int(index)))
        if terminal or not obstacles_present or step<self.min_steps or len(self.history)<self.window+1:return False
        if self.attempts>=self.max_attempts or self.recoveries>=self.max_recoveries or step-self.last_attempt<self.cooldown:return False
        first,last=self.history[0],self.history[-1]
        # Use actual target progress here: apparent progress toward a stale
        # reference must not prevent detection of movement away from the goal.
        no_progress=(first[1]-last[1]<self.position_epsilon and first[2]-last[2]<self.orientation_epsilon)
        little_motion=np.linalg.norm(last[0]-first[0])<=self.max_joint_displacement
        stale=(first[3]==last[3] and index<last_index and reference_distance>=self.min_reference_distance)
        return bool(no_progress and little_motion and stale)

    def record_attempt(self,step,recovered):
        self.attempts+=1;self.last_attempt=step
        if recovered:self.recoveries+=1


class ConditionalRecoveryEnv(StagedAvoidanceEnv):
    # Dynamic scenes must opt in through an explicitly validated environment.
    supports_dynamic_obstacles=False

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        if self.task=='dynamic' and not self.supports_dynamic_obstacles:
            raise ValueError('Conditional recovery pilot is restricted to static scenes')
        spec=self.settings['conditional_recovery']
        if spec['extended_radius']<=self.settings['reference_recovery']['radius']:
            raise ValueError('Extended radius must exceed the unchanged normal radius')
        self.recovery_gate=RecoveryGate(**spec);self.conditional_events=[]
        q,_=self.env.joint_state()
        self.recovery_gate.observe(0,q,info['position_error'],info['orientation_error'],self.cursor.index,
            len(self.cursor.path)-1,np.linalg.norm(self.ref-q),obstacles_present=bool(self.env.obstacle_ids))
        return obs,info

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        q,_=self.env.joint_state();old=self.cursor.index
        allowed=self.recovery_gate.observe(self.env.steps,q,info['position_error'],info['orientation_error'],
            old,len(self.cursor.path)-1,np.linalg.norm(self.ref-q),obstacles_present=bool(self.env.obstacle_ids),
            terminal=done or truncated)
        recovered=False
        if allowed:
            radius=self.cursor.radius
            try:
                self.cursor.radius=self.recovery_gate.extended_radius
                new_reference=self.cursor.reference(q)
            finally:self.cursor.radius=radius
            recovered=self.cursor.index>old
            self.recovery_gate.record_attempt(self.env.steps,recovered)
            if recovered:
                self.ref=new_reference
                obs=reference_observation(obs[:70],self.ref,self.env.lower,self.env.upper)
            event={'step':self.env.steps,'recovered':recovered,'old_index':old,'new_index':self.cursor.index,
                   'reference_distance_after_rad':float(np.linalg.norm(self.ref-q)),
                   'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error']))}
            self.conditional_events.append(event)
        info.update(conditional_recovery_attempted=allowed,conditional_recovery_triggered=recovered,
                    conditional_recovery_attempts=self.recovery_gate.attempts,
                    conditional_recovery_count=self.recovery_gate.recoveries,
                    reference_recoveries=self.cursor.recoveries)
        return obs,reward,done,truncated,info


class ConditionalTrainingEnv(ConditionalRecoveryEnv,WallAwareEnv):
    """Same reference rule with the existing wall/stall training rewards.

    Cooperative super order is Conditional -> WallAware -> StallAware -> Staged.
    Physics/action projection and both original shaping terms are evaluated
    first; conditional recovery changes only the next local reference input.
    The evaluation class above deliberately excludes extra training shaping.
    """
    pass
