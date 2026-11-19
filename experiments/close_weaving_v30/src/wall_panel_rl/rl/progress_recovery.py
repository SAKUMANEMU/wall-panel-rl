"""Progress-aware local recovery on top of the frozen predictive controller.

Only stalled episodes enter local recovery. A full baseline continuation first
protects trajectories which would already succeed. All trial physics is rolled
back; actual actions retain original collisions, limits and episode budget.
No global RRT is called, and no reference waypoint/path is overwritten.
"""
from collections import deque
import time
import numpy as np
import pybullet as p
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot,PredictiveSafetyController


def task_cost(info):return float(info['position_error']+.12*info['orientation_error'])


class ProgressRecoveryController(PredictiveSafetyController):
    def __init__(self,model,env,horizon=12,gains=(1.,.75,.4,0.,-.25,-.75),recovery=None):
        super().__init__(model,env,horizon,gains)
        self.recovery=dict(recovery or {});self.history=deque(maxlen=self.recovery.get('window',24)+1)
        self.protected=False;self.rescues=0;self.last_rescue=-100000;self.plan=[];self.plan_mode=None;self.plan_executed=0
        self.recovery_events=[];self.forecast_calls=0;self.recovery_actions=0
        self.forecast_cache={};self.forecast_reason=None;self.forecast_cache_hits=0
    def _observe(self):
        step=self.env.env.steps;cost=task_cost(self.env.env.info())
        if not self.history or self.history[-1][0]!=step:self.history.append((step,cost))
    def _stalled(self):
        r=self.recovery;step=self.env.env.steps
        if len(self.history)<self.history.maxlen:return False
        progress=self.history[0][1]-self.history[-1][1]
        span=max(1,self.history[-1][0]-self.history[0][0])
        remaining=self.env.env.config['simulation']['max_steps']-step
        budget_risk=(self.history[-1][1]>.03 and
            self.history[-1][1]/max(progress/span,1e-6)>max(1,remaining-r.get('commit_steps',8)))
        return (step>=r.get('minimum_step',100) and len(self.history)==self.history.maxlen and
            (progress<r.get('minimum_progress',.012) or budget_risk) and
            self.rescues<r.get('maximum_rescues',3) and step-self.last_rescue>=r.get('cooldown',24))
    def _baseline_forecast(self,observation):
        """Counterfactual baseline continuation uses only the simulator model."""
        current_step=self.env.env.steps
        saved=self.forecast_cache.get(current_step)
        if saved is not None and np.max(np.abs(saved[0]-observation))<=1e-6:
            target=min(current_step+self.recovery.get('horizon',36),max(self.forecast_cache))
            self.forecast_cache_hits+=1
            return self.forecast_reason,self.forecast_cache[target][1]
        self.forecast_calls+=1;short_cost=task_cost(self.env.env.info());reason='timeout'
        cache={current_step:(observation.copy(),short_cost)}
        child=PredictiveSafetyController(self.model,self.env,self.horizon,self.gains)
        with EpisodeSnapshot(self.env):
            self.env.env.render_mode=None
            obs=observation.copy();remaining=self.env.env.config['simulation']['max_steps']-self.env.env.steps
            for tick in range(remaining):
                action=child.action(obs);obs,_,done,truncated,info=self.env.step(action);self.prediction_steps+=1
                cache[self.env.env.steps]=(obs.copy(),task_cost(info))
                if tick+1==self.recovery.get('horizon',36):short_cost=task_cost(info)
                if done or truncated:
                    reason=info['termination_reason']
                    if tick+1<self.recovery.get('horizon',36):short_cost=task_cost(info)
                    break
        self.prediction_steps+=child.prediction_steps
        self.forecast_cache=cache;self.forecast_reason=reason
        return reason,short_cost
    def _planned_safe(self,observation):
        with EpisodeSnapshot(self.env):
            obs=observation.copy()
            for i in range(self.horizon):
                command=self.plan[i] if i<len(self.plan) else self.model.predict(obs,deterministic=True)[0]
                obs,_,done,truncated,info=self.env.step(command);self.prediction_steps+=1
                if info['collision'] or info['joint_limit_violation']:return False
                if done or truncated:return info['termination_reason']=='success'
        return True
    def _candidate(self,kind,value,observation):
        e=self.env.env;actions=[];safe=True;success=False;end_cost=task_cost(e.info())
        for tick in range(self.recovery.get('horizon',36)):
            if kind=='goal':
                q,_=e.joint_state();command=np.clip((self.env.cursor.path[-1]-q)/e.max_increment,-1,1)*value
            else:
                command=np.asarray(self.model.predict(observation,deterministic=True)[0]).copy()
                if tick<self.recovery.get('commit_steps',8):command[value[0]]+=value[1]
            command=np.clip(command,-1,1).astype(np.float32);actions.append(command)
            observation,_,done,truncated,info=self.env.step(command);self.prediction_steps+=1;end_cost=task_cost(info)
            if info['collision'] or info['joint_limit_violation']:safe=False;break
            if done or truncated:success=info['termination_reason']=='success';break
        return safe,success,end_cost,actions
    def _find_plan(self,observation,baseline_cost):
        modes=[('goal',g) for g in sorted(self.recovery.get('goal_gains',[.4,.7,1.]),reverse=True)]
        modes += [('bias',(j,s*self.recovery.get('joint_bias',.4))) for j in range(7) for s in (-1,1)]
        best=None;best_value=None
        with EpisodeSnapshot(self.env) as snapshot:
            self.env.env.render_mode=None
            for kind,value in modes:
                snapshot.restore();self.env.env.render_mode=None
                safe,success,cost,actions=self._candidate(kind,value,observation.copy())
                if not safe:continue
                if not success and baseline_cost-cost<self.recovery.get('minimum_predicted_gain',.015):continue
                rank=(int(success),-cost,-len(actions))
                if best_value is None or rank>best_value:best_value=rank;best=(kind,value,cost,actions)
                if success:break
        return best
    def _plan_action(self):
        self.forecast_cache={};self.forecast_reason=None
        action=self.plan.pop(0);self.plan_executed+=1;self.calls+=1;self.interventions+=1;self.recovery_actions+=1
        key='recovery_'+str(self.plan_mode);self.gain_counts[key]=self.gain_counts.get(key,0)+1
        return action
    def action(self,observation):
        self._observe();start=time.perf_counter();e=self.env.env;human=e.render_mode=='human'
        if human:p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=e.client)
        special=None
        try:
            if self.plan:
                if self.plan_executed and self.plan_executed%self.recovery.get('commit_steps',8)==0:
                    reason,_=self._baseline_forecast(observation)
                    if reason=='success':self.protected=True;self.plan=[]
                if self.plan:
                    if self._planned_safe(observation):special=self._plan_action()
                    else:self.plan=[]
            if special is None and not self.protected and self._stalled():
                reason,baseline_cost=self._baseline_forecast(observation)
                if reason=='success':self.protected=True
                else:
                    self.last_rescue=e.steps;candidate=self._find_plan(observation,baseline_cost)
                    event={'step':e.steps,'baseline_forecast':reason,'baseline_short_cost':baseline_cost,'accepted':candidate is not None}
                    if candidate is not None:
                        kind,value,cost,actions=candidate;self.rescues+=1;self.plan=actions;self.plan_mode=kind;self.plan_executed=0
                        event.update(mode=kind,parameter=value,predicted_end_cost=cost,predicted_gain=baseline_cost-cost,plan_steps=len(actions))
                        if self._planned_safe(observation):special=self._plan_action()
                        else:self.plan=[];event['accepted']=False
                    self.recovery_events.append(event)
        finally:
            if human:p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1,physicsClientId=e.client)
        self.seconds+=time.perf_counter()-start
        if special is not None:return special
        return super().action(observation)
