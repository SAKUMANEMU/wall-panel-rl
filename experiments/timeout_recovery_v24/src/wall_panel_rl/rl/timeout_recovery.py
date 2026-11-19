"""Deadline-aware, finite-horizon local recovery; no global route planner.

Tries persistent detours and fixed future waypoints on the immutable reference.
Every candidate uses physical rollout/rollback and the original collision rule.
"""
from collections import deque
import numpy as np
import pybullet as p
from wall_panel_rl.rl.expanded_recovery import DiverseRecoveryController
from wall_panel_rl.rl.progress_recovery import task_cost
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot


class TimeoutRecoveryController(DiverseRecoveryController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.real_history=deque(maxlen=self.recovery.get('window',16)+1)
        self.last_real_action=None

    def _observe(self):
        super()._observe();step=self.env.env.steps
        if not self.real_history or self.real_history[-1][0]!=step:
            stopped=self.last_real_action is not None and np.max(np.abs(self.last_real_action))<1e-8
            self.real_history.append((step,task_cost(self.env.env.info()),stopped))

    def _stalled(self):
        if super()._stalled():return True
        r=self.recovery;step=self.env.env.steps
        if len(self.real_history)<self.real_history.maxlen:return False
        waiting=float(np.mean([x[2] for x in self.real_history]))
        gain=self.real_history[0][1]-self.real_history[-1][1]
        return (step>=r.get('minimum_step',40) and waiting>=.6 and gain<.003 and
                self.rescues<r.get('maximum_rescues',6) and step-self.last_rescue>=r.get('cooldown',12))

    def action(self,observation):
        result=super().action(observation)
        self.last_real_action=np.asarray(result).copy()
        return result

    def candidate_modes(self):
        modes=super().candidate_modes()
        # A target is a local control goal, not a changed observation reference.
        modes += [('lookahead',(offset,gain)) for offset in (0,3,6,12) for gain in (.4,.7,1.)]
        for j in range(7):
            modes += [('persistent',([j],[sign*.28])) for sign in (-1,1)]
        for a,b in ((1,3),(1,5),(3,5),(0,2)):
            modes += [('persistent',([a,b],[sa*.28,sb*.28])) for sa in (-1,1) for sb in (-1,1)]
        return modes

    def _candidate(self,kind,value,observation):
        if kind not in ('lookahead','persistent'):return super()._candidate(kind,value,observation)
        e=self.env.env;path=self.env.cursor.path
        index=min(self.env.cursor.index+(value[0] if kind=='lookahead' else 3),len(path)-1)
        q,_=e.joint_state();staging=q.copy()
        if kind=='persistent':staging[value[0]]+=np.asarray(value[1])
        staging=np.clip(staging,e.lower+.01,e.upper-.01)
        actions=[];success=False;safe=True;cost=task_cost(e.info())
        for tick in range(self.recovery.get('horizon',60)):
            q,_=e.joint_state()
            if kind=='persistent' and tick<16 and np.linalg.norm(staging-q)>.04:
                target=staging;gain=.7
            else:
                # Continue along actual reference segments rather than returning
                # immediately to the policy that produced the original stall.
                while index<len(path)-1 and np.linalg.norm(path[index]-q)<.09:index+=1
                target=path[index];gain=value[1] if kind=='lookahead' else .7
            command=(np.clip((target-q)/e.max_increment,-1,1)*gain).astype(np.float32)
            actions.append(command)
            observation,_,done,truncated,info=self.env.step(command)
            self.prediction_steps+=1;cost=task_cost(info)
            if info['collision'] or info['joint_limit_violation']:safe=False;break
            if done or truncated:success=info['termination_reason']=='success';break
        return safe,success,cost,actions

    def _clearance(self):
        e=self.env.env;distance=.25
        for body in e.obstacle_ids:
            for moving in (e.robot,e.panel):
                contacts=p.getClosestPoints(moving,body,.25,physicsClientId=e.client)
                if contacts:distance=min(distance,min(float(c[8]) for c in contacts))
        return distance

    def _find_plan(self,observation,baseline_cost):
        initial_cost=task_cost(self.env.env.info());initial_clearance=self._clearance()
        initial_index=self.env.cursor.index;q0=self.env.env.joint_state()[0].copy()
        best=None;best_rank=None
        with EpisodeSnapshot(self.env) as snapshot:
            for kind,value in self.candidate_modes():
                snapshot.restore();self.env.env.render_mode=None
                safe,success,cost,actions=self._candidate(kind,value,observation.copy())
                if not safe:continue
                clearance=self._clearance();forward=self.env.cursor.index-initial_index
                moved=np.linalg.norm(self.env.env.joint_state()[0]-q0)
                normal=baseline_cost-cost>=self.recovery.get('minimum_predicted_gain',.015)
                # An explicitly bounded detour may temporarily worsen target
                # distance, but must actually move and create space or advance.
                detour=(self.recovery.get('allow_temporary_detour',False) and moved>=.06 and
                    cost<=initial_cost+.12 and
                    ((clearance-initial_clearance>=.015) or (forward>=2 and cost<=baseline_cost+.03)))
                if not success and not normal and not detour:continue
                value_rank=(int(success),-cost+.25*min(clearance,.15)+.005*min(forward,6),-len(actions))
                if best_rank is None or value_rank>best_rank:
                    best_rank=value_rank;best=(kind,value,cost,actions)
                if success:break
        return best
