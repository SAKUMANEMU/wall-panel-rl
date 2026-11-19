"""Bounded local recovery with reference-following and paired-joint candidates.

Uses the same rollback, collision definition and execution budget as v21.
No global RRT fallback is introduced. Existing modes are tried first.
"""
import numpy as np
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController,task_cost
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot

class DiverseRecoveryController(ProgressRecoveryController):
    def candidate_modes(self):
        modes=[('goal',g) for g in sorted(self.recovery.get('goal_gains',[.4,.7,1.]),reverse=True)]
        bias=float(self.recovery.get('joint_bias',.4))
        modes += [('bias',(j,s*bias)) for j in range(7) for s in (-1,1)]
        modes += [('waypoint',g) for g in (.4,.7,1.)]
        for a,b in ((1,3),(1,5),(3,5),(0,2)):
            modes += [('paired',([a,b],[sa*bias,sb*bias])) for sa in (-1,1) for sb in (-1,1)]
        return modes

    def _candidate(self,kind,value,observation):
        if kind=='paired':return super()._candidate('bias',value,observation)
        if kind!='waypoint':return super()._candidate(kind,value,observation)
        e=self.env.env;actions=[];safe=True;success=False;cost=task_cost(e.info())
        for _ in range(self.recovery.get('horizon',48)):
            q,_=e.joint_state()
            action=(np.clip((self.env.ref-q)/e.max_increment,-1,1)*value).astype(np.float32)
            actions.append(action)
            observation,_,done,truncated,info=self.env.step(action)
            self.prediction_steps+=1;cost=task_cost(info)
            if info['collision'] or info['joint_limit_violation']:safe=False;break
            if done or truncated:success=info['termination_reason']=='success';break
        return safe,success,cost,actions

    def _find_plan(self,observation,baseline_cost):
        best=None;best_rank=None
        with EpisodeSnapshot(self.env) as snapshot:
            for kind,value in self.candidate_modes():
                snapshot.restore();self.env.env.render_mode=None
                safe,success,cost,actions=self._candidate(kind,value,observation.copy())
                if not safe:continue
                if not success and baseline_cost-cost<self.recovery.get('minimum_predicted_gain',.015):continue
                rank=(int(success),-cost,-len(actions))
                if best_rank is None or rank>best_rank:best_rank=rank;best=(kind,value,cost,actions)
                if success:break
        return best
