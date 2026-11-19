"""Installation-first repairs over the restored v29 execution controller.

The old recovery search is tried first. Unfinished reference-escape detours
are rejected; additional wait/rejoin plans require a valid progress forecast. A proactive finish is accepted only after
the same physical simulator predicts actual successful installation. Reference,
episode budget, collision checking and installation tolerances stay unchanged.
"""
import numpy as np
from wall_panel_rl.planning.ik import solve_goal
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot
from wall_panel_rl.rl.progress_recovery import task_cost
from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController


class InstallationRecoveryController(TimeoutRecoveryController):
    def __init__(self,*args,installation=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.installation=dict(installation or {})
        self.installation_events=[]
        self.finish_attempt_step=-1000
        self.finish_active=False

    def _roll_targets(self,targets,gain,horizon,wait=0):
        targets=np.asarray(targets);e=self.env.env;index=0
        actions=[];success=False;cost=task_cost(e.info())
        for tick in range(min(horizon,e.config['simulation']['max_steps']-e.steps)):
            q,_=e.joint_state()
            while index<len(targets)-1 and np.linalg.norm(targets[index]-q)<.07:index+=1
            command=(np.zeros(7) if tick<wait else
                     np.clip((targets[index]-q)/e.max_increment,-1,1)*gain).astype(np.float32)
            actions.append(command)
            _,_,done,truncated,info=self.env.step(command);self.prediction_steps+=1
            cost=task_cost(info)
            if info['collision'] or info['joint_limit_violation']:return False,False,cost,actions
            if done or truncated:
                success=info['termination_reason']=='success';break
        return True,success,cost,actions

    def _rejoin_index(self):
        path=self.env.cursor.path;start=self.env.cursor.index
        q,_=self.env.env.joint_state()
        distances=np.linalg.norm(path[start:]-q,axis=1)
        closest=int(np.argmin(distances))
        if distances[closest]>self.installation.get('maximum_rejoin_distance_rad',.65):return None
        return min(start+closest+1,len(path)-1)

    def _candidate(self,kind,value,observation):
        if kind!='installation_rejoin':return super()._candidate(kind,value,observation)
        index,gain,wait=value
        return self._roll_targets(self.env.cursor.path[index:],gain,
                                  self.recovery.get('horizon',60),wait)

    def _loses_reference(self,plan,observation):
        """Reject a nonterminal detour that strands a stale reference farther away.

        Cartesian target distance alone can improve while the joint tracker
        diverges. Successful forecasts and advancing routes remain admissible.
        """
        e=self.env.env;path=self.env.cursor.path;index=self.env.cursor.index
        q=e.joint_state()[0]
        initial_route=float(np.min(np.linalg.norm(path[index:]-q,axis=1)))
        initial_goal=float(np.linalg.norm(path[-1]-q))
        with EpisodeSnapshot(self.env):
            e.render_mode=None
            safe,success,_,_=self._candidate(plan[0],plan[1],observation.copy())
            if success:return False
            q=e.joint_state()[0]
            end_route=float(np.min(np.linalg.norm(path[index:]-q,axis=1)))
            loses=(not safe or (self.env.cursor.index==index and
                end_route>max(.6,initial_route+.3) and
                np.linalg.norm(path[-1]-q)>initial_goal+.2))
        return bool(loses)

    def _find_plan(self,observation,baseline_cost):
        original=super()._find_plan(observation,baseline_cost)
        if not self.installation.get('enabled',True):return original
        if original is not None:
            if not self._loses_reference(original,observation):return original
            self.installation_events.append(dict(step=self.env.env.steps,
                mode='reject_reference_escape',candidate_mode=original[0]))
        index=self._rejoin_index()
        if index is None:return None
        e=self.env.env;initial_cost=task_cost(e.info());initial_index=self.env.cursor.index
        best=None;rank=None
        with EpisodeSnapshot(self.env) as snapshot:
            for wait in self.installation.get('wait_steps',[0,4,8,12]):
                for gain in (1.,.7):
                    snapshot.restore();self.env.env.render_mode=None
                    value=(index,gain,wait)
                    safe,success,cost,actions=self._candidate('installation_rejoin',value,observation)
                    if not safe:continue
                    forward=self.env.cursor.index-initial_index
                    q=e.joint_state()[0]
                    reserve=int(np.ceil(np.max(np.abs(self.env.cursor.path[-1]-q))/e.max_increment))+8
                    remaining=e.config['simulation']['max_steps']-e.steps
                    improved=initial_cost-cost>=self.installation.get('minimum_task_gain',.02)
                    if not success and not (improved and forward>0 and remaining>=reserve):continue
                    value_rank=(int(success),-cost,forward,-len(actions))
                    if rank is None or value_rank>rank:
                        rank=value_rank;best=('installation_rejoin',value,cost,actions)
                    if success:return best
        return best

    def _finish_plan(self,observation):
        e=self.env.env;remaining=e.config['simulation']['max_steps']-e.steps
        horizon=min(remaining,self.installation.get('finish_horizon',80))
        info=e.info();near=info['position_error']<=self.installation.get('terminal_position_m',.12)
        # IK searches another joint configuration for the same observed pose.
        # It never relaxes the true physical installation success condition.
        fitted=solve_goal(e,e.observed_target,seed=int(self.env.last_scene['seed'])+e.steps) if near else None
        modes=[('goal_joint',[self.env.cursor.path[-1]],g) for g in (1.,.7,.4)]
        if fitted is not None:modes=[('terminal_pose',[fitted],g) for g in (.7,1.,.4)]+modes
        index=self._rejoin_index()
        if index is not None:modes.append(('reference_finish',self.env.cursor.path[index:],.7))
        with EpisodeSnapshot(self.env) as snapshot:
            for label,targets,gain in modes:
                snapshot.restore();self.env.env.render_mode=None
                safe,success,cost,actions=self._roll_targets(targets,gain,horizon)
                if safe and success:return label,gain,cost,actions
        return None

    def action(self,observation):
        if not self.installation.get('enabled',True):return super().action(observation)
        e=self.env.env;self._observe()
        if self.finish_active and self.plan:
            if self._planned_safe(observation):
                result=self._plan_action();self.last_real_action=result.copy();return result
            self.plan=[]
        if self.finish_active and not self.plan:self.finish_active=False
        info=e.info();remaining=e.config['simulation']['max_steps']-e.steps
        eligible=(info['position_error']<=self.installation.get('terminal_position_m',.12) or
                  remaining<=self.installation.get('deadline_remaining_steps',64))
        if (eligible and not self.plan and
            e.steps-self.finish_attempt_step>=self.installation.get('finish_retry_steps',12)):
            self.finish_attempt_step=e.steps
            found=self._finish_plan(observation)
            if found is not None:
                label,gain,cost,actions=found
                self.plan=actions;self.plan_mode=label;self.plan_executed=0;self.finish_active=True
                self.installation_events.append(dict(step=e.steps,mode=label,gain=gain,
                    predicted_success=True,predicted_end_cost=cost,plan_steps=len(actions)))
                result=self._plan_action();self.last_real_action=result.copy();return result
        return super().action(observation)
