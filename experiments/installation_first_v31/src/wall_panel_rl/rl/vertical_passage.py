"""Bounded local Cartesian passage candidates over the unchanged joint policy.

IK proposes joint targets, never teleports the executed robot. Each candidate
is rolled through the physical environment and restored before real execution.
This is an explicit predictive-controller extension, not pure PPO or RRT.
"""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from wall_panel_rl.envs.known_route_motion import route_state
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot
from wall_panel_rl.rl.progress_recovery import task_cost
from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController


def panel_half_height(quaternion, dimensions):
    return float(np.abs(Rotation.from_quat(quaternion).as_matrix()[2]) @
                 (np.asarray(dimensions)/2))


class VerticalPassageController(TimeoutRecoveryController):
    def __init__(self,*args,passage=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.passage=dict(passage or {})
        self.vertical_attempt_step=-1000
        self.vertical_passed=set()
        self.vertical_events=[]
        self.vertical_plan=False
        self.finish_attempt_step=-1000

    def _encounter(self,slot):
        e=self.env.env;pos,quat=e.panel_pose()
        center,dimensions=e.obstacles[slot]
        record=self.env.last_scene['motion_anchor_placement']['obstacles'][slot]
        side=record['intended_side']
        half=panel_half_height(quat,e.config['panel']['dimensions'])
        clearance=(pos[2]-half-center[2]-dimensions[2]/2 if side=='above' else
                   center[2]-dimensions[2]/2-pos[2]-half)
        tangent=np.asarray(record['tangent_xy'])
        signed=float(np.dot(np.asarray(pos[:2])-center[:2],tangent))
        near=float(np.linalg.norm(np.asarray(pos[:2])-center[:2]))<=.24
        return near and clearance>=.01,signed

    def _upcoming(self):
        e=self.env.env;position=np.asarray(e.panel_pose()[0]);choices=[]
        records=self.env.last_scene['motion_anchor_placement']
        for rank,slot in enumerate(records['route_slot_order']):
            clear,signed=self._encounter(slot)
            if clear and signed>=0:
                self.vertical_passed.add(slot)
            if slot in self.vertical_passed:
                continue
            center,_=e.obstacles[slot]
            distance=float(np.linalg.norm(position[:2]-center[:2]))
            if signed<self.passage.get('progress_margin_m',.06) and distance<=self.passage.get('trigger_distance_m',.42):
                choices.append((rank,distance,slot))
        return min(choices)[2] if choices else None

    def _ik(self,position,quaternion,rest):
        e=self.env.env
        inverse=p.invertTransform(e.config['panel']['offset'],e.config['panel']['orientation'])
        target=p.multiplyTransforms(position,quaternion,*inverse)
        # Query-only reset is scoped and restores positions and velocities.
        with e.collision.at_configuration(rest):
            q=np.asarray(p.calculateInverseKinematics(e.robot,e.ee,*target,
                lowerLimits=e.lower.tolist(),upperLimits=e.upper.tolist(),
                jointRanges=(e.upper-e.lower).tolist(),restPoses=rest.tolist(),
                maxNumIterations=160,residualThreshold=1e-5,physicsClientId=e.client))[:7]
            if not e.collision.within_limits(q):return None
            e.set_configuration(q)
            actual=e.kinematic_panel_pose()[0]
            if np.linalg.norm(np.asarray(actual)-position)>.025:return None
        return q

    def _vertical_candidate(self,slot,lateral,gain):
        e=self.env.env;position,quaternion=e.panel_pose();position=np.asarray(position)
        q,_=e.joint_state();record=self.env.last_scene['motion_anchor_placement']['obstacles'][slot]
        tangent=np.asarray(record['tangent_xy']);normal=np.array([-tangent[1],tangent[0]])
        center=route_state(e.obstacle_motion[slot],e.sim_time+.7)[0]
        dimensions=np.asarray(e.obstacles[slot][1]);sign=1 if record['intended_side']=='above' else -1
        half=panel_half_height(quaternion,e.config['panel']['dimensions'])
        target=center.copy();target[:2]+=tangent*.08+normal*lateral
        target[2]+=sign*(dimensions[2]/2+half+self.passage.get('clearance_m',.025))
        site=e.config['construction_site']
        if target[2]-half<site['floor_z']+.015 or target[2]+half>site['ceiling_z']-.015:
            return None
        staging=position.copy();staging[2]=target[2]
        stage_q=self._ik(staging,quaternion,q)
        target_q=self._ik(target,quaternion,stage_q if stage_q is not None else q)
        if target_q is None:return None
        actions=[];passed=False;initial_signed=self._encounter(slot)[1]
        progress=0.;success=False;cost=task_cost(e.info())
        for tick in range(self.passage.get('horizon',40)):
            current,_=e.joint_state()
            # First gain height when necessary, then pass the box. No state reset
            # is used for these actuator steps, including prediction rollouts.
            reaching_height=stage_q is not None and tick<10 and abs(e.panel_pose()[0][2]-target[2])>.035
            goal=stage_q if reaching_height else target_q
            action=(np.clip((goal-current)/e.max_increment,-1,1)*gain).astype(np.float32)
            actions.append(action)
            _,_,done,truncated,info=self.env.step(action);self.prediction_steps+=1
            if info['collision'] or info['joint_limit_violation']:return None
            clear,signed=self._encounter(slot)
            progress=max(progress,signed-initial_signed)
            passed|=clear and signed>=0 and progress>=self.passage.get('minimum_progress_m',.025)
            cost=task_cost(info)
            success=info['termination_reason']=='success'
            if done or truncated or (passed and np.linalg.norm(current-target_q)<.06):break
        if not passed:return None
        return success,cost,actions,progress

    def _vertical_plan(self,slot):
        best=None;rank=None;initial_cost=task_cost(self.env.env.info())
        with EpisodeSnapshot(self.env) as snapshot:
            for lateral in self.passage.get('lateral_offsets_m',[0.,-.05,.05]):
                for gain in self.passage.get('gains',[.7,1.]):
                    snapshot.restore();self.env.env.render_mode=None
                    result=self._vertical_candidate(slot,lateral,gain)
                    if result is None:continue
                    success,cost,actions,progress=result
                    if not success and cost>initial_cost+.25:continue
                    # Reserve actuator time to finish installation after a
                    # detour. This changes neither the episode budget nor goal.
                    q=self.env.env.joint_state()[0]
                    minimum_finish=int(np.ceil(np.max(np.abs(self.env.cursor.path[-1]-q))/self.env.env.max_increment))+8
                    remaining=self.env.env.config['simulation']['max_steps']-self.env.env.steps
                    if not success and minimum_finish>remaining:continue
                    value=(int(success),-cost,-len(actions))
                    if rank is None or value>rank:
                        rank=value;best=(actions,dict(slot=slot,lateral_m=lateral,gain=gain,
                            predicted_end_cost=cost,predicted_forward_m=progress))
        return best

    def _finish_plan(self,observation):
        """Commit early only when a full physical rollout reaches the real goal."""
        original_horizon=self.recovery.get('horizon',60)
        self.recovery['horizon']=min(80,self.env.env.config['simulation']['max_steps']-self.env.env.steps)
        try:
            with EpisodeSnapshot(self.env) as snapshot:
                for kind,value in [('goal',1.),('goal',.7),('goal',.4),
                                   ('lookahead',(3,.7)),('lookahead',(6,1.))]:
                    snapshot.restore();self.env.env.render_mode=None
                    safe,success,cost,actions=self._candidate(kind,value,observation.copy())
                    if safe and success:return actions,kind,value,cost
        finally:self.recovery['horizon']=original_horizon
        return None

    def action(self,observation):
        e=self.env.env
        if not self.passage.get('enabled',True):return super().action(observation)
        self._observe()
        if self.vertical_plan and self.plan:
            if self._planned_safe(observation):
                action=self._plan_action();self.last_real_action=action.copy()
                return action
            self.plan=[]
        if self.vertical_plan and not self.plan:self.vertical_plan=False
        remaining=e.config['simulation']['max_steps']-e.steps
        if (not self.plan and remaining<=self.passage.get('finish_budget_steps',80) and
            e.steps-self.finish_attempt_step>=self.passage.get('finish_retry_steps',12)):
            self.finish_attempt_step=e.steps
            finish=self._finish_plan(observation)
            if finish is not None:
                actions,kind,value,cost=finish
                self.plan=actions;self.plan_mode='deadline_finish';self.plan_executed=0
                self.vertical_plan=True
                self.recovery_events.append(dict(step=e.steps,mode='deadline_finish',
                    accepted=True,parameter=[kind,value],predicted_end_cost=cost,plan_steps=len(actions)))
                action=self._plan_action();self.last_real_action=action.copy()
                return action
        if not self.plan and e.steps-self.vertical_attempt_step>=self.passage.get('cooldown_steps',8):
            slot=self._upcoming()
            if slot is not None:
                self.vertical_attempt_step=e.steps
                found=self._vertical_plan(slot)
                if found is not None:
                    actions,event=found
                    self.plan=actions;self.plan_mode='vertical_passage';self.plan_executed=0
                    self.vertical_plan=True
                    event.update(step=e.steps,plan_steps=len(actions))
                    self.vertical_events.append(event)
                    action=self._plan_action();self.last_real_action=action.copy()
                    return action
        return super().action(observation)
