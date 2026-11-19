"""Explicit PPO + short-horizon physical prediction; not a pure PPO policy.

Candidate speed changes are simulated using the same controller and moving
obstacles. Physics AND Python episode state are restored after every candidate.
No global RRT or expert fallback is used. Real execution uses the original step.
"""
import copy,time
import numpy as np
import pybullet as p


class EpisodeSnapshot:
    OUTER_SERVICES={'env','action_space','observation_space'}
    INNER_SERVICES={'collision','controller','action_space','observation_space'}
    def __init__(self,env):
        self.env=env;self.inner=env.env
        self.state=p.saveState(physicsClientId=self.inner.client)
        self.outer=copy.deepcopy({k:v for k,v in env.__dict__.items() if k not in self.OUTER_SERVICES})
        self.physical=copy.deepcopy({k:v for k,v in self.inner.__dict__.items() if k not in self.INNER_SERVICES})
        self.closed=False
    @staticmethod
    def restore_dict(obj,saved,services):
        for key in list(obj.__dict__):
            if key not in services and key not in saved:del obj.__dict__[key]
        obj.__dict__.update(copy.deepcopy(saved))
    def restore(self):
        if self.closed:raise RuntimeError('Snapshot already released')
        p.restoreState(self.state,physicsClientId=self.inner.client)
        self.restore_dict(self.inner,self.physical,self.INNER_SERVICES)
        self.restore_dict(self.env,self.outer,self.OUTER_SERVICES)
    def close(self):
        if not self.closed:
            self.restore();p.removeState(self.state,physicsClientId=self.inner.client);self.closed=True
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class PredictiveSafetyController:
    """Prefer the original PPO speed; slow/wait/retreat if a rollout collides.

    Safety is finite-horizon and not guaranteed. If all candidates fail, choose
    the candidate surviving longest; actual collisions remain terminal failures.
    """
    def __init__(self,model,env,horizon=8,gains=(1.,.75,.4,0.,-.25,-.75)):
        if int(horizon)<1 or not gains or gains[0]!=1 or not np.isfinite(gains).all():raise ValueError('Invalid predictive safety settings')
        self.model=model;self.env=env;self.horizon=int(horizon);self.gains=tuple(gains)
        self.interventions=0;self.no_safe_action=0;self.prediction_steps=0;self.calls=0;self.seconds=0.;self.gain_counts={}
    def action(self,observation):
        begin=time.perf_counter();self.calls+=1
        original=np.asarray(self.model.predict(observation,deterministic=True)[0],dtype=np.float32)
        e=self.env.env;human=e.render_mode=='human'
        if human:p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=e.client)
        candidates=[];chosen=None
        try:
            with EpisodeSnapshot(self.env) as snapshot:
                for gain in self.gains:
                    snapshot.restore();obs=observation.copy();steps=0;safe=True;success=False;first=(original*gain).clip(-1,1).astype(np.float32)
                    for tick in range(self.horizon):
                        action=first if tick==0 else (np.asarray(self.model.predict(obs,deterministic=True)[0])*gain).clip(-1,1).astype(np.float32)
                        obs,_,done,truncated,info=self.env.step(action);steps+=1;self.prediction_steps+=1
                        if info['collision'] or info['joint_limit_violation']:safe=False;break
                        if done or truncated:success=info['termination_reason']=='success';break
                    candidates.append((steps,-abs(gain-1),gain,first))
                    if safe:
                        chosen=(gain,first);break
                if chosen is None:
                    self.no_safe_action+=1
                    _,_,gain,first=max(candidates,key=lambda x:x[:2]);chosen=(gain,first)
        finally:
            if human:p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1,physicsClientId=e.client)
        gain,action=chosen;self.interventions+=gain!=1.;self.gain_counts[str(gain)]=self.gain_counts.get(str(gain),0)+1;self.seconds+=time.perf_counter()-begin
        return action
