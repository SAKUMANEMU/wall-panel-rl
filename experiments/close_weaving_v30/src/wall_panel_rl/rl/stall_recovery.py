"""Training-only stagnation feedback; observation/action/termination stay unchanged."""
from collections import deque
import numpy as np
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv


class StallDetector:
    """Require little task/path progress AND little net motion over a full window.

    A real detour can increase target error. Net movement or reference progress
    therefore exempts it; a changed reference alone must not create a penalty.
    """
    def __init__(self,window=20,position_epsilon=.001,orientation_epsilon_deg=.1,
                 path_epsilon=.01,joint_displacement=.02,weight=.10):
        if window<2 or min(position_epsilon,orientation_epsilon_deg,path_epsilon,joint_displacement,weight)<=0:
            raise ValueError('Invalid stagnation settings')
        self.window=window;self.pos=position_epsilon;self.rot=np.deg2rad(orientation_epsilon_deg)
        self.path=path_epsilon;self.motion=joint_displacement;self.weight=weight
        self.history=deque(maxlen=window+1)

    def observe(self,q,position,orientation,remaining,*,terminal=False):
        self.history.append((np.asarray(q,dtype=float).copy(),float(position),float(orientation),float(remaining)))
        if terminal or len(self.history)<self.window+1:return 0.
        first,last=self.history[0],self.history[-1]
        no_progress=(first[1]-last[1]<self.pos and first[2]-last[2]<self.rot and first[3]-last[3]<self.path)
        no_motion=np.linalg.norm(last[0]-first[0])<self.motion
        return -self.weight if no_progress and no_motion else 0.


class StallAwareEnv(StagedAvoidanceEnv):
    """Static pilot reward only; never override actions, waypoints or done flags."""
    def _remaining(self):
        q,_=self.env.joint_state();path=self.cursor.path;i=self.cursor.index
        return float(np.linalg.norm(self.ref-q)+np.linalg.norm(np.diff(path[i:],axis=0),axis=1).sum())

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        self.stall_detector=StallDetector(**self.settings['stall_reward'])
        self.stall_detector.observe(self.env.joint_state()[0],info['position_error'],info['orientation_error'],self._remaining())
        return obs,info

    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        penalty=self.stall_detector.observe(self.env.joint_state()[0],info['position_error'],
            info['orientation_error'],self._remaining(),terminal=done or truncated)
        info.update(reward_stall=penalty,stall_detected=penalty<0,reward_total=float(reward+penalty))
        return obs,float(reward+penalty),done,truncated,info
