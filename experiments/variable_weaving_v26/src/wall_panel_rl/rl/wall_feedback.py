"""Training-only wall approach reward; no actuator, target or collision changes."""
import numpy as np
import pybullet as p
from wall_panel_rl.rl.stall_recovery import StallAwareEnv


def wall_feedback(env,distance,weight):
    if distance<=0 or weight<0:raise ValueError('Invalid wall feedback')
    closest=distance
    for body in (env.robot,env.panel):
        for c in p.getClosestPoints(body,env.wall,distance,physicsClientId=env.client):
            closest=min(closest,float(c[8]))
    proximity=float(np.clip(1-closest/distance,0,1))
    return -weight*proximity**2,closest


class WallAwareEnv(StallAwareEnv):
    def step(self,action):
        obs,reward,done,truncated,info=super().step(action)
        spec=self.settings['wall_reward']
        penalty,distance=wall_feedback(self.env,spec['distance'],spec['weight'])
        # Successful installation remains a terminal success with no extra cost.
        if info['success']:penalty=0.
        info.update(reward_wall_clearance=penalty,wall_clearance_capped_m=distance,
                    reward_total=float(reward+penalty))
        return obs,float(reward+penalty),done,truncated,info
