"""167 observations: legacy 77 plus five known-route parameter slots of 18."""
import numpy as np
from gymnasium import spaces
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.envs.known_route_motion import OBSERVATIONS, validate_motion, route_state, observation_features


class KnownRouteEnv(ConditionalRecoveryEnv):
    supports_dynamic_obstacles = True

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observation_space = spaces.Box(-np.inf,np.inf,(OBSERVATIONS,),np.float32)

    def _observe_routes(self,legacy):
        if legacy.shape != (77,):
            raise AssertionError('Legacy robot/obstacle/reference features changed')
        return np.r_[legacy,observation_features(self.env)].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        if not options or options.get('task') != 'dynamic':
            raise ValueError('Explicit known-route dynamic scene required')
        scene = options['scenario']
        count = len(scene['obstacles'])
        if count not in (2,3,4,5) or len(scene['obstacle_motion']) != count:
            raise ValueError('Expected 2--5 matching dynamic obstacle routes')
        for box,motion in zip(scene['obstacles'],scene['obstacle_motion']):
            validate_motion(motion,speed_range=tuple(self.env.config['obstacles']['dynamic'].get('speed_range_m_s',(.04,.22))))
            if not np.allclose(box['center'],route_state(motion,0.)[0],atol=1e-12,rtol=0):
                raise ValueError('Initial obstacle position disagrees with its route')
        observation,info = super().reset(seed=seed,options=options)
        return self._observe_routes(observation),dict(info,obstacle_count=count,
            known_geometric_routes=True, all_obstacles_dynamic=True,
            speed_sampling='constant physical speed per obstacle per episode')

    def step(self,action):
        observation,reward,done,truncated,info = super().step(action)
        return self._observe_routes(observation),reward,done,truncated,info
