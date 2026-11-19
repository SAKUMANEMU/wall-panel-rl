"""Slow, known three-axis obstacle motion around the original on-path anchors.

Placement rejects invalid geometry only, never a PPO or RRT failure. Swept
boxes are used solely to protect endpoints/spacing during scene construction.
Execution uses the existing time-aware planner, not those swept boxes.
"""
import copy
from pathlib import Path
import numpy as np
import pybullet as p
import yaml
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.static_demo_scene import fresh_static_case


def load_profile(path):
    spec=yaml.safe_load(Path(path).read_text());motion=spec['motion']
    amplitude=float(motion['amplitude_m']);speed=float(motion['max_speed_m_s'])
    lo,hi=map(float,motion['frequency_fraction'])
    if not np.isfinite([amplitude,speed,lo,hi]).all() or min(amplitude,speed,lo)<=0 or not lo<=hi<=1:
        raise ValueError('Positive motion bounds and frequency fractions <= 1 required')
    if spec['expert_probability']!=.5 or int(spec['placement_attempts'])<1:
        raise ValueError('Expected 50/50 routing and positive placement attempts')
    return spec


def dynamic_configuration(config,settings,spec):
    """Copy v5 settings; preserve robot, rewards, horizon, reference and action API."""
    config=copy.deepcopy(config);settings=copy.deepcopy(settings);motion=spec['motion']
    if settings['active_stage']['count']!=4 or not config.get('observation',{}).get('obstacle_velocity'):
        raise ValueError('Expected the four-obstacle, velocity-aware v5 interface')
    config['noise'].update(position_std=0.,orientation_std_deg=0.)
    config['obstacles']['freeze_motion']=False
    config['obstacles']['dynamic'].update(motion_mode='random_3d',amplitude=motion['amplitude_m'],max_speed=motion['max_speed_m_s'])
    config['rrt']['dynamic_mode']='timed'
    settings['active_stage'].update(name='four_slow_dynamic',mode='dynamic',amplitude=motion['amplitude_m'],speed=motion['max_speed_m_s'])
    settings['mixture']={'alignment':0.,'static':0.,'dynamic':1.}
    return config,settings


def add_motion(static_case,spec):
    """Reproducible phases/frequencies; each vector norm has an analytic bound."""
    case=copy.deepcopy(static_case);scene=case['scene'];motion=spec['motion']
    if len(scene['obstacles'])!=4 or scene.get('obstacle_motion'):
        raise ValueError('Expected four static on-path boxes before adding motion')
    amplitude=float(motion['amplitude_m']);speed=float(motion['max_speed_m_s'])
    rng=np.random.default_rng(int(scene['seed'])+int(motion['seed_offset']))
    lo,hi=motion['frequency_fraction'];scene['obstacle_motion']=[]
    for obstacle in scene['obstacles']:
        anchor=np.asarray(obstacle['center'],dtype=float)
        weights=rng.uniform(.75,1.25,3);extent=amplitude*weights/np.linalg.norm(weights)
        omega=(speed/amplitude)*rng.uniform(lo,hi,3);phase=rng.uniform(0,2*np.pi,3)
        obstacle['center']=(anchor+extent*np.sin(phase)).tolist()
        scene['obstacle_motion'].append({'anchor':anchor.tolist(),'extent':extent.tolist(),
            'omega':omega.tolist(),'phase':phase.tolist(),'reference':anchor.tolist()})
    if 'on_path_placement' in scene:
        # The old collision witnesses describe anchors, not every moving pose.
        scene['motion_anchor_placement']=scene.pop('on_path_placement')
    scene['slow_dynamic_motion']={'kind':'random_phase_3d_sine','amplitude_bound_m':amplitude,
        'speed_bound_m_s':speed,'time_basis':'simulation_seconds','known_motion':True}
    case.update(task='dynamic',strength=[amplitude,speed])
    return case


def validate_placement(config,case):
    """Keep endpoints and inter-box spacing valid throughout the motion bounds.

    This finite scene filter is not an avoidance planner. The arm may still
    traverse the moving boxes' swept regions between the protected endpoints.
    """
    scene=case['scene'];env=WallPanelEnv(copy.deepcopy(config))
    try:
        env.reset(seed=int(scene['seed']),options={'scenario':scene})
        with env.planning_obstacles():
            for q in (scene['initial_q'],scene['reference_joint_path'][-1]):
                if not env.collision.is_configuration_valid(q):
                    raise RuntimeError('Obstacle motion bounds overlap the start or goal')
            gap=config['obstacles']['min_obstacle_gap']
            for i,body in enumerate(env.obstacle_ids):
                if p.getClosestPoints(body,env.wall,0.,physicsClientId=env.client):
                    raise RuntimeError('Obstacle motion bounds intersect the wall')
                if any(p.getClosestPoints(body,other,gap,physicsClientId=env.client) for other in env.obstacle_ids[:i]):
                    raise RuntimeError('Obstacle motion bounds violate sparse box spacing')
    finally:env.close()


def fresh_dynamic_case(config,settings,spec,seed=None,*,model):
    """Fresh robot/anchors by default, fixed scene and motion for an explicit seed."""
    requested=None if seed is None else int(seed);rng=np.random.default_rng();rejects=[]
    static_settings=copy.deepcopy(settings);static_settings['active_stage']['mode']='static'
    for _ in range(1 if requested is not None else int(spec['placement_attempts'])):
        current=requested if requested is not None else int(rng.integers(2_000_000,2_000_000_000))
        try:
            static,_=fresh_static_case(config,static_settings,current,model=model)
            case=add_motion(static,spec);validate_placement(config,case)
            return case,rejects
        except (RuntimeError,ValueError) as exc:
            rejects.append({'seed':current,'reason':str(exc)})
            if requested is not None:raise
    raise RuntimeError(f'Could not construct slow dynamic geometry: {rejects}')


def planning_snapshot(env):
    """Capture the physical clock and observations that a planning query must preserve."""
    return {'time':env.env.sim_time,'steps':env.env.steps,'joints':np.array(env.env.joint_state()),
            'centers':np.array([c for c,_ in env.env.obstacles]),'reference':env.cursor.path.copy()}


def assert_planning_paused(env,before):
    after=planning_snapshot(env)
    if before['time']!=after['time'] or before['steps']!=after['steps']:
        raise AssertionError('Planning advanced the simulation clock')
    for key in ('joints','centers','reference'):
        if not np.allclose(before[key],after[key],atol=1e-12,rtol=0):
            raise AssertionError('Planning modified '+key)


class SlowDynamicFourEnv(ConditionalRecoveryEnv):
    """Opt into dynamic scenes while retaining v5's local reference recovery."""
    supports_dynamic_obstacles=True

    def reset(self,*,seed=None,options=None):
        if not options or options.get('task')!='dynamic':raise ValueError('Explicit dynamic scenario required')
        scene=options['scenario'];motions=scene.get('obstacle_motion',[])
        if len(scene['obstacles'])!=4 or len(motions)!=4:raise ValueError('Expected four moving obstacles')
        amplitude,speed=options['strength']
        if not np.isfinite([amplitude,speed]).all() or min(amplitude,speed)<=0:raise ValueError('Invalid motion strength')
        for obstacle,motion in zip(scene['obstacles'],motions):
            extent=np.asarray(motion['extent']);omega=np.asarray(motion['omega']);phase=np.asarray(motion['phase'])
            anchor=np.asarray(motion['anchor'])
            if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (extent,omega,phase,anchor)):
                raise ValueError('Finite 3D motion vectors required')
            if np.any(extent<=0) or np.any(omega<=0) or np.linalg.norm(extent)>amplitude+1e-12 or np.linalg.norm(extent*omega)>speed+1e-12:
                raise ValueError('Obstacle motion exceeds the configured norm bounds')
            if not np.allclose(obstacle['center'],anchor+extent*np.sin(phase),atol=1e-12,rtol=0):
                raise ValueError('Replay obstacle position must match phase at t=0')
        obs,info=super().reset(seed=seed,options=options)
        if self.env.config['rrt']['dynamic_mode']!='timed':raise ValueError('Time-aware RRT is required')
        if obs.shape!=(77,) or self.action_space.shape!=(7,):raise ValueError('v5 interface changed')
        return obs,dict(info,planning_pauses_simulation_time=True,obstacle_motion='slow_dynamic')
