"""Sample on-path anchors against full motion bounds, not static boxes first.

This accelerates valid scene construction at large amplitudes. It never queries
the candidate controller's outcome with obstacles. Robot starts, references,
box sizes, motion norms and endpoint/wall/spacing guards are preserved.
"""
import copy
import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.static_demo_scene import fresh_static_case
from wall_panel_rl.rl.slow_dynamic_four import add_motion,validate_placement


def static_anchors(original):
    case=copy.deepcopy(original);scene=case['scene']
    if scene.get('obstacle_motion'):
        for box,motion in zip(scene['obstacles'],scene['obstacle_motion']):box['center']=copy.deepcopy(motion['anchor'])
        scene['obstacle_motion']=[]
    case['task']='static'
    return case


def fit_motion_anchors(config,settings,profile,original):
    static=static_anchors(original);case=add_motion(static,profile)
    try:
        validate_placement(config,case)
        return case
    except (RuntimeError,ValueError):pass
    scene=case['scene'];nominal=scene['nominal_motion'];points=np.asarray(nominal['panel_positions']);joints=np.asarray(nominal['joint_positions'])
    if len(points)!=len(joints) or len(points)<2:raise ValueError('Matching nominal joints and positions required')
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<.5:raise RuntimeError('Nominal route too short')
    progress=arc/arc[-1];spec=settings['on_path_obstacles'];fractions=spec['fractions']
    if len(scene['obstacles'])!=4 or len(fractions)!=4:raise ValueError('Four nominal anchor intervals required')
    empty=copy.deepcopy(scene);empty['obstacles']=[];empty['obstacle_motion']=[]
    env=WallPanelEnv(copy.deepcopy(config));rng=np.random.default_rng(int(scene['seed'])+17000000)
    attempts=int(profile.get('anchor_restarts',8));gap=config['obstacles']['min_obstacle_gap']
    # Resolve the endpoint-adjacent intervals first, then the interior slots.
    order=sorted(range(4),key=lambda i:abs(fractions[i]-.5),reverse=True)
    try:
        for restart in range(attempts):
            env.reset(seed=int(scene['seed']),options={'scenario':empty});selected={};records={}
            for slot in order:
                box=scene['obstacles'][slot];dims=np.asarray(box['dimensions']);extent=np.asarray(scene['obstacle_motion'][slot]['extent'])
                accepted=False
                for attempt in range(spec['attempts_per_obstacle']):
                    desired=float(np.clip(fractions[slot]+rng.uniform(-spec['fraction_jitter'],spec['fraction_jitter']),.08,.92))
                    index=int(np.argmin(np.abs(progress-desired)));center=points[index]+rng.uniform(-spec['center_jitter_m'],spec['center_jitter_m'],3)
                    proxy=env.box(dims+2*extent,center,[.4,.4,.4,0])
                    invalid=bool(p.getClosestPoints(proxy,env.wall,0.,physicsClientId=env.client))
                    if not invalid:invalid=any(p.getClosestPoints(proxy,other,gap,physicsClientId=env.client) for other in env.obstacle_ids)
                    if invalid:p.removeBody(proxy,physicsClientId=env.client);continue
                    env.obstacle_ids.append(proxy)
                    endpoints=(env.collision.is_configuration_valid(scene['initial_q']) and
                               env.collision.is_configuration_valid(scene['reference_joint_path'][-1]))
                    witness=[]
                    if endpoints:
                        actual=env.box(dims,center,[.8,.2,.2,0])
                        try:
                            with env.collision.at_configuration(joints[index]):
                                panel=p.getClosestPoints(env.panel,actual,0.,physicsClientId=env.client)
                                arm=p.getClosestPoints(env.robot,actual,0.,physicsClientId=env.client)
                                witness=list(panel)+list(arm)
                        finally:p.removeBody(actual,physicsClientId=env.client)
                    if not witness:
                        env.obstacle_ids.pop();p.removeBody(proxy,physicsClientId=env.client);continue
                    selected[slot]=center.copy()
                    records[slot]={'slot':slot,'requested_fraction':fractions[slot],'actual_fraction':float(progress[index]),
                        'nominal_index':index,'nominal_joint_position':joints[index].tolist(),'nominal_panel_position':points[index].tolist(),
                        'intersects_panel':bool(panel),'intersects_arm':bool(arm),
                        'minimum_contact_distance_m':min(float(c[8]) for c in witness),'attempts':attempt+1}
                    accepted=True;break
                if not accepted:break
            if len(selected)!=4:continue
            for slot in range(4):
                motion=scene['obstacle_motion'][slot];anchor=selected[slot]
                motion['anchor']=anchor.tolist();motion['reference']=anchor.tolist()
                scene['obstacles'][slot]['center']=(anchor+np.asarray(motion['extent'])*np.sin(motion['phase'])).tolist()
            scene['motion_anchor_placement']={'kind':nominal['route_kind'],'obstacles':[records[i] for i in range(4)],
                'all_obstacles_intersect_nominal':True,'start_goal_valid':True,'minimum_gap_m':gap,
                'reanchored_for_motion_bounds':True,'placement_restart':restart,
                'interval_jitter_unchanged':True,'method':'full-motion proxy geometry plus actual anchor collision witness'}
            validate_placement(config,case)
            return case
        raise RuntimeError('Could not fit four motion-safe anchors in the original path intervals')
    finally:env.close()


def fresh_motion_aware_case(config,settings,profile,seed,model):
    static_settings=copy.deepcopy(settings);static_settings['active_stage']['mode']='static'
    case,_=fresh_static_case(config,static_settings,int(seed),model=model)
    return fit_motion_anchors(config,settings,profile,case)
