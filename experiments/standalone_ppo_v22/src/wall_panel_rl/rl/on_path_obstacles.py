"""Static boxes intersecting a nominal, obstacle-free physical policy rollout.

Placement happens before animation. It does not query or filter the policy's
performance WITH obstacles, and it never changes the policy observation route.
"""
import copy
import numpy as np
import pybullet as p
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv


def nominal_rollout(model, config, settings, scene):
    empty=copy.deepcopy(scene);empty['obstacles']=[];empty['obstacle_motion']=[]
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings)
    joints=[];positions=[]
    try:
        obs,info=env.reset(options={'scenario':empty,'task':'alignment',
            'strength':[settings['active_stage']['amplitude'],settings['active_stage']['speed']]})
        while True:
            joints.append(env.env.joint_state()[0].tolist());positions.append(list(env.env.panel_pose()[0]))
            obs,_,done,truncated,info=env.step(model.predict(obs,deterministic=True)[0])
            if done or truncated:
                joints.append(env.env.joint_state()[0].tolist());positions.append(list(env.env.panel_pose()[0]));break
        if info['termination_reason']!='success':
            raise RuntimeError('Cannot build a complete nominal route: '+info['termination_reason'])
        return {'joint_positions':joints,'panel_positions':positions,'steps':env.env.steps,
                'termination_reason':info['termination_reason'],'route_kind':'initialization_obstacle_free_physical_rollout'}
    finally:env.close()


def place_on_nominal(env, scene, nominal, spec):
    """Keep dimensions; move each center onto one separated part of the route."""
    result=copy.deepcopy(scene);boxes=copy.deepcopy(scene['obstacles'])
    fractions=spec['fractions']
    if len(boxes)!=4 or len(fractions)!=4:raise ValueError('Expected four route boxes')
    empty=copy.deepcopy(scene);empty['obstacles']=[];empty['obstacle_motion']=[]
    env.reset(seed=int(scene['seed']),options={'scenario':empty})
    joints=np.asarray(nominal['joint_positions']);points=np.asarray(nominal['panel_positions'])
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1]<.5:raise RuntimeError('Nominal route is too short for four sparse boxes')
    progress=arc/arc[-1];rng=np.random.default_rng(int(scene['seed'])+7_000_000)
    selected=[];placement=[];gap=env.config['obstacles']['min_obstacle_gap']
    for slot,(box,fraction) in enumerate(zip(boxes,fractions)):
        dimensions=np.asarray(box['dimensions']);accepted=False
        for attempt in range(spec['attempts_per_obstacle']):
            desired=float(np.clip(fraction+rng.uniform(-spec['fraction_jitter'],spec['fraction_jitter']),.08,.92))
            index=int(np.argmin(np.abs(progress-desired)))
            center=points[index]+rng.uniform(-spec['center_jitter_m'],spec['center_jitter_m'],3)
            separated=all(np.linalg.norm(np.maximum(np.abs(center-b['center'])-
                (dimensions+np.asarray(b['dimensions']))/2,0))>=gap for b in selected)
            if not separated:continue
            body=env.box(dimensions,center,[.75,.16,.2,1]);env.obstacle_ids.append(body)
            endpoints=(env.collision.is_configuration_valid(scene['initial_q']) and
                       env.collision.is_configuration_valid(scene['reference_joint_path'][-1]))
            panel=[];robot=[]
            if endpoints and not p.getClosestPoints(body,env.wall,0.,physicsClientId=env.client):
                with env.collision.at_configuration(joints[index]):
                    panel=p.getClosestPoints(env.panel,body,0.,physicsClientId=env.client)
                    robot=p.getClosestPoints(env.robot,body,0.,physicsClientId=env.client)
            if not (panel or robot):
                env.obstacle_ids.pop();p.removeBody(body,physicsClientId=env.client);continue
            selected.append({'center':center.tolist(),'dimensions':dimensions.tolist()})
            env.obstacles.append((center,dimensions))
            placement.append({'slot':slot,'requested_fraction':fraction,'actual_fraction':float(progress[index]),
                'nominal_index':index,'nominal_joint_position':joints[index].tolist(),
                'nominal_panel_position':points[index].tolist(),'intersects_panel':bool(panel),'intersects_arm':bool(robot),
                'minimum_contact_distance_m':min(float(c[8]) for c in list(panel)+list(robot)),
                'attempts':attempt+1})
            accepted=True;break
        if not accepted:raise RuntimeError(f'Could not place on-route obstacle {slot} without endpoint overlap')
    result['obstacles']=selected;result['obstacle_motion']=[]
    result['nominal_motion']=nominal
    result['on_path_placement']={'kind':nominal['route_kind'],'obstacles':placement,
        'all_obstacles_intersect_nominal':all(r['intersects_panel'] or r['intersects_arm'] for r in placement),
        'start_goal_valid':True,'minimum_gap_m':gap}
    return result
