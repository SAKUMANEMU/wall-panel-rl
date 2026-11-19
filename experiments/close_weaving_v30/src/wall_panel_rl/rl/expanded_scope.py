"""Versioned, geometry-only expansion using two permanently frozen route guides."""
import copy
import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.vertical_corridor import fresh_vertical_corridor_case
from wall_panel_rl.rl.slow_dynamic_four import validate_placement


def workspace_offset(scene,rng,scope):
    dims=np.array([b['dimensions'] for b in scene['obstacles']])
    anchors=np.array([m['anchor'] for m in scene['obstacle_motion']])
    extent=np.array([m['extent'] for m in scene['obstacle_motion']])
    lower=float(np.min(anchors[:,2]-extent[:,2]-dims[:,2]/2))
    upper=float(np.max(anchors[:,2]+extent[:,2]+dims[:,2]/2))
    lo=max(scope['global_z_shift_m'][0],scope['swept_z_bounds_m'][0]-lower)
    hi=min(scope['global_z_shift_m'][1],scope['swept_z_bounds_m'][1]-upper)
    if lo>hi:raise RuntimeError('No allowed vertical translation fits the declared swept workspace')
    return np.array([rng.uniform(-scope['global_xy_shift_m'],scope['global_xy_shift_m']),
                     rng.uniform(-scope['global_xy_shift_m'],scope['global_xy_shift_m']),rng.uniform(lo,hi)])


def corridor_witnesses(config,case):
    """All four swept boxes must interact with frozen guide/reference corridors.

    The enlarged box is a geometry witness only; real execution still uses
    the original dimensions and moving positions, never swept collision bodies.
    """
    scene=case['scene']
    paths=[(name,np.array(scene[name]['joint_positions'])) for name in ('nominal_motion','original_nominal_motion')]
    env=WallPanelEnv(copy.deepcopy(config));found=[]
    try:
        env.reset(seed=scene['seed'],options={'scenario':scene})
        for slot,(box,motion) in enumerate(zip(scene['obstacles'],scene['obstacle_motion'])):
            proxy=env.box(np.array(box['dimensions'])+2*np.array(motion['extent']),motion['anchor'],[.5,.5,.5,0])
            witness=None
            try:
                for route_name,path in paths:
                    for index in range(0,len(path),2):
                        with env.collision.at_configuration(path[index]):
                            panel=bool(p.getClosestPoints(env.panel,proxy,0,physicsClientId=env.client))
                            arm=bool(p.getClosestPoints(env.robot,proxy,0,physicsClientId=env.client))
                        if panel or arm:
                            witness={'slot':slot,'route_name':route_name,'nominal_index':index,'panel':panel,'arm':arm};break
                    if witness is not None:break
            finally:p.removeBody(proxy,physicsClientId=env.client)
            if witness is None:raise RuntimeError(f'Expanded obstacle O{slot+1} misses the declared motion corridor')
            found.append(witness)
        return found
    finally:env.close()


def expanded_case(config,settings,profile,scope,seed,guide_model,controller,guide_tag):
    generation_config=copy.deepcopy(config)
    generation_config['robot']['start_randomization']=scope['start_randomization_rad']
    base=fresh_vertical_corridor_case(generation_config,settings,profile,seed,guide_model,controller)
    rng=np.random.default_rng(seed+23000000);last=None
    for _ in range(scope['transform_attempts']):
        case=copy.deepcopy(base);scene=case['scene']
        try:translation=workspace_offset(scene,rng,scope)
        except RuntimeError:raise
        for box,motion in zip(scene['obstacles'],scene['obstacle_motion']):
            shift=translation.copy();shift[:2]+=rng.uniform(-scope['local_xy_jitter_m'],scope['local_xy_jitter_m'],2)
            for key in ('anchor','reference'):motion[key]=(np.array(motion[key])+shift).tolist()
            box['center']=(np.array(motion['anchor'])+np.array(motion['extent'])*np.sin(motion['phase'])).tolist()
        try:
            validate_placement(generation_config,case)
            witnesses=corridor_witnesses(generation_config,case)
        except (RuntimeError,ValueError) as exc:last=str(exc);continue
        scene['expanded_scope_v23']={'guide':guide_tag,'global_translation_m':translation.tolist(),
            'swept_z_bounds_m':scope['swept_z_bounds_m'],'all_four_swept_boxes_intersect_frozen_corridors':True,
            'corridor_witnesses':witnesses,'geometry_only_acceptance':True,
            'historical_anchor_witnesses_precede_this_transform':True}
        return case
    raise RuntimeError('Expanded geometry transform budget exhausted: '+str(last))
