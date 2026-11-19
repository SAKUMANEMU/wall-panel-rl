"""Geometry-only 2--5 box generation against an obstacle-free physical guide.

Every real anchor box must intersect an arm/panel guide configuration. Swept
proxies protect endpoints, wall and spacing only; they are never execution
obstacles. Candidate policy outcomes with obstacles are not a generation filter.
"""
import copy
import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.metrics import capture_scenario
from wall_panel_rl.rl.on_path_obstacles import nominal_rollout
from wall_panel_rl.rl.slow_dynamic_four import validate_placement
from wall_panel_rl.rl.on_path_weaving import check_workspace


def generate_variable_case(config, settings, spec, seed, count, model):
    if count not in (2, 3, 4, 5):
        raise ValueError('Expected 2--5 obstacles')
    spec=copy.deepcopy(spec)
    site=config.get('construction_site',{})
    if site.get('enabled'):
        spec['swept_z_bounds_m']=[site['floor_z']+.015,site['ceiling_z']-.015]
    empty_config = copy.deepcopy(config)
    empty_config['obstacles']['stage'] = 'A'
    env = WallPanelEnv(empty_config)
    rng = np.random.default_rng(seed + 26000000)
    try:
        env.reset(seed=seed)
        scene = capture_scenario(env, seed)
        guide = nominal_rollout(model, config, settings, scene)
        scene['nominal_motion'] = guide
        points = np.asarray(guide['panel_positions'])
        joints = np.asarray(guide['joint_positions'])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        if arc[-1] < .8:
            raise RuntimeError('Physical guide too short for a sparse corridor')
        progress = arc / arc[-1]
        # Keep the established four-box input order when count == 4.
        order = {2: [0, 1], 3: [2, 0, 1], 4: [2, 0, 1, 3], 5: [2, 0, 3, 1, 4]}[count]
        fractions = np.linspace(.16,.84,count)
        dt = config['simulation']['timestep'] * config['simulation']['substeps']
        sizes = np.asarray(config['obstacles']['dimensions_by_obstacle'])[[2, 3, 1, 4, 0]]
        gap = config['obstacles']['min_obstacle_gap']
        for restart in range(spec['restarts']):
            env.reset(seed=seed, options={'scenario': scene})
            boxes, motions, records = {}, {}, {}
            flip = int(rng.choice([-1, 1]))
            failed = False
            ranks=([0,count-1]+list(range(1,count-1))) if site.get('enabled') else list(range(count))
            for rank in ranks:
                slot=order[rank]
                dimensions = sizes[slot] * rng.uniform(*spec['size_scale'])
                weights = rng.uniform(.65, 1., 3) * [1., 1., 1.25]
                extent = spec['amplitude_m'] * weights / np.linalg.norm(weights)
                omega = (spec['max_speed_m_s'] / spec['amplitude_m']) * rng.uniform(.65, 1., 3)
                accepted = False
                for attempt in range(spec['attempts_per_obstacle']):
                    f = fractions[rank] + rng.uniform(-spec['fraction_jitter'], spec['fraction_jitter'])
                    index = int(np.argmin(abs(progress - f)))
                    anchor = points[index].copy()
                    anchor[:2] += rng.uniform(-spec['xy_jitter_m'], spec['xy_jitter_m'], 2)
                    if site.get('enabled'):
                        direction=points[min(index+1,len(points)-1),:2]-points[max(0,index-1),:2]
                        normal=np.array([-direction[1],direction[0]])
                        normal/=max(np.linalg.norm(normal),1e-12)
                        # Use the panel's finite-width corridor instead of
                        # packing every enlarged sweep on its centerline.
                        anchor[:2]+=normal*rng.uniform(-spec.get('corridor_lateral_offset_m',.15),
                                                       spec.get('corridor_lateral_offset_m',.15))
                    offset = flip * (-1 if rank % 2 else 1) * rng.uniform(*spec['vertical_offsets_m'])
                    anchor[2] += offset
                    half = dimensions[2] / 2 + extent[2]
                    if anchor[2] - half < spec['swept_z_bounds_m'][0] or anchor[2] + half > spec['swept_z_bounds_m'][1]:
                        continue
                    proxy = env.box(dimensions + 2 * extent, anchor, [0, 0, 0, 0])
                    invalid = bool(p.getClosestPoints(proxy, env.wall, 0., physicsClientId=env.client))
                    invalid |= any(p.getClosestPoints(proxy, other, gap, physicsClientId=env.client) for other in env.obstacle_ids)
                    invalid |= any(p.getClosestPoints(proxy,fixed,0.,physicsClientId=env.client) for fixed in env.site_collision_ids)
                    if invalid:
                        p.removeBody(proxy, physicsClientId=env.client)
                        continue
                    env.obstacle_ids.append(proxy)
                    endpoints = (env.collision.is_configuration_valid(scene['initial_q']) and
                                 env.collision.is_configuration_valid(scene['reference_joint_path'][-1]))
                    panel, arm = [], []
                    if endpoints:
                        actual = env.box(dimensions, anchor, [0, 0, 0, 0])
                        try:
                            with env.collision.at_configuration(joints[index]):
                                panel = p.getClosestPoints(env.panel, actual, 0., physicsClientId=env.client)
                                arm = p.getClosestPoints(env.robot, actual, 0., physicsClientId=env.client)
                        finally:
                            p.removeBody(actual, physicsClientId=env.client)
                    if not (panel or arm):
                        env.obstacle_ids.pop()
                        p.removeBody(proxy, physicsClientId=env.client)
                        continue
                    phase = rng.uniform(0., 2 * np.pi, 3)
                    # One real timed conflict per scene, other phases stay random.
                    if rank == count // 2:
                        phase = np.mod(-omega * (index * dt), 2 * np.pi)
                    motions[slot] = dict(anchor=anchor.tolist(), reference=anchor.tolist(),
                                         extent=extent.tolist(), omega=omega.tolist(), phase=phase.tolist())
                    boxes[slot] = dict(dimensions=dimensions.tolist(), center=(anchor + extent * np.sin(phase)).tolist())
                    tangent = points[min(index+1, len(points)-1)] - points[max(0, index-1)]
                    tangent[2] = 0.
                    tangent /= max(np.linalg.norm(tangent), 1e-12)
                    records[slot] = dict(slot=slot, nominal_index=index, actual_fraction=float(progress[index]),
                        nominal_joint_position=joints[index].tolist(), nominal_panel_position=points[index].tolist(),
                        vertical_offset_m=float(offset), intended_side='below' if offset > 0 else 'above',
                        tangent_xy=tangent[:2].tolist(), intersects_panel=bool(panel), intersects_arm=bool(arm),
                        minimum_contact_distance_m=min(float(c[8]) for c in list(panel)+list(arm)),
                        timed_conflict=rank == count // 2, guide_time_s=index*dt)
                    accepted = True
                    break
                if not accepted:
                    failed = True
                    break
            if failed:
                continue
            case = {'suite': 'variable_dynamic', 'task': 'dynamic',
                    'strength': [spec['amplitude_m'], spec['max_speed_m_s']], 'scene': copy.deepcopy(scene)}
            final = case['scene']
            final['obstacles'] = [boxes[i] for i in range(count)]
            final['obstacle_motion'] = [motions[i] for i in range(count)]
            final['motion_anchor_placement'] = dict(kind='obstacle_free_physical_guide', route_slot_order=order,
                obstacles=[records[i] for i in range(count)], all_obstacles_intersect_nominal=True,
                geometry_only_acceptance=True, placement_restart=restart)
            final['variable_weaving_v26'] = dict(count=count, height_sign_flip=flip,
                known_motion=True, all_dynamic=True, workspace=check_workspace(case, spec['swept_z_bounds_m']))
            validate_placement(config, case)
            return case
        raise RuntimeError(f'Could not place {count} sparse, swept-safe, physical on-route boxes')
    finally:
        env.close()
