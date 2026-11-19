"""Build a staggered dynamic scene against a previous measured arm trajectory.

The guide determines placement only. Its poses are never used as executed
actions. Collision witnesses refer to actual box geometry at the motion anchor;
timed intersections with the unchanged old execution are reported separately.
"""
import copy

import numpy as np
import pybullet as p

from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors
from wall_panel_rl.rl.slow_dynamic_four import validate_placement
from wall_panel_rl.visualization.episode_replay import RecordedEpisode


def check_workspace(case, bounds):
    """Keep every swept box, including its physical size, within the Z bounds."""
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError('Two finite ordered workspace bounds are required')
    lower, upper = [], []
    for box, motion in zip(case['scene']['obstacles'], case['scene']['obstacle_motion']):
        z = motion['anchor'][2]
        half = box['dimensions'][2] / 2 + motion['extent'][2]
        lower.append(z - half)
        upper.append(z + half)
    if not lower or min(lower) < bounds[0] or max(upper) > bounds[1]:
        raise RuntimeError('The staggered scene exceeds the swept vertical workspace')
    return {'minimum_swept_z_m': float(min(lower)), 'maximum_swept_z_m': float(max(upper)),
            'allowed_z_bounds_m': bounds.tolist()}


def build_weaving_case(configuration, settings, guide_payload, layout):
    """Reanchor all four boxes along the physical guide; retain policy reference."""
    guide = RecordedEpisode(guide_payload)
    mode = layout.get('placement_mode', 'resample')
    if mode == 'refine_existing':
        return refine_existing_case(configuration, guide_payload, layout)
    if mode != 'resample':
        raise ValueError('Expected refine_existing or resample placement mode')
    case = copy.deepcopy(guide.case)
    scene = case['scene']
    original_seed = int(scene['seed'])
    layout_seed = int(layout.get('layout_seed', original_seed))
    # Use a separate reproducible layout seed without resampling the robot start.
    # The placement helper uses scene.seed for anchor and motion sampling.
    scene['seed'] = layout_seed
    original_reference = copy.deepcopy(scene['reference_joint_path'])
    # These witnesses describe older layouts and must not certify the new one.
    for key in ('expanded_scope_v23', 'demo_height_offsets_m',
                'unadjusted_motion_anchor_placement', 'motion_anchor_placement',
                'final_height_stagger', 'preliminary_episode'):
        scene.pop(key, None)
    scene['nominal_motion'] = {
        'joint_positions': guide.joints.tolist(),
        'panel_positions': guide.positions.tolist(),
        'route_kind': 'previous_physical_execution',
    }
    placement_settings = copy.deepcopy(settings)
    spec = placement_settings['on_path_obstacles']
    spec.update(fractions=layout['route_fractions'], fraction_jitter=layout['fraction_jitter'],
                attempts_per_obstacle=layout['attempts_per_obstacle'])
    case = fit_motion_anchors(configuration, placement_settings, layout['motion_profile'], case)
    scene = case['scene']
    scene['seed'] = original_seed
    scene['on_path_weaving_v25'] = {
        'guide_scene_seed': int(guide.case['scene']['seed']),
        'layout_seed': layout_seed,
        'guide_result': guide.result['reason'],
        'layout_rule': 'actual-guide intersections with alternating vertical anchor offsets',
        'new_execution_not_used_for_geometry_acceptance': True,
        'workspace': check_workspace(case, layout['swept_z_bounds_m']),
    }
    if scene['reference_joint_path'] != original_reference:
        raise AssertionError('Placement changed the policy reference')
    validate_placement(configuration, case)
    return case


def refine_existing_case(configuration, guide_payload, layout):
    """Move near-route boxes just enough to obtain physical collision witnesses.

    Prefer the established high/low corridor over a complete random replacement.
    Guides include the measured execution and the stored physical nominal runs.
    These are placement guides, not prescribed commands for the next episode.
    """
    guide = RecordedEpisode(guide_payload)
    case = copy.deepcopy(guide.case)
    scene = case['scene']
    original_reference = copy.deepcopy(scene['reference_joint_path'])
    old_placement = scene.get('unadjusted_motion_anchor_placement', scene.get('motion_anchor_placement'))
    if not old_placement or len(scene['obstacles']) != 4:
        raise ValueError('A four-box corridor with a recorded route order is required')
    order = list(old_placement['route_slot_order'])
    paths = {'previous_actual': guide.joints.tolist()}
    paths.update({name: scene[name]['joint_positions'] for name in
                  ('nominal_motion', 'original_nominal_motion') if name in scene})
    overlap = float(layout['contact_overlap_m'])
    maximum = float(layout['maximum_anchor_translation_m'])
    if not np.isfinite([overlap, maximum]).all() or not 0 < overlap <= maximum:
        raise ValueError('Positive finite overlap and translation bounds are required')
    env = WallPanelEnv(copy.deepcopy(configuration))
    records = []
    try:
        env.reset(seed=scene['seed'], options={'scenario': scene})
        for slot, (body, motion, box) in enumerate(zip(env.obstacle_ids, scene['obstacle_motion'], scene['obstacles'])):
            p.resetBasePositionAndOrientation(body, motion['anchor'], [0, 0, 0, 1], physicsClientId=env.client)
            best = None
            for name, path in paths.items():
                for index, q in enumerate(path):
                    with env.collision.at_configuration(q):
                        contacts = (list(p.getClosestPoints(env.robot, body, .2, physicsClientId=env.client)) +
                                    list(p.getClosestPoints(env.panel, body, .2, physicsClientId=env.client)))
                    for contact in contacts:
                        if best is None or contact[8] < best[0]:
                            best = (float(contact[8]), name, index, q, contact[7])
            if best is None:
                raise RuntimeError(f'No physical guide close to obstacle O{slot+1}')
            shift = np.asarray(best[4]) * max(0., best[0] + overlap)
            if np.linalg.norm(shift) > maximum:
                raise RuntimeError(f'Obstacle O{slot+1} requires too large an anchor translation')
            for key in ('anchor', 'reference'):
                motion[key] = (np.asarray(motion[key]) + shift).tolist()
            box['center'] = (np.asarray(motion['anchor']) + np.asarray(motion['extent']) * np.sin(motion['phase'])).tolist()
            p.resetBasePositionAndOrientation(body, motion['anchor'], [0, 0, 0, 1], physicsClientId=env.client)
            with env.collision.at_configuration(best[3]):
                panel = p.getClosestPoints(env.panel, body, 0., physicsClientId=env.client)
                arm = p.getClosestPoints(env.robot, body, 0., physicsClientId=env.client)
            if not (panel or arm):
                raise RuntimeError(f'Anchor translation failed to put O{slot+1} on the physical guide')
            records.append({'slot': slot, 'guide_name': best[1], 'nominal_index': best[2],
                            'nominal_joint_position': best[3], 'translation_m': shift.tolist(),
                            'minimum_original_contact_distance_m': best[0],
                            'minimum_contact_distance_m': min(float(c[8]) for c in list(panel) + list(arm)),
                            'intersects_panel': bool(panel), 'intersects_arm': bool(arm)})
    finally:
        env.close()
    for key in ('expanded_scope_v23', 'unadjusted_motion_anchor_placement',
                'demo_height_offsets_m', 'final_height_stagger'):
        scene.pop(key, None)
    scene['motion_anchor_placement'] = {
        'kind': 'minimal_physical_guide_intersection', 'route_slot_order': order,
        'obstacles': records, 'all_obstacles_intersect_nominal': True,
    }
    # Optional timed conflicts make a selected box cross the old execution
    # exactly at its physical collision witness. Frequencies and bounds stay fixed.
    for slot in layout.get('align_guide_crossing_slots', []):
        if not isinstance(slot, int) or not 0 <= slot < 4:
            raise ValueError('Timed crossing slots must be integers from 0 to 3')
        record = records[slot]
        if record['guide_name'] != 'previous_actual':
            raise ValueError('Timed alignment requires an actual-execution witness')
        when = guide_payload['trajectory'][record['nominal_index']]['time_s']
        motion = scene['obstacle_motion'][slot]
        motion['phase'] = np.mod(-np.asarray(motion['omega']) * when, 2 * np.pi).tolist()
        scene['obstacles'][slot]['center'] = (np.asarray(motion['anchor']) +
            np.asarray(motion['extent']) * np.sin(motion['phase'])).tolist()
        record['aligned_anchor_crossing_time_s'] = when
    scene['on_path_weaving_v25'] = {
        'mode': 'minimal_anchor_refinement', 'guide_scene_seed': int(scene['seed']),
        'guide_result': guide.result['reason'], 'maximum_anchor_translation_m': maximum,
        'workspace': check_workspace(case, layout['swept_z_bounds_m']),
        'new_execution_not_used_for_geometry_acceptance': True,
    }
    if scene['reference_joint_path'] != original_reference:
        raise AssertionError('Placement changed the policy reference')
    validate_placement(configuration, case)
    return case


def audit_guide_intersections(configuration, case, guide_payload):
    """Query all links and panel against real boxes, without stepping physics.

    An anchor witness means the physical box blocks one guide configuration
    when it crosses its anchor. A timed witness additionally checks the moving
    box at the old guide's original simulation timestamp. Neither certifies
    success in the new scene; an actual new episode is still required.
    """
    scene = case['scene']
    records = scene['motion_anchor_placement']['obstacles']
    env = WallPanelEnv(copy.deepcopy(configuration))
    rows = []
    try:
        env.reset(seed=scene['seed'], options={'scenario': scene})
        for slot, (body, motion, record) in enumerate(zip(env.obstacle_ids, scene['obstacle_motion'], records)):
            p.resetBasePositionAndOrientation(body, motion['anchor'], [0, 0, 0, 1], physicsClientId=env.client)
            with env.collision.at_configuration(record['nominal_joint_position']):
                panel = p.getClosestPoints(env.panel, body, 0., physicsClientId=env.client)
                arm = p.getClosestPoints(env.robot, body, 0., physicsClientId=env.client)
            if not (panel or arm):
                raise AssertionError(f'Obstacle O{slot+1} misses the measured guide at its anchor')
            timed = None
            for index, frame in enumerate(guide_payload['trajectory']):
                center = (np.asarray(motion['anchor']) + np.asarray(motion['extent']) *
                          np.sin(np.asarray(motion['phase']) + np.asarray(motion['omega']) * frame['time_s']))
                p.resetBasePositionAndOrientation(body, center, [0, 0, 0, 1], physicsClientId=env.client)
                with env.collision.at_configuration(frame['joints']):
                    contacts = (list(p.getClosestPoints(env.panel, body, 0., physicsClientId=env.client)) +
                                list(p.getClosestPoints(env.robot, body, 0., physicsClientId=env.client)))
                if contacts:
                    timed = {'frame': index, 'time_s': frame['time_s'],
                             'minimum_contact_distance_m': min(float(c[8]) for c in contacts)}
                    break
            rows.append({'slot': slot, 'anchor_intersects_panel': bool(panel),
                         'anchor_intersects_arm': bool(arm), 'timed_guide_intersection': timed})
        return {'all_four_physical_anchors_intersect_guide': True,
                'timed_intersection_count': sum(r['timed_guide_intersection'] is not None for r in rows),
                'obstacles': rows, 'physics_steps_during_audit': 0}
    finally:
        env.close()
