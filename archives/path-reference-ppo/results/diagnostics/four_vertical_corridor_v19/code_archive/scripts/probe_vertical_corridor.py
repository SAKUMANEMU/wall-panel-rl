"""Explicit demonstration search, not a performance test or training.

Use complete recorded preliminary motions as geometric placement guides. Final
PPO+prediction+recovery runs keep the original observation reference. All final
outcomes, including construction failures, are saved rather than hidden.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.slow_dynamic_four import dynamic_configuration, SlowDynamicFourEnv
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors
from wall_panel_rl.rl.demo_scene_adjustments import adjust_obstacle_heights
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
from wall_panel_rl.visualization.episode_replay import capture_frame, recording_payload

REPORT = ROOT/'results/diagnostics/four_vertical_corridor_v19'
WORK = {}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def initialize():
    torch.set_num_threads(1)
    profile = yaml.safe_load((ROOT/'configs/four_vertical_corridor.yaml').read_text())
    manifest = json.loads((ROOT/profile['source_report']/'candidate_manifest.json').read_text())
    config, settings = dynamic_configuration(manifest['configuration'], manifest['evaluation_settings'], profile)
    WORK.update(profile=profile, config=config, settings=settings,
                model=PPO.load(ROOT/profile['source_report']/'selected_model.zip', device='cpu'),
                control=json.loads((ROOT/'configs/four_progress_controller.json').read_text()))


def run_case(case):
    env = SlowDynamicFourEnv(copy.deepcopy(WORK['config']), [], copy.deepcopy(WORK['settings']))
    try:
        obs, _ = env.reset(options={'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']})
        original = env.cursor.path.copy()
        c = WORK['control']
        agent = ProgressRecoveryController(WORK['model'], env, c['selected']['horizon'], c['gains'], c['recovery'])
        frames = [capture_frame(env.env)]
        while True:
            before = (env.env.steps, env.env.sim_time)
            action = agent.action(obs)
            assert before == (env.env.steps, env.env.sim_time)
            obs, _, done, truncated, info = env.step(action)
            frames.append(capture_frame(env.env))
            if done or truncated:
                break
        assert np.array_equal(original, env.cursor.path)
        result = {'seed': case['scene']['seed'], 'reason': info['termination_reason'], 'steps': env.env.steps,
                  'position_m': float(info['position_error']), 'orientation_deg': float(np.rad2deg(info['orientation_error'])),
                  'controller': 'ppo_progress', 'recovery_actions': agent.recovery_actions,
                  'weights_changed': False, 'uses_global_rrt': False, 'real_step_limit': 240}
        return recording_payload(WORK['config'], case, result, frames)
    finally:
        env.close()


def passage_measurements(payload, radius=.24, margin=.01, minimum_height_change=.07):
    """Ordered near-box passages using the WHOLE panel vertical bounding box.

    Four and three-box alternating above/below sequences also require alternating
    actual height changes >=7cm. Checking control samples is a display diagnostic;
    physical collision outcomes still come from all actual physics substeps.
    """
    frames = payload['trajectory']
    scene = payload['case']['scene']
    original = scene.get('unadjusted_motion_anchor_placement', scene.get('motion_anchor_placement'))
    order = original['route_slot_order']
    points = np.array([f['panel'] for f in frames])
    orientations = np.array([f['panel_orientation'] for f in frames])
    panel_half_z = np.abs(Rotation.from_quat(orientations).as_matrix())[:, 2, :] @ (np.array(payload['configuration']['panel']['dimensions'])/2)
    possibilities, nearest = {}, []
    for slot in order:
        centers = np.array([f['obstacle_centers'][slot] for f in frames])
        xy = np.linalg.norm((points-centers)[:, :2], axis=1)
        half_z = scene['obstacles'][slot]['dimensions'][2]/2
        above = points[:, 2]-panel_half_z-centers[:, 2]-half_z
        below = centers[:, 2]-half_z-points[:, 2]-panel_half_z
        i = int(np.argmin(xy))
        nearest.append({'slot': slot, 'frame': i, 'xy_distance_m': float(xy[i]),
                        'panel_z': float(points[i,2]), 'box_z': float(centers[i,2]),
                        'panel_above_clearance_m': float(above[i]), 'panel_below_clearance_m': float(below[i])})
        for side, distances in [(1, above), (-1, below)]:
            possibilities[slot, side] = [dict(slot=slot, frame=int(i), time_s=frames[i]['time_s'],
                side='above' if side==1 else 'below', panel_z=float(points[i,2]), box_z=float(centers[i,2]),
                xy_distance_m=float(xy[i]), vertical_clearance_m=float(distances[i]))
                for i in np.where((xy<=radius)&(distances>=margin))[0]]
    # Prefer each box's closest XY encounter. This avoids presenting a remote
    # state as the passage when a tighter, ordered encounter also qualifies.
    closest_found = None
    for count in [4,3]:
        for start in range(5-count):
            selected = nearest[start:start+count]
            sides = [1 if r['panel_above_clearance_m']>=margin else
                     -1 if r['panel_below_clearance_m']>=margin else 0 for r in selected]
            dz = np.diff([r['panel_z'] for r in selected])
            if (all(r['xy_distance_m']<=radius for r in selected) and
                all(a['frame']+1<b['frame'] for a,b in zip(selected,selected[1:])) and
                all(a*b==-1 for a,b in zip(sides,sides[1:])) and
                np.all(np.abs(dz)>=minimum_height_change) and np.all(dz[:-1]*dz[1:]<0)):
                closest_found = [dict(slot=r['slot'], frame=r['frame'], time_s=frames[r['frame']]['time_s'],
                    side='above' if side==1 else 'below', panel_z=r['panel_z'], box_z=r['box_z'],
                    xy_distance_m=r['xy_distance_m'], vertical_clearance_m=r['panel_above_clearance_m']
                    if side==1 else r['panel_below_clearance_m']) for r,side in zip(selected,sides)]
                break
        if closest_found:
            break
    failed = set()
    def search(slots, side, direction, history):
        if not slots:
            return history
        key=(tuple(slots),side,direction,history[-1]['frame'] if history else -1)
        if key in failed:
            return None
        for choice in possibilities[slots[0], side]:
            if history and (choice['frame']<=history[-1]['frame']+1 or
                            direction*(choice['panel_z']-history[-1]['panel_z'])<minimum_height_change):
                continue
            result = search(slots[1:], -side, -direction if history else direction, history+[choice])
            if result:
                return result
        failed.add(key)
        return None
    found = closest_found
    for count in ([] if found else [4,3]):
        for start in range(5-count):
            for side in [1,-1]:
                for direction in [1,-1]:
                    found = search(order[start:start+count], side, direction, [])
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    z = points[:,2]
    return {'ordered_alternating_passages': found or [], 'passage_count': len(found or []),
            'xy_radius_m': radius, 'whole_panel_vertical_margin_m': margin,
            'minimum_actual_alternating_height_change_m': minimum_height_change,
            'nearest_xy_observations': nearest, 'upward_travel_m': float(np.maximum(np.diff(z),0).sum()),
            'downward_travel_m': float(-np.minimum(np.diff(z),0).sum()), 'height_range_m': float(np.ptp(z))}


def probe(original):
    seed = original['scene']['seed']
    try:
        preliminary = run_case(original)
        candidate = copy.deepcopy(original)
        frames = preliminary['trajectory']
        scene = candidate['scene']
        scene['original_nominal_motion'] = scene['nominal_motion']
        scene['nominal_motion'] = {'joint_positions': [f['joints'] for f in frames],
                                   'panel_positions': [f['panel'] for f in frames],
                                   'steps': preliminary['result']['steps'],
                                   'termination_reason': preliminary['result']['reason'],
                                   'route_kind': 'preliminary_physical_episode_no_success_filter'}
        scene.pop('demo_height_offsets_m', None)
        scene.pop('unadjusted_motion_anchor_placement', None)
        candidate = fit_motion_anchors(WORK['config'], WORK['settings'], WORK['profile'], candidate)
        candidate = adjust_obstacle_heights(WORK['config'], candidate, WORK['profile']['demo_obstacle_height_offsets_m'])
        final = run_case(candidate)
        measurements = passage_measurements(final)
        return {'seed': seed, 'preliminary_reason': preliminary['result']['reason'],
                'recording': final, 'measurements': measurements}
    except (RuntimeError, ValueError) as exc:
        return {'seed': seed, 'construction_rejection': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--count', type=int, default=12)
    args = parser.parse_args()
    if not args.run:
        return
    if REPORT.exists():
        raise FileExistsError('Refusing to overwrite development search')
    REPORT.mkdir(parents=True)
    visible = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/REPORT.name
    if not visible.exists():
        visible.symlink_to(REPORT, target_is_directory=True)
    originals = json.loads((ROOT/'data/four_vertical_validation_v18_seed161/validation_cases.json').read_text())[:args.count]
    outcomes = []
    def status(phase):
        value = {'phase': phase, 'completed': len(outcomes), 'total': len(originals),
                 'updated_at': datetime.now().astimezone().isoformat(),
                 'successful': sum(x.get('reason')=='success' for x in outcomes),
                 'verified_weaving_examples': sum(x.get('reason')=='success' and x.get('passage_count',0)>=3 for x in outcomes)}
        dump(REPORT/'status.json', value)
        print(json.dumps(value), flush=True)
    status('running')
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        futures = {pool.submit(probe, c): c['scene']['seed'] for c in originals}
        for future in as_completed(futures):
            result = future.result()
            seed = result['seed']
            dump(REPORT/f'probe_{seed}.json', result)
            if 'recording' in result:
                outcomes.append({'seed': seed, **result['recording']['result'],
                                 **result['measurements']})
            else:
                outcomes.append(result)
            dump(REPORT/'outcomes.json', outcomes)
            status('running')
    status('completed')


if __name__ == '__main__':
    main()
