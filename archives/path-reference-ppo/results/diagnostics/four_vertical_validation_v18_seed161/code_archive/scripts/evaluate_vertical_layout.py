"""Freeze the current layout and compare controllers on64 common validation cases.

No training, outcome-based scene filtering, policy fallback, or final test use.
Mixture ownership is sampled/frozen before outcomes, once per episode. Its
metrics reuse the matching full-controller episodes, not a stepwise blend.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pybullet as p
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.motion_aware_placement import fresh_motion_aware_case
from wall_panel_rl.rl.demo_scene_adjustments import adjust_obstacle_heights
from wall_panel_rl.rl.slow_dynamic_four import dynamic_configuration, SlowDynamicFourEnv, planning_snapshot, assert_planning_paused
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
from wall_panel_rl.rl.episode_imitation import make_rrt
from wall_panel_rl.rl.episode_policy import EpisodeChoice

WORK = {}
MODES = ('ppo', 'rrt', 'ppo_progress')


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def initialize(protocol, report):
    torch.set_num_threads(1)
    model = PPO.load(Path(report)/'frozen_model.zip', device='cpu')
    assert model.observation_space.shape == (77,) and model.action_space.shape == (7,)
    WORK.update(protocol=protocol, report=Path(report), model=model,
                weights=copy.deepcopy(model.policy.state_dict()))


def heartbeat(phase, **extra):
    dump(WORK['report']/'workers'/f'{os.getpid()}.json',
         {'phase': phase, 'updated_at': datetime.now().astimezone().isoformat(), **extra})


def construct(seed):
    protocol = WORK['protocol']
    heartbeat('geometry', seed=seed)
    try:
        case = fresh_motion_aware_case(protocol['configuration'], protocol['settings'],
                                       protocol['profile'], seed, WORK['model'])
        case = adjust_obstacle_heights(protocol['configuration'], case,
                                        protocol['profile']['demo_obstacle_height_offsets_m'])
        return {'case': case}
    except (RuntimeError, ValueError) as exc:
        return {'rejection': {'seed': seed, 'reason': str(exc)}}
    finally:
        heartbeat('idle')


def collision_objects(env):
    """Terminal contact geometry; never count expected adjacent-link overlaps."""
    contacts = []
    def append(a, b, label, allowed=lambda x: True):
        found = [c for c in p.getClosestPoints(a, b, 0., physicsClientId=env.client) if allowed(c)]
        if found:
            contacts.append({'pair': label, 'minimum_distance_m': min(float(c[8]) for c in found),
                             'links': [list(map(int, c[3:5])) for c in found]})
    for index, body in enumerate(env.obstacle_ids):
        append(env.robot, body, f'arm-O{index+1}')
        append(env.panel, body, f'panel-O{index+1}')
    append(env.robot, env.wall, 'arm-wall')
    append(env.panel, env.wall, 'panel-wall')
    append(env.robot, env.robot, 'arm-self', lambda c: abs(c[3]-c[4]) > 1)
    append(env.robot, env.panel, 'arm-panel', lambda c: c[3] != env.ee)
    return contacts


def timeout_category(row, position_tolerance, orientation_tolerance_deg):
    if row['reason'] != 'timeout':
        return None
    if (row['position_m'] <= 3*position_tolerance and
            row['orientation_deg'] <= 3*orientation_tolerance_deg):
        return 'near_goal_precision'
    if max(row['zero_command_steps'], row['stationary_steps']) >= .25*row['steps']:
        return 'waiting_or_stagnation'
    return 'slow_or_ineffective_progress'


def execute(job):
    mode, case = job
    protocol = WORK['protocol']
    env = SlowDynamicFourEnv(copy.deepcopy(protocol['configuration']), [], copy.deepcopy(protocol['settings']))
    started = time.perf_counter()
    seed = int(case['scene']['seed'])
    trace, zero, stationary, projection, longest, streak = [], 0, 0, 0, 0, 0
    control_seconds, initial_plan_seconds, reward = 0., 0., 0.
    expert, recovery = None, None
    heartbeat('episode', mode=mode, seed=seed, step=0)
    try:
        obs, info = env.reset(options={'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']})
        reference = env.cursor.path.copy()
        assert env.env.config['simulation']['max_steps'] == 240
        assert case['strength'] == [.18, .12] and len(env.env.obstacle_ids) == 4
        assert np.allclose(case['scene']['demo_height_offsets_m'], [0, -.05, 0, 0])
        snapshot = planning_snapshot(env)
        if mode == 'rrt':
            expert = make_rrt(env, seed+100000)
            initial_plan_seconds = expert.planning_time
        elif mode == 'ppo_progress':
            selection = protocol['controller']
            recovery = ProgressRecoveryController(WORK['model'], env, selection['selected']['horizon'],
                                                   selection['gains'], selection['recovery'])
        assert_planning_paused(env, snapshot)
        reason = 'planning_failure' if expert is not None and not expert.planning_success else None
        while reason is None:
            q_before = env.env.joint_state()[0].copy()
            snapshot = planning_snapshot(env)
            before = time.perf_counter()
            action = (expert.action(obs) if expert is not None else recovery.action(obs)
                      if recovery is not None else WORK['model'].predict(obs, deterministic=True)[0])
            elapsed = time.perf_counter()-before
            control_seconds += elapsed
            assert_planning_paused(env, snapshot)
            obs, r, done, truncated, info = env.step(action)
            reward += r
            zero_action = bool(np.max(np.abs(action)) < 1e-8)
            still = bool(np.max(np.abs(env.env.joint_state()[0]-q_before)) < 1e-4)
            zero += zero_action
            stationary += still
            streak = streak+1 if still else 0
            longest = max(longest, streak)
            projected = info.get('executed_action', action)
            projection += bool(np.max(np.abs(np.asarray(projected)-action)) > 1e-6)
            trace.append({'step': int(env.env.steps), 'time_s': env.env.sim_time,
                          'position_m': float(info['position_error']),
                          'orientation_deg': float(np.rad2deg(info['orientation_error'])),
                          'panel_height_m': float(env.env.panel_pose()[0][2]),
                          'zero_command': zero_action, 'stationary': still,
                          'reference_index': int(env.cursor.index), 'control_wall_s': elapsed})
            if env.env.steps % 20 == 0:
                heartbeat('episode', mode=mode, seed=seed, step=int(env.env.steps))
            if done or truncated:
                reason = info['termination_reason']
        assert np.array_equal(reference, env.cursor.path), 'Immutable observation reference changed'
        assert all(torch.equal(value, WORK['model'].policy.state_dict()[key]) for key, value in WORK['weights'].items())
        row = {'mode': mode, 'seed': seed, 'reason': reason, 'steps': int(env.env.steps),
               'position_m': float(info['position_error']), 'orientation_deg': float(np.rad2deg(info['orientation_error'])),
               'reward': float(reward), 'zero_command_steps': int(zero), 'stationary_steps': int(stationary),
               'longest_stationary_run': longest, 'projection_modified_steps': int(projection),
               'initial_planning_s': initial_plan_seconds, 'control_wall_s': control_seconds,
               'total_wall_s': time.perf_counter()-started, 'physics_elapsed_s': env.env.sim_time,
               'reference_index': int(env.cursor.index), 'reference_last': len(reference)-1,
               'reference_preserved': True, 'weights_changed': False, 'global_rrt': mode == 'rrt',
               'collision': reason == 'collision', 'joint_limit_violation': reason == 'joint_limit',
               'collision_objects': collision_objects(env.env) if reason == 'collision' else [],
               'replans': expert.replans if expert is not None else 0,
               'replan_failures': expert.replan_failures if expert is not None else 0,
               'scheduled_waits': expert.scheduled_waits if expert is not None else 0,
               'rrt_hold_steps': expert.hold_steps if expert is not None else 0,
               'predictive_interventions': recovery.interventions if recovery is not None else 0,
               'gain_counts': recovery.gain_counts if recovery is not None else {},
               'discarded_prediction_steps': recovery.prediction_steps if recovery is not None else 0,
               'local_rescues': recovery.rescues if recovery is not None else 0,
               'local_recovery_actions': recovery.recovery_actions if recovery is not None else 0,
               'recovery_events': recovery.recovery_events if recovery is not None else [],
               'conditional_recovery_attempts': int(info.get('conditional_recovery_attempts', 0)),
               'conditional_recoveries': int(info.get('conditional_recovery_count', 0)),
               'trace': trace}
        target = protocol['configuration']['target']
        row['position_exceeds_tolerance'] = row['position_m'] >= target['position_tolerance']
        row['orientation_exceeds_tolerance'] = row['orientation_deg'] >= target['orientation_tolerance_deg']
        row['timeout_category'] = timeout_category(row, target['position_tolerance'], target['orientation_tolerance_deg'])
        return row
    finally:
        env.close()
        heartbeat('idle')


def metrics(rows):
    n = len(rows)
    counts = {reason: sum(r['reason'] == reason for r in rows)
              for reason in ('success', 'collision', 'timeout', 'joint_limit', 'planning_failure')}
    assert sum(counts.values()) == n
    successful = [r for r in rows if r['reason'] == 'success']
    return {'episodes': n, **counts, 'success_rate': counts['success']/n if n else 0.,
            'collision_rate': counts['collision']/n if n else 0.,
            'timeout_rate': counts['timeout']/n if n else 0.,
            'mean_steps': float(np.mean([r['steps'] for r in rows])) if n else None,
            'mean_success_steps': float(np.mean([r['steps'] for r in successful])) if successful else None,
            'zero_command_steps': sum(r['zero_command_steps'] for r in rows),
            'stationary_steps': sum(r['stationary_steps'] for r in rows),
            'mean_initial_planning_s': float(np.mean([r['initial_planning_s'] for r in rows])) if n else None,
            'mean_control_wall_s': float(np.mean([r['control_wall_s'] for r in rows])) if n else None,
            'mean_total_wall_s': float(np.mean([r['total_wall_s'] for r in rows])) if n else None}


def mixture_rows(results, assignments):
    by_mode = {mode: {r['seed']: r for r in rows} for mode, rows in results.items()}
    rows = []
    for selection in assignments:
        row = copy.deepcopy(by_mode[selection['mode']][selection['seed']])
        row['mixture_branch'] = selection['mode']
        row['reused_matching_full_episode'] = True
        rows.append(row)
    return rows


def prior_seeds():
    used = set()
    def scan(value):
        if isinstance(value, dict):
            if isinstance(value.get('seed'), int):
                used.add(value['seed'])
            for item in value.values():
                scan(item)
        elif isinstance(value, list):
            for item in value:
                scan(item)
        elif isinstance(value, int):
            used.add(value)
    paths = list((ROOT/'data').rglob('*.json'))
    for name in ('four_vertical_weave_v15', 'four_vertical_weave_o2_lower_v16'):
        folder = ROOT/'results/diagnostics'/name
        paths += [p for p in folder.glob('*case*.json') if p.is_file()]
    for path in paths:
        scan(read(path))
    return used


def run(spec, spec_path):
    report, data = ROOT/spec['report_root'], ROOT/spec['data_root']
    if report.exists() or data.exists():
        raise FileExistsError('Refusing to overwrite frozen validation')
    profile = yaml.safe_load((ROOT/spec['motion_profile']).read_text())
    manifest = read(ROOT/spec['source_report']/'candidate_manifest.json')
    config, settings = dynamic_configuration(manifest['configuration'], manifest['evaluation_settings'], profile)
    controller = read(ROOT/spec['controller_config'])
    used = prior_seeds()
    report.mkdir(parents=True)
    data.mkdir(parents=True)
    visible = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve() != report.resolve():
        visible.symlink_to(report, target_is_directory=True)
    files = list((ROOT/'src').rglob('*.py'))+[spec_path, ROOT/'scripts/evaluate_vertical_layout.py',
            ROOT/spec['motion_profile'], ROOT/spec['controller_config'],
            ROOT/spec['source_report']/'candidate_manifest.json']
    hashes = {str(f.relative_to(ROOT)): sha(f) for f in files}
    original_model_sha = sha(ROOT/spec['source_model'])
    shutil.copy2(ROOT/spec['source_model'], report/'frozen_model.zip')
    assert sha(report/'frozen_model.zip') == original_model_sha
    for name in hashes:
        dest = report/'code_archive'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
    protocol = {'spec': spec, 'configuration': config, 'settings': settings, 'profile': profile,
                'controller': controller, 'source_model_sha256': original_model_sha,
                'code_hashes': hashes, 'optimizer_updates': 0, 'final_test_episodes': 0,
                'selection_rule': 'Fresh seeds; complete obstacle-free nominal route plus geometry/full-motion guards only. No with-obstacle outcome selection.',
                'layout_rule': 'Exact default random-demo generator; not the selected examples reanchored along earlier observed executions. O2 lowered5cm after base stagger, then full bounds revalidated.',
                'baseline': 'PPO inference with the same77/7 environment and existing projection/reference wrapper; no added predictive controller or online RRT.',
                'mixture': 'Precommitted per-episode Bernoulli50%; RRT versus PPO+prediction+local recovery; matching cached complete episodes.',
                'measurement': {'zero_command': 'max(abs(requested_joint_action)) < 1e-8',
                                'stationary': 'max(abs(actual_joint_delta)) < 1e-4 rad per real control step',
                                'near_goal_timeout': 'both terminal errors within3 times actual success tolerances',
                                'waiting_timeout': 'otherwise zero commands or stationary steps at least25% of real steps',
                                'wall_time': 'Includes CPU prediction/planning; measured under4 concurrent workers, not a real-time deployment benchmark'}}
    dump(report/'protocol.json', protocol)
    state = {'status': 'running', 'optimizer_updates': 0, 'final_test_episodes': 0}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report/'status.json', state)
        print('VERTICAL_VALIDATION', json.dumps(state), flush=True)
    cases, rejected, results = [], [], {}
    try:
        with ProcessPoolExecutor(max_workers=spec['workers'], mp_context=mp.get_context('spawn'),
                                 initializer=initialize, initargs=(protocol, str(report))) as pool:
            seed = int(spec['first_scene_seed'])
            attempted = 0
            while len(cases) < spec['validation_count']:
                batch = []
                while len(batch) < min(16, max(4, spec['validation_count']-len(cases))):
                    if seed not in used:
                        batch.append(seed)
                    seed += 1
                attempted += len(batch)
                if attempted > spec['maximum_generation_attempts']:
                    raise RuntimeError('Too many geometric scene construction attempts')
                status('generating_geometry', accepted=len(cases), total=spec['validation_count'], attempted=attempted)
                # Consume deterministic seed order, independent of worker timing.
                for outcome in pool.map(construct, batch):
                    if 'case' in outcome and len(cases) < spec['validation_count']:
                        cases.append(outcome['case'])
                    elif 'rejection' in outcome:
                        rejected.append(outcome['rejection'])
                dump(report/'geometry_rejections.json', rejected)
            ids = [int(c['scene']['seed']) for c in cases]
            assert len(set(ids)) == 64 and not set(ids)&used
            router = EpisodeChoice(spec['expert_probability'], spec['mixture_seed'])
            assignments = []
            for current in ids:
                branch = router.begin_episode()
                assignments.append({'seed': current, 'mode': 'rrt' if branch == 'rrt' else 'ppo_progress'})
                router.end_episode()
            dump(data/'validation_cases.json', cases)
            dump(data/'split.json', {'validation': ids, 'excluded_prior_ids': sorted(used), 'train': [], 'test': []})
            dump(report/'mixture_assignments.json', assignments)
            bank_sha, assignment_sha = sha(data/'validation_cases.json'), sha(report/'mixture_assignments.json')
            dump(report/'frozen_inputs.json', {'validation_bank_sha256': bank_sha, 'mixture_assignments_sha256': assignment_sha,
                 'no_outcome_evaluations_before_freeze': True, 'accepted': len(cases), 'rejected': len(rejected),
                 'mixture_rrt_episodes': sum(a['mode']=='rrt' for a in assignments),
                 'mixture_progress_episodes': sum(a['mode']=='ppo_progress' for a in assignments)})
            for mode in MODES:
                rows = []
                status('evaluating', mode=mode, completed=0, total=len(cases))
                futures = {pool.submit(execute, (mode, case)): case['scene']['seed'] for case in cases}
                for future in as_completed(futures):
                    rows.append(future.result())
                    ordered = sorted(rows, key=lambda r: ids.index(r['seed']))
                    dump(report/f'{mode}.json', {'rows': ordered, 'metrics': metrics(ordered)})
                    if len(rows)%4 == 0 or len(rows)==len(cases):
                        status('evaluating', mode=mode, completed=len(rows), total=len(cases), outcomes=metrics(rows))
                results[mode] = sorted(rows, key=lambda r: ids.index(r['seed']))
            mixed = mixture_rows(results, assignments)
            results['mixture'] = mixed
            dump(report/'mixture.json', {'rows': mixed, 'metrics': metrics(mixed), 'physical_rollouts_reused': True})
        assert sha(data/'validation_cases.json') == bank_sha
        assert sha(report/'mixture_assignments.json') == assignment_sha
        for name, digest in hashes.items():
            assert sha(ROOT/name) == digest, f'Frozen runtime code changed: {name}'
        assert sha(ROOT/spec['source_model']) == sha(report/'frozen_model.zip') == original_model_sha
        summary = {'metrics': {mode: metrics(rows) for mode, rows in results.items()},
                   'source_weights_unchanged': True, 'runtime_inputs_unchanged': True,
                   'optimizer_updates': 0, 'final_test_episodes': 0,
                   'validation_scenes': len(cases), 'physical_controller_rollouts': 3*len(cases),
                   'mixture_reuses_matching_complete_episodes': True}
        dump(report/'summary.json', summary)
        status('complete', status='completed', process_exited=True, summary=summary)
    except Exception as exc:
        status('failed', status='failed', error=repr(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/four_vertical_validation.yaml')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    if spec['validation_count'] != 64 or spec['expert_probability'] != .5:
        raise ValueError('Expected64 validation cases and50/50 episode routing')
    print({'mode': 'run' if args.run else 'check_only', 'training': False, 'final_test': False}, flush=True)
    if args.run:
        run(spec, args.config.resolve())


if __name__ == '__main__':
    main()
