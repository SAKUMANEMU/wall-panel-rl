"""Development-only geometric/trajectory check, not an independent success test.

Reanchor eight existing scenes without selecting on controller outcomes. Record
real execution only. Select a successful animation by its measured vertical
departure from the original reference, and disclose that selection explicitly.
"""
import _bootstrap
import argparse
import copy
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors
from wall_panel_rl.rl.slow_dynamic_four import dynamic_configuration, SlowDynamicFourEnv
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController

REPORT = ROOT/'results/diagnostics/four_vertical_weave_v15'
WORK = {}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def initialize():
    torch.set_num_threads(1)
    profile = yaml.safe_load((ROOT/'configs/four_vertical_weave.yaml').read_text())
    manifest = json.loads((ROOT/profile['source_report']/'candidate_manifest.json').read_text())
    config, settings = dynamic_configuration(manifest['configuration'], manifest['evaluation_settings'], profile)
    WORK.update(profile=profile, config=config, settings=settings,
                model=PPO.load(ROOT/profile['source_report']/'selected_model.zip', device='cpu'),
                control=json.loads((ROOT/'configs/four_progress_controller.json').read_text()))


def construct(original):
    try:
        return {'case': fit_motion_anchors(WORK['config'], WORK['settings'], WORK['profile'], original)}
    except (RuntimeError, ValueError) as exc:
        return {'rejection': {'seed': original['scene']['seed'], 'reason': str(exc)}}


def execute(case):
    env = SlowDynamicFourEnv(copy.deepcopy(WORK['config']), [], copy.deepcopy(WORK['settings']))
    try:
        obs, info = env.reset(options={'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']})
        path = env.cursor.path.copy()
        # Recover exact panel-center positions for the immutable joint reference.
        nominal = []
        for q in path:
            with env.env.collision.at_configuration(q):
                nominal.append(list(env.env.panel_pose()[0]))
        control = WORK['control']
        agent = ProgressRecoveryController(WORK['model'], env, control['selected']['horizon'],
                                           control['gains'], control['recovery'])
        trace = []
        def record():
            q = env.env.joint_state()[0]
            # Project joints onto the reference polyline, rather than matching
            # elapsed times (safe execution can deliberately wait or reverse).
            delta = np.diff(path, axis=0)
            t = np.clip(np.sum((q-path[:-1])*delta, axis=1)/np.maximum(np.sum(delta*delta, axis=1), 1e-12), 0, 1)
            projected = path[:-1]+t[:, None]*delta
            i = int(np.argmin(np.sum((projected-q)**2, axis=1)))
            z = float((1-t[i])*nominal[i][2]+t[i]*nominal[i+1][2])
            trace.append({'step': env.env.steps, 'time_s': env.env.sim_time,
                          'joints': q.tolist(),
                          'panel': list(env.env.panel_pose()[0]), 'reference_z': z,
                          'reference_index': int(env.cursor.index),
                          'obstacle_centers': [list(center) for center, _ in env.env.obstacles]})
        record()
        while True:
            clock = (env.env.steps, env.env.sim_time)
            action = agent.action(obs)
            assert clock == (env.env.steps, env.env.sim_time)
            obs, _, done, truncated, info = env.step(action)
            record()
            if done or truncated:
                break
        assert np.array_equal(path, env.cursor.path)
        assert env.env.config['simulation']['max_steps'] == 240
        dz = np.array([r['panel'][2]-r['reference_z'] for r in trace])
        positions = np.array([r['panel'] for r in trace])
        # Separate actual up/down travel from deviation relative to reference.
        vertical = np.diff(positions[:, 2])
        return {'seed': case['scene']['seed'], 'reason': info['termination_reason'],
                'steps': env.env.steps, 'position_m': float(info['position_error']),
                'orientation_deg': float(np.rad2deg(info['orientation_error'])),
                'anchor_height_span_m': case['scene']['motion_anchor_placement']['anchor_height_span_m'],
                'upward_travel_m': float(np.maximum(vertical, 0).sum()),
                'downward_travel_m': float(-np.minimum(vertical, 0).sum()),
                'reference_departure_up_m': float(max(0, dz.max())),
                'reference_departure_down_m': float(max(0, -dz.min())),
                'recovery_actions': agent.recovery_actions, 'trajectory': trace,
                'nominal_panel_positions': nominal}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        print('Check only; use --run to generate and execute eight development scenes.')
        return
    if REPORT.exists():
        raise FileExistsError('Refusing to overwrite previous development results')
    REPORT.mkdir(parents=True)
    visible = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/REPORT.name
    if visible.resolve() != REPORT.resolve():
        visible.symlink_to(REPORT, target_is_directory=True)
    def status(phase, **extra):
        value = dict(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(REPORT/'status.json', value)
        print(json.dumps(value), flush=True)
    try:
        old = json.loads((ROOT/'data/four_progress_recovery_test_v14_seed121/test_cases.json').read_text())
        cases, rejected, rows = [], [], []
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
            for start in range(0, len(old), 8):
                status('geometry', accepted=len(cases), target=8)
                for result in pool.map(construct, old[start:start+8]):
                    if 'case' in result and len(cases) < 8:
                        cases.append(result['case'])
                    elif 'rejection' in result:
                        rejected.append(result['rejection'])
                if len(cases) == 8:
                    break
            if len(cases) != 8:
                raise RuntimeError('Could not construct eight valid enlarged-height scenes')
            dump(REPORT/'cases.json', cases)
            dump(REPORT/'rejections.json', rejected)
            for row in pool.map(execute, cases):
                rows.append(row)
                dump(REPORT/'episodes.json', rows)
                status('execution', completed=len(rows), total=len(cases),
                       outcomes={k: sum(r['reason']==k for r in rows) for k in ['success', 'collision', 'timeout']})
        successes = [r for r in rows if r['reason']=='success']
        if not successes:
            raise RuntimeError('No successful development animation; do not promote this profile')
        selected = max(successes, key=lambda r: (min(r['reference_departure_up_m'], r['reference_departure_down_m']),
                                                 min(r['upward_travel_m'], r['downward_travel_m'])))
        summary = {k:v for k,v in selected.items() if k not in ('trajectory', 'nominal_panel_positions')}
        summary.update(scope='Eight development scenes; reused source references; not an independent success-rate estimate',
                       selection='Successful example maximizing measured two-sided vertical departure from reference',
                       outcomes={k:sum(r['reason']==k for r in rows) for k in ['success','collision','timeout']},
                       model_weights_changed=False, optimizer_updates=0, real_step_limit=240,
                       profile='configs/four_vertical_weave.yaml')
        dump(REPORT/'selected_example.json', summary)
        dump(REPORT/'selected_case.json', [c for c in cases if c['scene']['seed']==selected['seed']])
        dump(REPORT/'selected_trajectory.json', selected)
        status('completed', summary=summary)
    except Exception as exc:
        status('failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
