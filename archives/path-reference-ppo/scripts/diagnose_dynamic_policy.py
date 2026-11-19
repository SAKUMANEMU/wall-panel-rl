"""Diagnose saved PPO policies on validation scenes, without resuming the test set.

Dynamic runs reproduce checkpoint selection. Frozen/no-obstacle runs are diagnostic
ablations, not benchmark results. Near-target labels never change success criteria.
"""
import _bootstrap
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p
import torch
import yaml
from stable_baselines3 import PPO

from wall_panel_rl.envs.wall_panel_env import WallPanelEnv

ROOT = Path('models/dynamic70_study')
OUT = Path('results/diagnostics/dynamic70_failure')


def collision_pairs(env):
    """Name penetrating pairs using the environment's collision exclusions."""
    pairs = []
    bodies = [(env.robot, 'robot'), (env.panel, 'panel'), (env.wall, 'wall')]
    bodies += [(body, f'obstacle_{i}') for i, body in enumerate(env.obstacle_ids)]
    names = dict(bodies)
    for a, b in [(env.robot, env.robot), (env.robot, env.panel)] + [
        (moving, other) for moving in [env.robot, env.panel]
        for other in env.obstacle_ids + [env.wall]
    ]:
        contacts = list(p.getContactPoints(a, b, physicsClientId=env.client))
        contacts += list(p.getClosestPoints(a, b, 0, physicsClientId=env.client))
        for contact in contacts:
            if contact[8] > 0:
                continue
            la, lb = contact[3:5]
            if a == b and abs(la-lb) <= 1:
                continue
            if b == env.panel and a == env.robot and la == env.ee:
                continue
            pairs.append(f'{names[a]}[{la}]--{names[b]}[{lb}]')
    return sorted(set(pairs))


def run_episode(env, model, scene, condition, checkpoint, variant):
    scene = copy.deepcopy(scene)
    if variant == 'no_obstacles':
        scene['obstacles'] = []
        scene['obstacle_motion'] = []
    obs, initial = env.reset(seed=scene['seed'], options={
        'scenario': scene, 'noise_seed': scene['seed']+1000000})
    trace, terms = [], {}
    for step in range(1, env.config['simulation']['max_steps']+1):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        for key, value in info.items():
            if key.startswith('reward_'):
                terms[key] = terms.get(key, 0.) + float(value)
        trace.append({'step': step, 'position_m': float(info['position_error']),
                      'orientation_deg': float(np.rad2deg(info['orientation_error'])),
                      'action_norm': float(np.linalg.norm(action)), 'reward': reward})
        if done or truncated:
            break
    pos, angle = trace[-1]['position_m'], trace[-1]['orientation_deg']
    reason = ('success' if info['success'] else 'collision' if info['collision']
              else 'joint_limit' if info['joint_limit_violation'] else 'timeout')
    # These bins describe timeout severity only, not a relaxed definition of success.
    detail = ('position_only' if pos >= .01 and angle < 2 else
              'orientation_only' if pos < .01 and angle >= 2 else 'both')
    row = {'condition': condition, 'checkpoint': checkpoint, 'variant': variant,
           'seed': scene['seed'], 'reason': reason, 'steps': step,
           'collision': bool(info['collision']), 'joint_limit': bool(info['joint_limit_violation']),
           'position_m': pos, 'orientation_deg': angle,
           'initial_position_m': float(initial['position_error']),
           'initial_orientation_deg': float(np.rad2deg(initial['orientation_error'])),
           'min_position_m': min(t['position_m'] for t in trace),
           'min_orientation_deg': min(t['orientation_deg'] for t in trace),
           'ever_jointly_near_5cm_10deg': any(t['position_m'] < .05 and t['orientation_deg'] < 10 for t in trace),
           'timeout_detail': detail if reason == 'timeout' else '',
           'near_target_timeout_5cm_10deg': reason == 'timeout' and pos < .05 and angle < 10,
           'last40_position_improvement_m': trace[max(0,len(trace)-40)]['position_m']-pos,
           'last40_mean_action_norm': float(np.mean([t['action_norm'] for t in trace[-40:]])),
           'collision_pairs': ';'.join(collision_pairs(env)) if info['collision'] else '',
           'cumulative_reward': sum(t['reward'] for t in trace), **terms}
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--followup', action='store_true',
                        help='Append longer-horizon and original alignment-policy checks')
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=args.followup)
    scenes = json.loads((ROOT/'scenes/validation.json').read_text())
    rows = pd.read_csv(OUT/'episodes.csv').fillna('').to_dict('records') if args.followup else []
    traces = json.loads((OUT/'traces.json').read_text()) if args.followup else []
    if args.followup and any(r['variant'] == 'extended_dynamic' for r in rows):
        raise RuntimeError('Follow-up results already exist')
    runs = ([('best', ['extended_dynamic']), ('alignment_final', ['no_obstacles'])]
            if args.followup else [('best', ['dynamic', 'frozen', 'no_obstacles']),
                                   ('final', ['dynamic'])])
    for condition in ['clean', 'noise']:
        config = yaml.safe_load((ROOT/f'ppo_{condition}/config.yaml').read_text())
        for checkpoint, variants in runs:
            model = PPO.load(ROOT/f'ppo_{condition}/{checkpoint}_model.zip', device='cpu')
            for variant in variants:
                c = copy.deepcopy(config)
                if variant == 'frozen':
                    c['obstacles']['freeze_motion'] = True
                if variant == 'extended_dynamic':
                    c['simulation']['max_steps'] = 720
                env = WallPanelEnv(c)
                try:
                    for scene in scenes:
                        row, trace = run_episode(env, model, scene, condition, checkpoint, variant)
                        rows.append(row)
                        traces.append({k: row[k] for k in ['condition','checkpoint','variant','seed']} | {'trace': trace})
                finally:
                    env.close()
                pd.DataFrame(rows).to_csv(OUT/'episodes.csv', index=False)
                print(condition, checkpoint, variant, len(rows), flush=True)
    (OUT/'traces.json').write_text(json.dumps(traces))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['condition','checkpoint','variant','reason']).size()
    summary.rename('episodes').to_csv(OUT/'failure_counts.csv')
    print(summary.to_string())


if __name__ == '__main__':
    main()
