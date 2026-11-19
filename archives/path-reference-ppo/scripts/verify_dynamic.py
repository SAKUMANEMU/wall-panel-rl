"""Record physical executions with five moving obstacles, including failures."""
import _bootstrap
import argparse
import json
from pathlib import Path
import yaml
import pybullet as bullet
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--start-seed', type=int, default=10000)
    p.add_argument('--run-id', required=True)
    args = p.parse_args()
    if args.episodes < 1:
        p.error('--episodes must be positive')
    folder = Path('results/raw') / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    config = load_config('configs/far_five_dynamic.yaml')
    print(f"Planning mode: {config['rrt'].get('dynamic_mode', 'swept')}", flush=True)
    (folder/'config.yaml').write_text(yaml.safe_dump(config))
    e = WallPanelEnv(config)
    rows = []
    try:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            entered, crossing_steps = set(), []
            def audit_crossing(env, step, info):
                if info['collision']:
                    return
                with env.planning_obstacles():
                    hit = False
                    for i, body in enumerate(env.obstacle_ids):
                        if any(bullet.getClosestPoints(moving,body,0,physicsClientId=env.client)
                               for moving in [env.robot, env.panel]):
                            entered.add(i)
                            hit = True
                    if hit:
                        crossing_steps.append(step)
            result, trajectory = rollout(e, 'rrt', seed, on_step=audit_crossing)
            result['swept_envelope_entry_steps'] = len(crossing_steps)
            result['entered_obstacle_envelopes'] = sorted(entered)
            trajectory['swept_envelope_entry_steps'] = crossing_steps
            rows.append(result)
            (folder/'episodes.json').write_text(json.dumps(rows, indent=2))
            if seed == args.start_seed + 2:
                (folder/'trajectory.json').write_text(json.dumps(trajectory, indent=2))
            print(seed, result['success'], result['collision'], result['replans'], flush=True)
    finally:
        e.close()
    summary = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                   collisions=sum(r['collision'] for r in rows),
                   planning_failures=sum(not r['planning_success'] for r in rows),
                   replans=sum(r['replans'] for r in rows),
                   scheduled_waits=sum(r['scheduled_waits'] for r in rows),
                   successful_envelope_crossings=sum(r['success'] and r['swept_envelope_entry_steps'] > 0 for r in rows))
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
