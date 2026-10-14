"""Record physical executions with five moving obstacles, including failures."""
import _bootstrap
import argparse
import json
from pathlib import Path
import yaml
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
    (folder/'config.yaml').write_text(yaml.safe_dump(config))
    e = WallPanelEnv(config)
    rows = []
    try:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            result, trajectory = rollout(e, 'rrt', seed)
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
                   replans=sum(r['replans'] for r in rows))
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == '__main__':
    main()
