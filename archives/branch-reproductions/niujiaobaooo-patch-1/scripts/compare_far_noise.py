"""Paired RRT comparison: same scene seeds, clean versus noisy target observations."""
import _bootstrap
import argparse
import json
from pathlib import Path
import pandas as pd
import yaml
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--start-seed', type=int, default=10000)
    parser.add_argument('--run-id', default='far_sparse_noise_comparison')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    folder = Path('results/raw') / args.run_id
    folder.mkdir(parents=True, exist_ok=False)
    rows, scenes, summary = [], {}, {}
    for name, filename in [('clean','far_three_obstacles'), ('noise','far_three_obstacles_noise')]:
        config = load_config(f'configs/{filename}.yaml')
        (folder/f'{name}_config.yaml').write_text(yaml.safe_dump(config))
        env = WallPanelEnv(config)
        scenes[name] = []
        try:
            for seed in range(args.start_seed, args.start_seed + args.episodes):
                result, trajectory = rollout(env, 'rrt', seed)
                result['condition'] = name
                rows.append(result)
                scenes[name].append(trajectory['scenario'])
                if seed == args.start_seed:
                    (folder/f'{name}_trajectory.json').write_text(json.dumps(trajectory, indent=2))
                pd.DataFrame(rows).to_csv(folder/'episodes.csv', index=False)
                print(f'{name} seed={seed} success={result["success"]} collision={result["collision"]}', flush=True)
        finally:
            env.close()
        frame = pd.DataFrame([r for r in rows if r['condition'] == name])
        summary[name] = {'episodes': len(frame), 'successes': int(frame.success.sum()),
                         'collisions': int(frame.collision.sum()),
                         'planning_failures': int((~frame.planning_success).sum()),
                         'mean_final_position_error_m': float(frame.position_error_m.mean()),
                         'mean_final_orientation_error_deg': float(frame.orientation_error_deg.mean())}
    assert scenes['clean'] == scenes['noise'], 'Paired scene geometry differs'
    (folder/'scenarios.json').write_text(json.dumps(scenes, indent=2))
    (folder/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
