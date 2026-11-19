"""Validate farther starts first, then exactly three randomly placed obstacles."""
import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--start-seed', type=int, default=10000)
    parser.add_argument('--run-id', default='far_random_three')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    output = Path('results/raw') / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    rows, summaries, saved_scenarios = [], {}, {}
    reference = WallPanelEnv(load_config())
    original_distances = []
    try:
        for seed in range(args.start_seed, args.start_seed + args.episodes):
            _, info = reference.reset(seed=seed)
            original_distances.append(info['position_error'])
    finally:
        reference.close()
    (output/'arguments.json').write_text(json.dumps(vars(args), indent=2))
    for stage in ['far_start', 'far_three_obstacles']:
        config = load_config(f'configs/{stage}.yaml')
        (output/f'{stage}_config.yaml').write_text(yaml.safe_dump(config))
        env = WallPanelEnv(config)
        scenarios = []
        try:
            for index, seed in enumerate(range(args.start_seed, args.start_seed + args.episodes)):
                audit = {}
                def check_plan(e, controller):
                    audit.update(
                        initial_position_error_m=e.info()['position_error'],
                        obstacle_count=len(e.obstacles),
                        direct_path_valid=None if controller.goal_q is None else e.collision.edge_valid(e.joint_state()[0], controller.goal_q),
                        rrt_iterations=controller.planner.iterations,
                        planned_path_valid=False if controller.path is None else all(
                            e.collision.edge_valid(a,b) for a,b in zip(controller.path[:-1],controller.path[1:])),
                    )
                result, trajectory = rollout(env, 'rrt', seed, on_plan=check_plan)
                result.update(audit, stage=stage, original_position_error_m=original_distances[index])
                result['successful_detour'] = bool(result['direct_path_valid'] is False and result['success'])
                rows.append(result)
                scenarios.append(trajectory['scenario'])
                if index < 3:
                    (output/f'{stage}_trajectory_{seed}.json').write_text(json.dumps(trajectory, indent=2))
                pd.DataFrame(rows).to_csv(output/'episodes.csv', index=False)
                if (index+1)%10 == 0:
                    print(f'{stage}: {index+1}/{args.episodes}', flush=True)
        finally:
            env.close()
        saved_scenarios[stage] = scenarios
        frame = pd.DataFrame([r for r in rows if r['stage'] == stage])
        summaries[stage] = {
            'episodes':len(frame),
            'original_mean_distance_m':float(frame.original_position_error_m.mean()),
            'initial_distance_mean_m':float(frame.initial_position_error_m.mean()),
            'initial_distance_min_m':float(frame.initial_position_error_m.min()),
            'initial_distance_max_m':float(frame.initial_position_error_m.max()),
            'success_count':int(frame.success.sum()),
            'collision_count':int(frame.collision.sum()),
            'ik_failure_count':int((~frame.ik_success).sum()),
            'planning_failure_count':int((~frame.planning_success).sum()),
            'blocked_direct_paths':sum(r['direct_path_valid'] is False for r in rows if r['stage']==stage),
            'successful_detours':int(frame.successful_detour.sum()),
            'mean_steps':float(frame.steps.mean()),
            'mean_final_position_error_m':float(frame.position_error_m.mean()),
            'mean_final_orientation_error_deg':float(frame.orientation_error_deg.mean()),
        }
        print(json.dumps(summaries[stage], indent=2), flush=True)
    assert [s['initial_q'] for s in saved_scenarios['far_start']] == [s['initial_q'] for s in saved_scenarios['far_three_obstacles']]
    (output/'scenarios.json').write_text(json.dumps(saved_scenarios, indent=2))
    (output/'summary.json').write_text(json.dumps(summaries, indent=2))
    print(f'Recorded physical executions and detour audits in {output}', flush=True)


if __name__ == '__main__':
    main()
