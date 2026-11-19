"""Run a real deterministic episode, record it, then show fixed-rate playback."""
import _bootstrap
import argparse
from pathlib import Path
import torch
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_agent, load_json, run_episode, save_json
from wall_panel_rl.visualization.episode_replay import replay_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['enhanced','ppo','rrt'], default='enhanced')
    parser.add_argument('--bank', type=Path, default=ROOT/'data/demo_case.json')
    parser.add_argument('--seed', type=int, default=1166026)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/demo_recording.json')
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--playback-speed', type=float, default=.6)
    args = parser.parse_args()
    matches = [c for c in load_json(args.bank) if c['scene']['seed']==args.seed]
    if len(matches)!=1:
        parser.error('Seed must identify exactly one case in the selected bank')
    torch.set_num_threads(1)
    print('Stage 1: executing and recording the full physical episode without a window.', flush=True)
    model = load_agent() if args.mode!='rrt' else None
    result, payload = run_episode(model, matches[0], args.mode)
    save_json(args.output, payload)
    print(result, flush=True)
    print(f'Recording saved: {args.output.resolve()}', flush=True)
    if not args.headless:
        print('Stage 2: playback only; no planning, inference, or physics stepping.', flush=True)
        replay_episode(payload, fps=args.fps, speed=args.playback_speed)


if __name__ == '__main__':
    main()
