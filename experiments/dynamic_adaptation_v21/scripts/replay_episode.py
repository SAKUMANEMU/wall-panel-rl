"""Replay a saved episode without loading a model, planning, or physical steps."""
import _bootstrap
import argparse
import json
from pathlib import Path
from wall_panel_rl.visualization.episode_replay import RecordedEpisode, replay_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--playback-speed', type=float, default=.6)
    parser.add_argument('--hold-seconds', type=float, default=30.)
    parser.add_argument('--start-hold-seconds', type=float, default=2.)
    parser.add_argument('--headless', action='store_true', help='Validate playback poses without a window or wall-clock waits')
    parser.add_argument('--stats-output', type=Path)
    args = parser.parse_args()
    try:
        stats = replay_episode(RecordedEpisode.load(args.recording), fps=args.fps,
                               speed=args.playback_speed, headless=args.headless,
                               start_hold=args.start_hold_seconds, end_hold=args.hold_seconds)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, indent=2)+'\n')


if __name__ == '__main__':
    main()
