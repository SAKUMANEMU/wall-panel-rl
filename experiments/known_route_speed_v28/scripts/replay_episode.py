"""Replay a saved episode without loading a model, planning, or physical steps."""
import _bootstrap
import argparse
import json
from pathlib import Path
from wall_panel_rl.visualization.episode_replay import RecordedEpisode
from wall_panel_rl.visualization.smooth_replay import replay_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--fps', type=float, default=60.)
    parser.add_argument('--playback-speed', type=float, default=.6)
    parser.add_argument('--hold-seconds', type=float, default=30.)
    parser.add_argument('--start-hold-seconds', type=float, default=2.)
    parser.add_argument('--headless', action='store_true', help='Validate playback poses without a window or wall-clock waits')
    parser.add_argument('--stats-output', type=Path)
    parser.add_argument('--timing', choices=['smooth','realtime'], default='smooth',
                        help='Smooth keeps every pose; realtime may skip late poses to keep clock speed')
    parser.add_argument('--no-adaptive-fps', action='store_true', help='Disable display-speed calibration')
    parser.add_argument('--show-path', action='store_true', help='Draw the recorded path once before playback')
    parser.add_argument('--shadows', action='store_true', help='Enable more expensive shadows')
    args = parser.parse_args()
    try:
        stats = replay_episode(RecordedEpisode.load(args.recording), fps=args.fps,
                               speed=args.playback_speed, headless=args.headless,
                               start_hold=args.start_hold_seconds, end_hold=args.hold_seconds,
                               pacing=args.timing, adaptive_fps=not args.no_adaptive_fps,
                               show_path=args.show_path, shadows=args.shadows)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(stats, indent=2), flush=True)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(json.dumps(stats, indent=2)+'\n')


if __name__ == '__main__':
    main()
