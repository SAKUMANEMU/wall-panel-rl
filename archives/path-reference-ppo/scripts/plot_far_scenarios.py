"""Plot the separate farther-start/random-three-box audit from recorded data."""
import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='far_random_three')
    parser.add_argument('--seed', type=int, default=10002)
    args = parser.parse_args()
    folder = Path('results/raw') / args.run_id
    frame = pd.read_csv(folder/'episodes.csv')
    starts = frame[frame.stage == 'far_start']
    trajectory = json.loads((folder/f'far_three_obstacles_trajectory_{args.seed}.json').read_text())
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(121)
    ax.hist(starts.original_position_error_m, bins=12, alpha=.75, label='Original starts')
    ax.hist(starts.initial_position_error_m, bins=12, alpha=.75, label='Farther starts')
    ax.set(xlabel='Initial panel-to-target distance (m)', ylabel='Episodes', title='Measured initial distances (same seeds)')
    ax.legend()
    scene = trajectory['scenario']
    points = np.array(trajectory['panel_positions'])
    ax = fig.add_subplot(122, projection='3d')
    ax.plot(*points.T, color='tab:blue', label='Executed RRT panel path')
    ax.scatter(*points[0], marker='o', color='tab:orange', label='Start')
    ax.scatter(*scene['target_position'], marker='*', color='green', s=90, label='Target')
    for obstacle in scene['obstacles']:
        center, size = np.array(obstacle['center']), np.array(obstacle['dimensions'])
        ax.bar3d(*(center-size/2), *size, color='red', alpha=.25)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title=f'Three random obstacles: seed {args.seed}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    target=Path('results/figures')/f'{args.run_id}.png'
    fig.savefig(target, dpi=180)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
