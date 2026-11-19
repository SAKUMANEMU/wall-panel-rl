"""Plot recorded real execution and clearly distinguish it from the reference."""
import _bootstrap
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wall_panel_rl.utils.config import ROOT


def main():
    root = ROOT/'results/diagnostics/four_vertical_weave_v15'
    result = json.loads((root/'selected_trajectory.json').read_text())
    case = json.loads((root/'selected_case.json').read_text())[0]
    trace = result['trajectory']
    xyz = np.array([r['panel'] for r in trace])
    reference = np.array(result['nominal_panel_positions'])
    placement = np.array(case['scene']['nominal_motion']['panel_positions'])
    seconds = np.array([r['time_s'] for r in trace])
    anchors = np.array([m['anchor'] for m in case['scene']['obstacle_motion']])
    fig = plt.figure(figsize=(13, 5.3), layout='constrained')
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(*reference.T, '--', color='gray', label='Original observation reference')
    ax.plot(*placement.T, ':', color='mediumpurple', label='Route used to place obstacles')
    ax.plot(*xyz.T, color='royalblue', lw=2, label='Executed panel center')
    for i, (anchor, box) in enumerate(zip(anchors, case['scene']['obstacles'])):
        ax.scatter(*anchor, color='darkorange', s=60)
        ax.text(*anchor, f'  O{i+1}')
        dims = np.array(box['dimensions'])
        for step in np.linspace(0, len(trace)-1, 4, dtype=int):
            center = np.array(trace[step]['obstacle_centers'][i])
            # Faint wire boxes are four sampled moving poses, not static barriers.
            for axis in range(3):
                others = [j for j in range(3) if j != axis]
                for a in (-1, 1):
                    for b in (-1, 1):
                        points = np.stack([center-dims/2, center+dims/2])
                        points[:, others[0]] = center[others[0]]+a*dims[others[0]]/2
                        points[:, others[1]] = center[others[1]]+b*dims[others[1]]/2
                        ax.plot(*points.T, color='darkorange', alpha=.15, lw=.6)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Dynamic boxes: sampled poses and anchors')
    ax.legend(loc='upper left', fontsize=8)
    ax = fig.add_subplot(122)
    ax.plot(seconds, xyz[:, 2], color='royalblue', label='Executed panel height')
    ax.plot(seconds, [r['reference_z'] for r in trace], '--', color='gray', label='Reference height at nearest joint projection')
    for i in range(4):
        ax.plot(seconds, [r['obstacle_centers'][i][2] for r in trace], alpha=.6, lw=1, label=f'O{i+1} center height')
    ax.set(xlabel='Simulation time (s)', ylabel='Z (m)', title=f"Measured successful example: seed {result['seed']}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.suptitle('Larger vertical stagger — selected development demonstration', fontsize=13)
    fig.savefig(root/'trajectory.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
