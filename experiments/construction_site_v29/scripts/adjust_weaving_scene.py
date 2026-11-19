"""Build an on-path dynamic scene, execute the current controller, and record it.

This is a demonstration layout study, not an independent success-rate benchmark.
The existing PPO weights, reference, collision rule and episode budget are kept.
"""
import _bootstrap
import argparse
import hashlib
from pathlib import Path

import torch
import yaml

from wall_panel_rl.rl.on_path_weaving import build_weaving_case, audit_guide_intersections
from wall_panel_rl.submission import load_agent, load_json, run_episode, save_json
from wall_panel_rl.utils.config import ROOT


PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / 'results/diagnostics/timeout_recovery_v24_seed20261113'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guide', type=Path, default=PREVIOUS / 'recordings/expanded_test_selected/1231415.json')
    parser.add_argument('--runtime', type=Path, default=PREVIOUS / 'selected_runtime.json')
    parser.add_argument('--layout', type=Path, default=ROOT / 'configs/on_path_weaving.yaml')
    parser.add_argument('--layout-seed', type=int, help='Sample another layout without changing the robot start')
    parser.add_argument('--output', type=Path, required=True, help='New directory for the scene, outcome and recording')
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output directory to preserve previous measurements')
    args.output.mkdir(parents=True)
    model_path = ROOT / 'models/current_ppo.zip'
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    save_json(args.output / 'status.json', {'status': 'running', 'phase': 'placement'})
    try:
        runtime = load_json(args.runtime)
        guide = load_json(args.guide)
        layout = yaml.safe_load(args.layout.read_text())
        if args.layout_seed is not None:
            layout['layout_seed'] = args.layout_seed
            layout['placement_mode'] = 'resample'
        case = build_weaving_case(runtime['configuration'], runtime['settings'], guide, layout)
        evidence = audit_guide_intersections(runtime['configuration'], case, guide)
        save_json(args.output / 'case.json', case)
        save_json(args.output / 'layout.json', layout)
        save_json(args.output / 'runtime.json', runtime)
        save_json(args.output / 'geometry.json', evidence)
        save_json(args.output / 'manifest.json', {
            'guide': str(args.guide.resolve()), 'guide_sha256': hashlib.sha256(args.guide.read_bytes()).hexdigest(),
            'model': str(model_path), 'model_sha256': model_hash, 'training_performed': False,
            'demonstration_layout_study': True, 'independent_success_rate_benchmark': False,
        })
        print('Geometry:', evidence, flush=True)
        if not args.geometry_only:
            save_json(args.output / 'status.json', {'status': 'running', 'phase': 'physical_execution'})
            torch.set_num_threads(1)
            result, recording = run_episode(load_agent(model_path), case, 'enhanced', runtime)
            save_json(args.output / 'result.json', result)
            save_json(args.output / 'recording.json', recording)
            print('Physical result:', result, flush=True)
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_hash:
            raise AssertionError('The model file changed during the scene study')
        save_json(args.output / 'status.json', {'status': 'completed', 'phase': 'complete'})
    except BaseException as exc:
        save_json(args.output / 'status.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
