"""Explicit coverage/consistency candidate demo; does not switch the default."""
import _bootstrap
import json
import sys
from wall_panel_rl.utils.config import ROOT
from demo_autonomous_ppo import main


if __name__ == '__main__':
    report = ROOT/'results/diagnostics/consistent_two_static_v4_seed61'
    selection = json.loads((report/'demo_selection.json').read_text())
    if selection['seed'] is None: raise RuntimeError('No successful demonstration was recorded')
    print('Coverage/consistency diagnostic candidate; consult assessment.json. Default model preserved.', flush=True)
    sys.argv = [sys.argv[0], '--model', str(report/'candidate_model.zip'),
        '--manifest', str(report/'candidate_manifest.json'), '--scene-bank', selection['scene_bank'],
        '--seed', str(selection['seed']), *sys.argv[1:]]
    main(default_controller='ppo')
