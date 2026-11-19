"""Explicit two-static diagnostic demo; the verified default one-static is preserved."""
import _bootstrap
import sys
from wall_panel_rl.utils.config import ROOT
from demo_autonomous_ppo import main

if __name__=='__main__':
    report=ROOT/'results/diagnostics/protected_two_static_v2_seed61'
    print('Two-static candidate demonstration; consult assessment.json for qualification.',flush=True)
    sys.argv=[sys.argv[0],
        '--model',str(report/'candidate_model.zip'),'--manifest',str(report/'candidate_manifest.json'),
        '--scene-bank',str(ROOT/'data/protected_two_static_v2_seed61/two_validation.json'),'--seed','900001',*sys.argv[1:]]
    main(default_controller='mixture')
