"""Physical check of timeout-specific controller on one known stalled scene."""
import _bootstrap
import copy
from pathlib import Path
from run_timeout_study import initialize,runtime_for,CACHE
from run_adaptation import save,digest
from wall_panel_rl.submission import load_json,run_episode
from wall_panel_rl.utils.config import ROOT

def main():
    initialize();out=ROOT/'checks_timeout';out.mkdir(exist_ok=True)
    source=ROOT/'models/current_ppo.zip';sha=digest(source)
    case=next(c for c in load_json(ROOT/'data/low_diagnostic_cases.json') if c['scene']['seed']==1225022)
    result,recording=run_episode(CACHE['model'],case,'enhanced',runtime_for('detour'))
    assert result['reference_preserved'] and not result['uses_global_rrt'] and not result['weights_changed']
    assert digest(source)==sha
    save(out/'pilot_result.json',result);save(out/'pilot_recording.json',recording)
    save(out/'verification.json',{'physical_check_passed':True,'seed':case['scene']['seed'],
        'reason':result['reason'],'source_model_unchanged':True,'global_rrt_used':False})
    print('TIMEOUT CONTROLLER PHYSICAL CHECK PASSED',result,flush=True)
if __name__=='__main__':main()
