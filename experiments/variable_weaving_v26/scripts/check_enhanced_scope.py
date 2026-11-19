"""Check baseline invariance, diverse recovery and expanded generation before run."""
import _bootstrap
import copy,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from run_enhanced_scope import initialize,generate,runtime_for,policy
from run_adaptation import save,digest
from wall_panel_rl.submission import load_json,run_episode
from wall_panel_rl.utils.config import ROOT

def main():
    initialize();out=ROOT/'checks_scope';out.mkdir(exist_ok=True)
    case=load_json(ROOT/'data/validation_cases.json')[0]
    reference=load_json(Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/dynamic_adaptation_v21_seed20261112/evaluations/regression64.json'))['enhanced']['rows']
    expected=next(x for x in reference if x['seed']==case['scene']['seed'])
    row,_=run_episode(policy('v21'),case,'enhanced',runtime_for('original'))
    assert row['reason']==expected['reason'] and row['steps']==expected['steps']
    assert abs(row['position_m']-expected['position_m'])<1e-9
    assert abs(row['orientation_deg']-expected['orientation_deg'])<1e-8
    diverse,recording=run_episode(policy('v21'),case,'enhanced',runtime_for('diverse'))
    assert diverse['reason'] in ['success','collision','timeout','joint_limit']
    assert diverse['reference_preserved'] and not diverse['uses_global_rrt']
    assert not diverse['weights_changed']
    save(out/'execution_check.json',{'baseline_exact_reproduction':True,'source_sha256':digest(ROOT/'models/current_ppo.zip'),
        'baseline_result':row,'diverse_interface_result':diverse})
    # Try the first eight new seeds per fixed guide and journal every rejection.
    accepted={};attempts=[]
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for index in range(8):
            active=[tag for tag in ('high','low') if tag not in accepted]
            if not active:break
            tasks=[(tag,1231000+2*index+int(tag=='low')) for tag in active]
            for result in pool.map(generate,tasks):
                attempts.append({k:v for k,v in result.items() if k!='case'})
                if result['geometry_valid']:accepted[result['guide']]=result['case']
            save(out/'generation_attempts.json',attempts);save(out/'accepted_cases.json',accepted)
    if len(accepted)<2:raise RuntimeError('Expanded generation needs review; see preflight rejection log')
    save(out/'summary.json',{'passed':True,'baseline_exact_reproduction':True,
        'diverse_execution_checked':True,'accepted_guides':list(accepted),
        'generation_attempts':len(attempts),'training_performed':False})
    print('ENHANCED SCOPE PREFLIGHT PASSED',flush=True)
if __name__=='__main__':main()
