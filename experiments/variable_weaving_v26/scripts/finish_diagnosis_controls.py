"""Continue only unfinished diagnostic controls; no model fitting."""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from diagnose_standalone import OUT,RUN,run_rollout
from run_adaptation import Status,jobs,save,digest,initialize_worker
from wall_panel_rl.submission import load_json
from pathlib import Path

def main():
    status=Status(OUT);status.update('resume_diagnostic_controls')
    results=load_json(OUT/'rollouts.json');bank=load_json(RUN/'frozen_test_cases.json')
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        for variant in ('zero','no_obstacles','static','rrt'):
            existing=[r for r in results if r['group']=='fresh32' and r['variant']==variant]
            if len(existing)==len(bank):continue
            assert not existing,'Partial saved controls require explicit seed reconciliation'
            rows=jobs(pool,run_rollout,[('fresh32',c,'selected',variant) for c in bank],status,'control_'+variant)
            results.extend(rows);save(OUT/'rollouts.json',results)
    for filename,sha in load_json(OUT/'protocol.json')['input_hashes'].items():assert digest(filename)==sha
    save(OUT/'verification.json',{'frozen_inputs_unchanged':True,'training_performed':False,
        'interface_cases_checked':len(load_json(OUT/'interface_checks.json')),'rollouts':len(results),
        'no_obstacle_clearance_recorded_as_null':True,'final_diagnostic_code_sha256':digest(Path(__file__).with_name('diagnose_standalone.py'))})
    status.update('complete',status='completed')
if __name__=='__main__':main()
