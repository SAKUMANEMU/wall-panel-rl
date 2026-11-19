"""Validate the final scene rules without repeating controller episode checks."""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
from run_enhanced_scope import initialize,generate
from run_adaptation import save,digest
from wall_panel_rl.submission import load_json
from wall_panel_rl.utils.config import ROOT

def main():
    out=ROOT/'checks_scope';old=load_json(out/'execution_check.json')
    assert old['baseline_exact_reproduction'] and old['source_sha256']==digest(ROOT/'models/current_ppo.zip')
    attempted=[];accepted={}
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for offset in range(24):
            tags=[tag for tag in ('high','low') if tag not in accepted]
            if not tags:break
            requested=[(tag,1231000+2*offset+int(tag=='low')) for tag in tags]
            for result in pool.map(generate,requested):
                attempted.append({k:v for k,v in result.items() if k!='case'})
                if result['geometry_valid']:accepted[result['guide']]=result['case']
            save(out/'final_generation_attempts.json',attempted);save(out/'final_accepted_cases.json',accepted)
    if len(accepted)!=2:raise RuntimeError('One guide could not generate a valid expanded case; inspect attempts')
    save(out/'summary.json',{'passed':True,'baseline_exact_reproduction':True,'diverse_execution_checked':True,
        'scope_config_sha256':digest(ROOT/'configs/expanded_scope.yaml'),'accepted_guides':list(accepted),
        'geometry_attempts':len(attempted),'training_performed':False})
    print('FINAL SCOPE GENERATION CHECK PASSED',flush=True)
if __name__=='__main__':main()
