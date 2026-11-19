"""Check spawn-worker scene generation independently of the formal split."""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from run_adaptation import initialize_worker,generate_one,save
from wall_panel_rl.utils.config import ROOT

def main():
    with ProcessPoolExecutor(1,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        result=pool.submit(generate_one,1209003).result()
    assert result['geometry_valid'],result
    case=result['case'];assert case['task']=='dynamic'
    assert len(case['scene']['obstacles'])==4
    assert len(case['scene']['obstacle_motion'])==4
    save(ROOT/'checks/generation_smoke.json',{'passed':True,'seed':result['seed'],
        'count':4,'task':'dynamic','formal_split_member':False})
    print('SPAWN GENERATION CHECK PASSED',flush=True)
if __name__=='__main__':main()
