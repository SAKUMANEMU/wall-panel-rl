"""Paired generator probe: same preliminary scene, two frozen placement models."""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
import torch
from wall_panel_rl.submission import load_json,load_agent,load_runtime
from wall_panel_rl.rl.slow_dynamic_four import load_profile
from wall_panel_rl.rl.motion_aware_placement import fresh_motion_aware_case
from wall_panel_rl.rl.demo_scene_adjustments import adjust_obstacle_heights
from wall_panel_rl.rl.vertical_corridor import preliminary_route
from run_adaptation import save,digest
from wall_panel_rl.utils.config import ROOT
from diagnose_standalone import RUN,OUT

def probe(seed):
    torch.set_num_threads(1)
    oldpath=Path('/home/johnsonchen/wall-panel-rl/Learning_Agent_CR/models/current_ppo.zip')
    old=load_agent(oldpath);new=load_agent(ROOT/'models/current_ppo.zip');rt=load_runtime()
    profile=load_profile(ROOT/'configs/preliminary_scene.yaml')
    # The SAME preliminary case is passed to both controllers. We do not
    # regenerate another case based on either controller's final outcome.
    case=fresh_motion_aware_case(rt['configuration'],rt['settings'],profile,seed,new)
    case=adjust_obstacle_heights(rt['configuration'],case,profile['demo_obstacle_height_offsets_m'])
    row={'seed':seed,'same_preliminary_scene':True,'old_model_sha256':digest(oldpath),
         'new_model_sha256':digest(ROOT/'models/current_ppo.zip')}
    for label,model in [('old_v20',old),('adapted_v21',new)]:
        route=preliminary_route(model,rt['configuration'],rt['settings'],case,rt['controller'])
        points=np.array(route['panel_positions'])
        row[label]={'reason':route['termination_reason'],'steps':route['steps'],
            'min_panel_z_m':float(points[:,2].min()),'max_panel_z_m':float(points[:,2].max()),
            'mean_panel_z_m':float(points[:,2].mean()),'positions':route['panel_positions']}
    return row

from pathlib import Path

def main():
    cases=load_json(RUN/'frozen_test_cases.json')
    chosen=[next(c['scene']['seed'] for c in cases if c['scene']['preliminary_episode']['reason']==reason) for reason in ['success','timeout']]
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        rows=list(pool.map(probe,chosen))
    save(OUT/'paired_generator_probe.json',rows)
    print('Paired generator probe complete',chosen,flush=True)
if __name__=='__main__':main()
