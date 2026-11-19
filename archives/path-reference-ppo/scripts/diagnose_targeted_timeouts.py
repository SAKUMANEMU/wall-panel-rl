"""Read-only policy diagnosis: replay known timeouts with a longer diagnostic horizon.

No fitting, no expert, no checkpoint selection. Original 240-step results remain
unchanged; extended runs are explicitly separate, already-seen-case diagnostics.
"""
import _bootstrap
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print('Check only. --run replays the 27 known timeouts; no training.');return
    torch.set_num_threads(1)
    root=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/targeted_bc_v3_seed61')
    output=root/'timeout_diagnosis.json'
    if output.exists():raise FileExistsError(output)
    def read(path):return json.loads(Path(path).read_text())
    manifest=read(root/'manifest.json');config=copy.deepcopy(manifest['configuration'])
    assert config['simulation']['max_steps']==240
    config['simulation']['max_steps']=480
    settings=read(root/'targeted_mix.json')['settings']
    original={r['seed']:r for r in read(root/'independent_targeted_mix.json')['rows']
              if r['suite']=='stage_target' and r['reason']=='timeout'}
    cases=[c for c in read(Path(manifest['study']['data_root'])/'independent_cases.json')
           if c['suite']=='stage_target' and c['scene']['seed'] in original]
    path=root/'selected_bc_model.zip';before=hashlib.sha256(path.read_bytes()).hexdigest()
    model=PPO.load(path,device='cpu');env=StagedAvoidanceEnv(config,[],settings);rows=[]
    try:
        for case in cases:
            obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            history=[];snapshots=[]
            for step in range(1,481):
                action=model.predict(obs,deterministic=True)[0]
                obs,_,done,truncated,info=env.step(action)
                q=env.env.joint_state()[0]
                history.append({'step':step,'q':q.copy(),'position_m':float(info['position_error']),
                    'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                    'action_norm':float(np.linalg.norm(action)),'reference_index':env.cursor.index})
                if step%20==0 or done or truncated:
                    snapshots.append({k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in history[-1].items()})
                if step==240:
                    error=abs(info['position_error']-original[case['scene']['seed']]['position_m'])
                    if error>1e-6:raise AssertionError(f'Original prefix not reproduced: {error}')
                if done or truncated:break
            if len(history)<240:raise AssertionError('Previously timed-out prefix changed')
            window=history[199:240];travel=sum(float(np.linalg.norm(b['q']-a['q'])) for a,b in zip(window,window[1:]))
            displacement=float(np.linalg.norm(window[-1]['q']-window[0]['q']))
            gain=window[0]['position_m']-window[-1]['position_m']
            if travel<.01:kind='nearly_stationary'
            elif displacement<.25*travel:kind='oscillating_or_looping'
            elif gain>.001:kind='approaching_position'
            else:kind='moving_without_position_progress'
            rows.append({'seed':case['scene']['seed'],'reason_at_480_or_termination':info['termination_reason'],
                'final_step':env.env.steps,'late_reason':info['termination_reason'],
                'position_m_240':history[239]['position_m'],'orientation_deg_240':history[239]['orientation_deg'],
                'position_m_final':float(info['position_error']),'orientation_deg_final':float(np.rad2deg(info['orientation_error'])),
                'last40_joint_travel_rad':travel,'last40_joint_displacement_rad':displacement,
                'last40_position_progress_m':gain,'classification_240':kind,
                'last40_reference_index_change':window[-1]['reference_index']-window[0]['reference_index'],
                'snapshots':snapshots})
            print('timeout',len(rows),'/',len(cases),'seed',case['scene']['seed'],kind,info['termination_reason'],env.env.steps,flush=True)
    finally:env.close()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    counts={}
    for horizon in (240,320,480):
        counts[horizon]=dict(Counter(r['late_reason'] if r['final_step']<=horizon else 'timeout' for r in rows))
    report={'training_performed':False,'original_model_unchanged':True,'model_sha256':before,
        'known_timeout_cases_only':True,'independent_test_used_for_tuning_now':True,
        'original_240_step_benchmark_unchanged':True,'conditional_horizon_counts':counts,
        'classification_counts':dict(Counter(r['classification_240'] for r in rows)),
        'classification_rule':'steps 200-240: stationary if total joint travel < .01 rad; looping if displacement/travel < .25; approaching if position improves > .001 m; otherwise moving without positional progress',
        'rows':rows}
    output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
