"""Inference-only checks of v3 using existing v2 scenes and weights in memory.

Does not collect demonstrations, update parameters, or produce a trained model.
The old BC transplant is an ablation, not a valid v3 training initialization.
"""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from staged_avoidance import resolve,check
from wall_panel_rl.rl.avoidance_curriculum import initialize_gated,StagedAvoidanceEnv,RecoveringReferenceCursor
from wall_panel_rl.rl.autonomous_dynamic_ppo import ReferenceCursor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    def forbidden(*a,**k):raise AssertionError('Training forbidden in reference-fix check')
    PPO.learn=forbidden;torch.optim.Adam.step=forbidden
    settings=resolve(yaml.safe_load(Path('configs/staged_avoidance.yaml').read_text()),'one_static');check(settings)
    old=Path('data/staged_avoidance_v2_seed61/one_static')
    bc_path=Path('models/staged_avoidance_v2_seed61/one_static/bc/model.zip')
    config=json.loads((old/'stage_metadata.json').read_text())['environment_config']
    cases=[c for c in json.loads((old/'validation_staged.json').read_text())
           if c['suite']!='five_full_diagnostic']
    model=initialize_gated(settings['source_model'],settings)
    env=StagedAvoidanceEnv(config,[],settings);rows=[];probes=[]
    initial_hash={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [bc_path,Path(settings['source_model'])]}
    try:
        for variant in ('v3_untrained_initial','v3_with_old_bc_weights_in_memory'):
            if variant.endswith('in_memory'):
                old_bc=PPO.load(bc_path,device='cpu')
                model.policy.load_state_dict(old_bc.policy.state_dict())
            for case in cases:
                obs,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
                while True:
                    action=model.predict(obs,deterministic=True)[0]
                    obs,_,done,truncated,info=env.step(action)
                    if done or truncated:break
                rows.append({'variant':variant,'suite':case['suite'],'seed':case['scene']['seed'],
                    'reason':info['termination_reason'],'steps':env.env.steps,
                    'position_m':float(info['position_error']),
                    'orientation_deg':float(np.rad2deg(info['orientation_error']))})
            # Probe the fixed goal with an obstacle present to exercise taper,
            # rather than relying only on the no-obstacle correction mask.
            scene=dict(cases[0]['scene']);goal=config['robot']['goal_q']
            scene.update(initial_q=goal,reference_joint_path=[goal,goal],obstacle_motion=[],
                         obstacles=[{'center':[5.,5.,5.],'dimensions':[.1,.1,.1]}])
            obs,info=env.reset(options={'scenario':scene,'task':'static','strength':[.05,.02]})
            probes.append({'variant':variant,'initial_success':bool(info['success']),
                'goal_max_abs_action':float(np.max(np.abs(model.predict(obs,deterministic=True)[0])))})
    finally:env.close()
    with np.load(old/'teacher/teacher.npz',allow_pickle=False) as d:
        observations=d['obs'].copy();offsets=d['offsets'].copy()
    attempts=[a for a in json.loads((old/'teacher/attempts.json').read_text()) if a['reason']=='success']
    lower=np.asarray(model.policy.joint_lower);span=np.asarray(model.policy.joint_upper)-lower
    cursor_rows=[]
    for i,a in enumerate(attempts):
        cursors=[ReferenceCursor(a['scene']['reference_joint_path']),
                 RecoveringReferenceCursor(a['scene']['reference_joint_path'],**settings['reference_recovery'])]
        for cursor in cursors:
            for observation in observations[offsets[i]:offsets[i+1]]:
                q=lower+(observation[:7]+1)*.5*span;cursor.reference(q)
        cursor_rows.append({'episode':i,'old_fraction':cursors[0].index/(len(cursors[0].path)-1),
            'new_fraction':cursors[1].index/(len(cursors[1].path)-1),
            'recoveries':cursors[1].recoveries})
    summary=[]
    for variant in sorted({r['variant'] for r in rows}):
        for suite in sorted({r['suite'] for r in rows}):
            group=[r for r in rows if r['variant']==variant and r['suite']==suite]
            summary.append({'variant':variant,'suite':suite,'episodes':len(group),
                **{reason:sum(r['reason']==reason for r in group) for reason in ('success','collision','timeout','joint_limit')}})
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in initial_hash.items())
    result={'training_performed':False,'transplant_saved_as_model':False,'input_hashes':initial_hash,
        'rollout_summary':summary,'rollouts':rows,'goal_probes':probes,'cursor_comparison':cursor_rows,
        'cursor_under_90percent':{name:sum(r[name+'_fraction']<.9 for r in cursor_rows) for name in ('old','new')}}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('rollout_summary','goal_probes','cursor_under_90percent')},indent=2))


if __name__=='__main__':main()
