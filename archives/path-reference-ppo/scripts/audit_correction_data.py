"""Replay stored prefixes + labels from completed corrective data, no expert/fit."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data/bc_budget_v3_seed61/expanded/teacher')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();data=Path(args.data);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    meta=json.loads((data/'metadata.json').read_text())
    if not meta['completed']:raise ValueError('Demonstrations not complete')
    before=hashlib.sha256((data/'teacher.npz').read_bytes()).hexdigest()
    if before!=meta['teacher_sha256']:raise ValueError('Demonstration hash changed')
    with np.load(data/'teacher.npz',allow_pickle=False) as z:
        obs=z['obs'].copy();actions=z['actions'].copy();offsets=z['offsets'].copy()
    successful=[a for a in json.loads((data/'attempts.json').read_text()) if a['reason']=='success']
    if len(successful)!=len(offsets)-1:raise ValueError('Attempt/label episode mismatch')
    indices=[i for i,a in enumerate(successful) if a['group']=='correction']
    if not indices:raise ValueError('No corrections')
    chosen=[indices[i] for i in np.linspace(0,len(indices)-1,min(10,len(indices)),dtype=int)]
    env=StagedAvoidanceEnv(meta['environment_config'],[],meta['settings']);rows=[]
    try:
        for i in chosen:
            a=successful[i]
            x,_=env.reset(options={'scenario':a['scene'],'task':meta['settings']['active_stage']['mode'],
                'strength':[meta['settings']['active_stage']['amplitude'],meta['settings']['active_stage']['speed']]})
            for command in a['warmup_actions']:
                x,_,done,truncated,info=env.step(command)
                if done or truncated:raise RuntimeError('Prefix replay terminated unexpectedly')
            maximum=0.;steps=0
            for j in range(offsets[i],offsets[i+1]):
                maximum=max(maximum,float(np.max(np.abs(x-obs[j]))))
                x,_,done,truncated,info=env.step(actions[j]);steps+=1
                if done or truncated:break
            rows.append({'episode':i,'seed':a['seed'],'prefix_steps':len(a['warmup_actions']),
                'labelled_steps':steps,'reason':info['termination_reason'],
                'max_observation_error':maximum,'total_steps':env.env.steps,'original_total_steps':a['steps']})
    finally:env.close()
    assert hashlib.sha256((data/'teacher.npz').read_bytes()).hexdigest()==before
    report={'training_performed':False,'teacher_sha256':before,'sampled_corrections':len(rows),
        'success':sum(r['reason']=='success' for r in rows),'max_observation_error':max(r['max_observation_error'] for r in rows),
        'prefixes_not_labels':True,'rows':rows}
    out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
