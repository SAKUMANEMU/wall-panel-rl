"""Inference-only budget diagnosis on existing demonstrations and BC weights."""
import os
from pathlib import Path
import json
import hashlib
import sys
import numpy as np
import torch
import pybullet as p
from stable_baselines3 import PPO

work=Path('/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo')
sys.path.insert(0,str(work/'src'))
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv

torch.set_num_threads(1)
def forbidden(*args,**kwargs):raise RuntimeError('Learning is forbidden in budget analysis')
PPO.learn=forbidden;torch.optim.Adam.step=forbidden
out=Path(__file__).resolve().parent
base=work/'models/staged_avoidance_v3_seed61/one_static'
data=work/'data/staged_avoidance_v3_seed61/one_static/teacher'
meta=json.loads((base/'bc/metadata.json').read_text());s=meta['settings']
config=json.loads((data/'metadata.json').read_text())['environment_config']
model_path=base/'bc/model.zip';initial_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
model=PPO.load(model_path,device='cpu')
with np.load(data/'teacher.npz',allow_pickle=False) as d:
    obs=d['obs'].copy();actions=d['actions'].copy();offsets=d['offsets'].copy()
    seeds=d['seeds'].copy();tasks=d['tasks'].copy()
attempts=[a for a in json.loads((data/'attempts.json').read_text()) if a['reason']=='success']
groups={}
for split in ('train','heldout'):
    allowed=set(meta[split+'_scene_seeds'])
    episodes=[i for i in range(len(seeds)) if int(seeds[i]) in allowed and tasks[i]=='avoidance']
    indices=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in episodes])
    with torch.no_grad():
        x=torch.as_tensor(obs[indices]);mean=model.policy.tracking_mean(x).numpy()
        strength,gate,residual=model.policy.correction_components(x)
    delta=mean-actions[indices]
    groups[split]={'episodes':episodes,'scene_count':len(set(int(seeds[i]) for i in episodes)),
        'transitions':len(indices),'mse':float(np.mean(delta**2)),
        'rmse_action':float(np.sqrt(np.mean(delta**2))),
        'mae_action':float(np.mean(np.abs(delta))),
        'gate_mean':float(gate.mean()),'residual_mean_abs':float(residual.abs().mean())}

def contacts(e):
    result=[];margin=e.config['collision']['margin']
    for name,body in [('arm',e.robot),('panel',e.panel)]:
        for other_name,other in [('wall',e.wall),*[(f'obstacle_{i}',b) for i,b in enumerate(e.obstacle_ids)]]:
            for c in p.getClosestPoints(body,other,margin,physicsClientId=e.client):
                result.append({'body':name,'other':other_name,'link':c[3],'distance':c[8]})
    for c in p.getClosestPoints(e.robot,e.robot,margin,physicsClientId=e.client):
        if abs(c[3]-c[4])>1:result.append({'body':'arm','other':'self','links':list(c[3:5]),'distance':c[8]})
    for c in p.getClosestPoints(e.robot,e.panel,margin,physicsClientId=e.client):
        if c[3]!=e.ee:result.append({'body':'arm','other':'panel','link':c[3],'distance':c[8]})
    return result

env=StagedAvoidanceEnv(config,[],s);rows=[]
try:
    for split in ('train','heldout'):
        # Distinct seeds, not multiple phases of one layout.
        selected=[];seen=set()
        for i in groups[split]['episodes']:
            if int(seeds[i]) in seen:continue
            seen.add(int(seeds[i]));selected.append(i)
            if len(selected)==10:break
        for i in selected:
            case=attempts[i]
            for method in ('stored_expert_actions','bc_closed_loop'):
                x,_=env.reset(options={'scenario':case['scene'],'task':'static','strength':[.05,.02]})
                step=0;state_error=0.
                while True:
                    j=offsets[i]+step
                    if method=='stored_expert_actions':
                        if j>=offsets[i+1]:break
                        state_error=max(state_error,float(np.max(np.abs(x-obs[j]))))
                        a=actions[j]
                    else:a=model.predict(x,deterministic=True)[0]
                    x,_,done,truncated,info=env.step(a);step+=1
                    if done or truncated:break
                rows.append({'split':split,'episode':i,'seed':int(seeds[i]),'method':method,
                    'reason':info['termination_reason'],'steps':step,
                    'position_error_m':float(info['position_error']),
                    'contacts':contacts(env.env) if info['collision'] else [],
                    'replay_observation_max_error':state_error if method=='stored_expert_actions' else None})
finally:env.close()
summary=[]
for split in ('train','heldout'):
    for method in ('stored_expert_actions','bc_closed_loop'):
        group=[r for r in rows if r['split']==split and r['method']==method]
        summary.append({'split':split,'method':method,'episodes':len(group),
            **{reason:sum(r['reason']==reason for r in group) for reason in ('success','collision','timeout','joint_limit')}})
assert hashlib.sha256(model_path.read_bytes()).hexdigest()==initial_hash
report={'training_performed':False,'model_sha256':initial_hash,'teacher_forced_avoidance':groups,
    'closed_loop_on_collected_scenes':summary,'episodes':rows}
(out/'budget_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ('teacher_forced_avoidance','closed_loop_on_collected_scenes')},indent=2))
print('COLLISION_CONTACTS',json.dumps([r['contacts'] for r in rows if r['method']=='bc_closed_loop']))
