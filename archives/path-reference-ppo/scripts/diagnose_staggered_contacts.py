"""Replay unresolved TRAIN collisions and capture contacts at the physics substep."""
import _bootstrap
import copy
import numpy as np
import pybullet as pb
import torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv
from run_targeted_bc import read,dump


def snapshot(e):
    kw={'physicsClientId':e.client};margin=e.config['collision']['margin'];rows=[]
    targets=[('obstacle_'+str(i),b) for i,b in enumerate(e.obstacle_ids)]+[('wall',e.wall)]
    queries=[(name,body,other,body2) for name,body in [('arm',e.robot),('panel',e.panel)] for other,body2 in targets]
    queries += [('arm',e.robot,'arm',e.robot),('arm',e.robot,'panel',e.panel)]
    for name,a,other,b in queries:
        hits=([c for c in pb.getContactPoints(a,b,**kw) if c[8]<=0] if margin==0 else pb.getClosestPoints(a,b,margin,**kw))
        for c in hits:
            if a==b and abs(c[3]-c[4])<=1:continue
            if a==e.robot and b==e.panel and c[3]==e.ee:continue
            rows.append({'body':name,'other':other,'link':c[3],'other_link':c[4],'distance_m':float(c[8])})
    return rows


def main():
    torch.set_num_threads(1)
    spec=yaml.safe_load((ROOT/'configs/four_staggered_dynamic_preparation.yaml').read_text());report=ROOT/spec['warmup']['report_root']
    previous=read(report/'training_diagnosis.json');unresolved=[r for r in previous if r['reason']=='collision' and not r['collision_pairs']]
    cases={c['scene']['seed']:c for c in read(ROOT/spec['data_root']/'train_cases.json')}
    prep=read(ROOT/spec['preparation_report']/'protocol.json');model=PPO.load(ROOT/spec['source_model'],device='cpu');rows=[]
    for original in unresolved:
        case=cases[original['seed']];env=HybridFourEnv(copy.deepcopy(prep['configuration']),[],copy.deepcopy(prep['evaluation_settings']));e=env.env;capture=[]
        check=e.collision.is_collision
        def recorded(*args,**kwargs):
            hit=check(*args,**kwargs)
            if hit and kwargs.get('use_contacts'):capture.extend(snapshot(e))
            return hit
        e.collision.is_collision=recorded
        try:
            obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
            while True:
                obs,_,done,truncated,info=env.step(model.predict(obs,deterministic=True)[0])
                if done or truncated:break
            assert info['termination_reason']==original['reason'] and e.steps==original['steps'],'Source replay differed'
            rows.append({'seed':original['seed'],'reason':info['termination_reason'],'steps':e.steps,'execution_substep_contacts':capture})
        finally:env.close()
    dump(report/'execution_contact_replay.json',{'train_only':True,'optimizer_updates':0,'source_outcomes_reproduced':True,'rows':rows,
        'resolved':sum(bool(r['execution_substep_contacts']) for r in rows),'remaining_unresolved':sum(not r['execution_substep_contacts'] for r in rows)})
    print({'replayed':len(rows),'resolved':sum(bool(r['execution_substep_contacts']) for r in rows)})

if __name__=='__main__':main()
