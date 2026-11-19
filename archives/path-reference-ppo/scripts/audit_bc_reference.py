"""Supplementary inference-only BC reference/terminal-action diagnostics."""
import _bootstrap
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import pybullet as p
from stable_baselines3 import PPO
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv
from wall_panel_rl.rl.autonomous_dynamic_ppo import ReferenceCursor
from wall_panel_rl.rl.waypoint_tracking import WaypointTrackingPolicy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);torch.set_num_threads(1)
    if not (out/'audit.json').exists():raise ValueError('Run audit_staged_bc.py first')
    artifact=out/'reference_details.json'
    if artifact.exists():raise FileExistsError(artifact)
    folder=Path('models/staged_avoidance_v2_seed61/one_static')
    meta=json.loads((folder/'bc/metadata.json').read_text());s=meta['settings']
    data=Path(s['data']);config=json.loads((data/'stage_metadata.json').read_text())['environment_config']
    bc=PPO.load(folder/'bc/model.zip',device='cpu');source=PPO.load(s['source_model'],device='cpu')
    lower=np.asarray(bc.policy.joint_lower);span=np.asarray(bc.policy.joint_upper)-lower
    with np.load(data/'teacher/teacher.npz',allow_pickle=False) as d:
        obs=d['obs'].copy();offsets=d['offsets'].copy();actions=d['actions'].copy();tasks=d['tasks'].copy()
    attempts=[a for a in json.loads((data/'teacher/attempts.json').read_text()) if a['reason']=='success']
    episodes=[]
    for i,a in enumerate(attempts):
        segment=obs[offsets[i]:offsets[i+1]];labels=actions[offsets[i]:offsets[i+1]]
        q=lower+(segment[:,:7]+1)*.5*span
        cursor=ReferenceCursor(a['scene']['reference_joint_path'])
        indices=[]
        for state in q:cursor.reference(state);indices.append(cursor.index)
        endpoint=np.asarray(a['scene']['reference_joint_path'][-1])
        with torch.no_grad():prior=WaypointTrackingPolicy.tracking_mean(bc.policy,torch.as_tensor(segment)).numpy()
        active=np.abs(labels)>.05
        episodes.append({'episode':i,'task':str(tasks[i]),'seed':a['seed'],
            'last_reference_fraction':indices[-1]/(len(cursor.path)-1),
            'preterminal_q_distance_to_reference_goal_rad':float(np.linalg.norm(q[-1]-endpoint)),
            'preterminal_target_position_m':float(np.linalg.norm(segment[-1,14:17])),
            'last_reference_distance_to_goal_rad':float(np.linalg.norm(cursor.path[cursor.index]-endpoint)),
            'prior_vs_teacher_opposite_sign_fraction':float(np.mean(prior[active]*labels[active]<0))})
    summary={}
    for task in ('alignment','avoidance'):
        group=[v for v in episodes if v['task']==task]
        summary[task]={'episodes':len(group),
            **{key:float(np.mean([v[key] for v in group])) for key in
            ('last_reference_fraction','preterminal_q_distance_to_reference_goal_rad',
             'preterminal_target_position_m','last_reference_distance_to_goal_rad',
             'prior_vs_teacher_opposite_sign_fraction')},
            'reference_not_reached_90percent':sum(v['last_reference_fraction']<.9 for v in group)}
    env=StagedAvoidanceEnv(config,[],s);terminal=[];traces=[]
    try:
        cases=[c for c in json.loads((data/'validation_staged.json').read_text())
            if c['suite'].startswith('alignment')]
        for case in cases:
            scene=copy.deepcopy(case['scene']);goal=scene['reference_joint_path'][-1]
            scene['initial_q']=goal;scene['reference_joint_path']=[goal,goal]
            observation,info=env.reset(options={'scenario':scene,'task':'alignment','strength':case['strength']})
            terminal.append({'seed':scene['seed'],'suite':case['suite'],'initial_success':bool(info['success']),
                'source_action':source.predict(observation,deterministic=True)[0].tolist(),
                'bc_action':bc.predict(observation,deterministic=True)[0].tolist()})
        # Trace one far and one near case without changing any checkpoint.
        for suite in ('alignment_far','alignment_near'):
            case=next(v for v in cases if v['suite']==suite)
            observation,_=env.reset(options={'scenario':case['scene'],'task':'alignment','strength':case['strength']})
            trace=[]
            while True:
                with torch.no_grad():
                    x=torch.as_tensor(observation[None]);gate=float(torch.sigmoid(bc.policy.prior_gate(x)).item())
                action=bc.predict(observation,deterministic=True)[0]
                observation,_,done,truncated,info=env.step(action)
                trace.append({'step':env.env.steps,'position_m':float(info['position_error']),
                    'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                    'action':action.tolist(),'gate':gate,'cursor':env.cursor.index,
                    'reason':info['termination_reason']})
                if done or truncated:break
            e=env.env;contacts=[]
            for body_name,body in [('arm',e.robot),('panel',e.panel)]:
                for other_name,other in [('wall',e.wall),*[(f'obstacle_{i}',b) for i,b in enumerate(e.obstacle_ids)]]:
                    for contact in p.getClosestPoints(body,other,.001,physicsClientId=e.client):
                        contacts.append({'body':body_name,'other':other_name,
                            'link':contact[3],'distance_m':contact[8]})
            traces.append({'suite':suite,'seed':case['scene']['seed'],'trace':trace,'terminal_contacts':contacts})
    finally:env.close()
    result={'training_performed':False,'reference_summary':summary,'reference_episodes':episodes,
        'at_reference_goal_actions':terminal,'bc_closed_loop_traces':traces}
    artifact.write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
    print('Terminal source/BC max action',
        max(np.max(np.abs(v['source_action'])) for v in terminal),
        max(np.max(np.abs(v['bc_action'])) for v in terminal))
    print('Trace terminals',[(v['suite'],v['trace'][-1],v['terminal_contacts']) for v in traces])


if __name__=='__main__':main()
