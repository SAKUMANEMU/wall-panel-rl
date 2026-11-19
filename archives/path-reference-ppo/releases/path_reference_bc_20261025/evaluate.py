"""Plan once per packaged scene and compare ordinary versus learned tracking."""
import argparse
import copy
import json
from pathlib import Path
from run_tracker import HERE,PlannedTracker
import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.waypoint_tracking import RouteCursor,reference_observation,bounded_action
from wall_panel_rl.utils.config import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=HERE/'tracker_bc.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,help='Optional smoke-test prefix of the manifest')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output.exists():parser.error('Output exists; choose a new report filename')
    model=PPO.load(args.model,device='cpu');manifest=json.loads((HERE/'evaluation/manifest.json').read_text())
    if args.limit is not None:manifest=manifest[:args.limit]
    rows=[]
    for case in manifest:
        env=WallPanelEnv(load_config(HERE/case['config']))
        try:
            scene=json.loads((HERE/case['scene']).read_text())
            obs,_=env.reset(seed=case['seed'],options={'scenario':scene,'noise_seed':case['seed']+1000000})
            planned=PlannedTracker(env,model,case['seed'])
            state=p.saveState(physicsClientId=env.client)
            attrs={name:copy.deepcopy(getattr(env,name)) for name in ('steps','sim_time','previous_error','joint_path_length','cartesian_path_length','observed_target')}
            noise=copy.deepcopy(env.noise_rng.bit_generator.state)
            for method in ('ordinary','bc'):
                p.restoreState(stateId=state,physicsClientId=env.client)
                for name,value in attrs.items():setattr(env,name,copy.deepcopy(value))
                env.noise_rng.bit_generator.state=copy.deepcopy(noise)
                obs=env.observation();info=env.info()
                if planned.planning_success:
                    cursor=RouteCursor(planned.path,env.config['rrt']['waypoint_tolerance'])
                    while env.steps<env.config['simulation']['max_steps']:
                        q,_=env.joint_state();ref=cursor.reference(q)
                        if method=='ordinary':action=np.clip((ref-q)/env.max_increment,-1,1)
                        else:action=model.predict(reference_observation(obs,ref,env.lower,env.upper),deterministic=True)[0]
                        action=bounded_action(q,action,env.lower,env.upper,env.max_increment,model.policy.joint_margin)
                        obs,_,done,truncated,info=env.step(action)
                        if done or truncated:break
                reason='planning_failure' if not planned.planning_success else 'success' if info['success'] else 'collision' if info['collision'] else 'joint_limit' if info['joint_limit_violation'] else 'timeout'
                rows.append({'seed':case['seed'],'suite':case['suite'],'method':method,'reason':reason,'steps':env.steps,
                    'success':bool(info['success']) and planned.planning_success,'collision':bool(info['collision']),
                    'joint_limit':bool(info['joint_limit_violation']),'planning_seconds':planned.planning_seconds})
            p.removeState(stateUniqueId=state,physicsClientId=env.client)
            print(case['key'],rows[-2]['reason'],rows[-1]['reason'],flush=True)
        finally:env.close()
    metrics={}
    for method in ('ordinary','bc'):
        group=[r for r in rows if r['method']==method and r['suite'] not in ('alignment_far','alignment_near')]
        metrics[method]={'episodes':len(group),'successes':sum(r['success'] for r in group),
            'collisions':sum(r['collision'] for r in group),'joint_limits':sum(r['joint_limit'] for r in group),
            'planning_failures':sum(r['reason']=='planning_failure' for r in group)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'metrics':metrics,'episodes':rows},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics))


if __name__=='__main__':main()
