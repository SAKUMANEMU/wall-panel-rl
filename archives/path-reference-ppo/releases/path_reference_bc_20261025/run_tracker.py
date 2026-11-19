"""Run the released BC servo behind live static IK/RRT planning."""
import argparse
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'src'))

import numpy as np
import torch
from stable_baselines3 import PPO
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.waypoint_tracking import RouteCursor,reference_observation,bounded_action
from wall_panel_rl.utils.config import load_config


class PlannedTracker:
    def __init__(self,env,model,seed):
        if env.obstacle_motion:raise ValueError('Only static obstacles were validated for this release')
        if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):
            raise ValueError('Use the released 77-observation waypoint tracker')
        if not (np.allclose(model.policy.joint_lower,env.lower) and np.allclose(model.policy.joint_upper,env.upper)
                and np.isclose(model.policy.joint_increment,env.max_increment)):
            raise ValueError('Robot limits/control increment do not match the checkpoint')
        self.env=env;self.model=model;planner=ClassicalController(env,seed)
        self.planning_success=bool(planner.planning_success);self.planning_seconds=float(planner.planning_time)
        self.path=planner.path
        self.cursor=RouteCursor(self.path,env.config['rrt']['waypoint_tolerance']) if self.planning_success else None

    def action(self,observation):
        if not self.planning_success:raise RuntimeError('Planning failed; no tracking action is executed')
        env=self.env;q,_=env.joint_state();ref=self.cursor.reference(q)
        x=reference_observation(observation,ref,env.lower,env.upper)
        action=self.model.predict(x,deterministic=True)[0]
        return bounded_action(q,action,env.lower,env.upper,env.max_increment,self.model.policy.joint_margin)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=HERE/'tracker_bc.zip')
    parser.add_argument('--config',type=Path,default=HERE/'config.yaml')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--scene',type=Path)
    group.add_argument('--random',action='store_true',help='Generate a new random-five static layout')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args();torch.set_num_threads(1)
    scene_path=None if args.random else args.scene or HERE/'scenes/original_validation_81009.json'
    scene=json.loads(scene_path.read_text()) if scene_path else None
    seed=args.seed if args.seed is not None else scene['seed'] if scene else 81009
    model=PPO.load(args.model,device='cpu');config=load_config(args.config);env=WallPanelEnv(config)
    try:
        options={'noise_seed':seed+1000000}
        if scene is not None:options['scenario']=scene
        obs,info=env.reset(seed=seed,options=options)
        controller=PlannedTracker(env,model,seed)
        if controller.planning_success:
            for _ in range(config['simulation']['max_steps']):
                obs,_,done,truncated,info=env.step(controller.action(obs))
                if done or truncated:break
        reason='planning_failure' if not controller.planning_success else 'success' if info['success'] else 'collision' if info['collision'] else 'joint_limit' if info['joint_limit_violation'] else 'timeout'
        print(json.dumps({'method':'RRT + BC waypoint tracking','seed':seed,'reason':reason,'steps':env.steps,
            'planning_seconds':controller.planning_seconds,'position_error_m':float(info['position_error']),
            'orientation_error_deg':float(np.rad2deg(info['orientation_error']))}))
    finally:env.close()


if __name__=='__main__':main()
