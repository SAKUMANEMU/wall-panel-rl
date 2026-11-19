"""Reproduce offline BC servo calibration from the included teacher arrays."""
import argparse
import json
from pathlib import Path
from run_tracker import HERE
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.waypoint_tracking import WaypointTrackingPolicy
from wall_panel_rl.utils.config import load_config


class OfflineSpaces(gym.Env):
    # Defines IO spaces for offline optimization; no simulated reward is used.
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)
    def reset(self,*,seed=None,options=None):return np.zeros(77,np.float32),{}
    def step(self,action):raise RuntimeError('BC must not step the offline space adapter')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=61)
    parser.add_argument('--output',type=Path,required=True,help='New ZIP path; existing files are refused')
    args=parser.parse_args();torch.set_num_threads(1)
    if args.output.exists():parser.error('Output already exists; choose a new filename')
    robot=WallPanelEnv(load_config(HERE/'config.yaml'))
    try:
        kwargs={'joint_lower':robot.lower.tolist(),'joint_upper':robot.upper.tolist(),
            'joint_increment':robot.max_increment,'joint_margin':.01,'initial_gain':.7,
            'net_arch':dict(pi=[],vf=[128,128])}
    finally:robot.close()
    model=PPO(WaypointTrackingPolicy,OfflineSpaces(),device='cpu',seed=args.seed,n_steps=250,
        batch_size=50,learning_rate=.003,policy_kwargs=kwargs)
    with torch.no_grad():model.policy.log_std.fill_(np.log(.02))
    for name,param in model.policy.named_parameters():param.requires_grad_(name in ('tracking_gain_raw','tracking_bias'))
    frozen={n:p.detach().clone() for n,p in model.policy.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.Adam([p for p in model.policy.parameters() if p.requires_grad],lr=.003)
    with np.load(HERE/'training/teacher_arrays.npz',allow_pickle=False) as a:
        obs=torch.as_tensor(a['obs'].copy());actions=torch.as_tensor(a['actions'].copy());offsets=a['offsets'].copy()
    rng=np.random.default_rng(args.seed)
    for _ in range(2000):
        ep=rng.integers(len(offsets)-1,size=256);lo,hi=offsets[ep],offsets[ep+1]
        index=lo+(rng.random(256)*(hi-lo)).astype(np.int64)
        loss=F.mse_loss(model.policy.tracking_mean(obs[index]),actions[index])
        optimizer.zero_grad();loss.backward();optimizer.step()
    for name,param in model.policy.named_parameters():
        if name in frozen:assert torch.equal(param.detach(),frozen[name])
    args.output.parent.mkdir(parents=True,exist_ok=True);model.save(args.output)
    with torch.no_grad():mse=float(F.mse_loss(model.policy.tracking_mean(obs),actions))
    print(json.dumps({'output':str(args.output),'bc_updates':2000,'ppo_steps':0,'teacher_transitions':len(obs),'mse':mse}))


if __name__=='__main__':main()
