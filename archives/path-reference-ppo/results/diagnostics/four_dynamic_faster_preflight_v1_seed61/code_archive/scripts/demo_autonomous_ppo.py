"""Show current PPO, RRT, or an episode-level random mixture.

Default: retained initialization, four static boxes and a fresh random scene.
An obstacle-free reference is prepared before execution (or loaded from a bank);
PPO has no online avoidance planner. Explicit RRT/mixture modes label ownership.
"""
import _bootstrap
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.episode_policy import EpisodeChoice
from wall_panel_rl.rl.episode_imitation import make_rrt


def read(path):return json.loads(Path(path).read_text())


def main(default_controller='ppo', *, default_report=None, manifest_name='manifest.json',motion_profile=None):
    parser=argparse.ArgumentParser(description=__doc__)
    report=Path(default_report) if default_report is not None else ROOT/'results/diagnostics/four_static_on_path_v1_seed61'
    parser.add_argument('--model',type=Path,default=None if motion_profile else report/'selected_model.zip')
    parser.add_argument('--manifest',type=Path,default=None if motion_profile else report/manifest_name)
    if motion_profile:
        parser.add_argument('--motion-config',type=Path,default=motion_profile,help='Slow dynamic motion bounds and v5 source report')
    parser.add_argument('--scene-bank',type=Path,default=None,help='Optional frozen bank; otherwise generate a fresh static scene')
    parser.add_argument('--seed',type=int,default=None,help='Reproduce a scene; otherwise choose a fresh random seed')
    parser.add_argument('--playback-speed',type=float,default=.6)
    parser.add_argument('--hold-seconds',type=float,default=30)
    parser.add_argument('--start-hold-seconds',type=float,default=2)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--controller',choices=['ppo','rrt','mixture'],default=default_controller)
    parser.add_argument('--expert-probability',type=float,default=.5)
    parser.add_argument('--policy-seed',type=int,default=None)
    args=parser.parse_args()
    profile=None
    if motion_profile:
        from wall_panel_rl.rl.slow_dynamic_four import (load_profile,dynamic_configuration,fresh_dynamic_case,
            add_motion,validate_placement,SlowDynamicFourEnv,planning_snapshot,assert_planning_paused)
        profile=load_profile(args.motion_config);report=ROOT/profile['source_report']
        args.model=args.model or report/'selected_model.zip'
        args.manifest=args.manifest or report/manifest_name
    if not np.isfinite(args.playback_speed) or args.playback_speed<=0:parser.error('Playback speed must be positive')
    if min(args.hold_seconds,args.start_hold_seconds)<0:parser.error('Hold times must be nonnegative')
    if not np.isfinite(args.expert_probability) or not 0<=args.expert_probability<=1:parser.error('Invalid episode probability')
    torch.set_num_threads(1);manifest=read(args.manifest)
    config=copy.deepcopy(manifest['configuration'])
    settings=manifest.get('evaluation_settings',manifest.get('settings'))
    if settings is None:parser.error('Manifest must contain evaluation_settings or settings')
    if profile:config,settings=dynamic_configuration(config,settings,profile)
    model=PPO.load(args.model,device='cpu')
    if args.scene_bank is None:
        if profile:case,rejects=fresh_dynamic_case(config,settings,profile,args.seed,model=model)
        else:
            from wall_panel_rl.rl.static_demo_scene import fresh_static_case
            case,rejects=fresh_static_case(config,settings,args.seed,model=model)
        if rejects:print('Scene placement retries:',json.dumps(rejects),flush=True)
    else:
        cases=[c for c in read(args.scene_bank) if c['suite']=='stage_target']
        if args.seed is None:
            if not cases:parser.error('Scene bank contains no stage_target cases')
            case=cases[int(np.random.default_rng().integers(len(cases)))]
        else:
            cases=[c for c in cases if c['scene']['seed']==args.seed]
            if len(cases)!=1:parser.error('Seed must identify exactly one stage_target case in the selected scene bank')
            case=cases[0]
        if profile:
            if not case['scene'].get('obstacle_motion'):case=add_motion(case,profile)
            if not np.allclose(case['strength'],[profile['motion']['amplitude_m'],profile['motion']['max_speed_m_s']]):
                parser.error('Frozen scene motion strength differs from --motion-config')
            validate_placement(config,case)
    args.seed=int(case['scene']['seed'])
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):parser.error('Expected the current 77-input / 7-action model')
    factory=SlowDynamicFourEnv if profile else (ConditionalRecoveryEnv if 'conditional_recovery' in settings else StagedAvoidanceEnv)
    env=factory(config,[],settings)
    if not args.headless:
        # Replace only the rendering connection, retaining the exact evaluated
        # reset/step/cursor/projection implementation in StagedAvoidanceEnv.
        env.env.close();env.env=WallPanelEnv(config,'human')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=env.env.client)
    total=0.;previous=None
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        controller=(EpisodeChoice(args.expert_probability,args.policy_seed).begin_episode()
                    if args.controller=='mixture' else args.controller)
        if profile:before=planning_snapshot(env)
        expert=make_rrt(env,args.seed+100000) if controller=='rrt' else None
        if profile:assert_planning_paused(env,before)
        print(f'Episode controller: {controller} | seed={args.seed} | obstacles={len(env.env.obstacles)} | noise={config["noise"]}',flush=True)
        print(f'Model: {args.model.resolve()} | playback={args.playback_speed:g}x | online avoidance planner={controller=="rrt"}',flush=True)
        print(f'Execution wrapper: {factory.__name__}',flush=True)
        if profile:
            print('Slow 3D motion:',json.dumps(profile['motion']),'| known motion | planning pauses simulation time',flush=True)
        if expert is not None and not expert.planning_success:
            print(json.dumps({'seed':args.seed,'reason':'planning_failure','controller':'rrt',
                'steps':0,'online_avoidance_planner':True,'fallback_used':False}),flush=True)
            return
        if not args.headless:
            p.resetDebugVisualizerCamera(2.8,-50,-25,[0,0,.58],physicsClientId=env.env.client)
            nominal=case['scene'].get('nominal_motion',{}).get('panel_positions',[])
            for a,b in zip(nominal[:-1],nominal[1:]):
                p.addUserDebugLine(a,b,[.2,.8,.85],1.5,physicsClientId=env.env.client)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1,physicsClientId=env.env.client)
            time.sleep(args.start_hold_seconds)
        while True:
            start=time.perf_counter()
            if profile:before=planning_snapshot(env)
            action=expert.action(obs) if expert else model.predict(obs,deterministic=True)[0]
            if profile:assert_planning_paused(env,before)
            obs,reward,done,truncated,info=env.step(action);total+=reward
            position=list(env.env.panel_pose()[0])
            if not args.headless and previous is not None:
                p.addUserDebugLine(previous,position,[.05,.4,.95],2,physicsClientId=env.env.client)
            previous=position
            if env.env.steps%10==0 or done or truncated:
                print(f'step={env.env.steps} position={info["position_error"]:.4f}m orientation={np.rad2deg(info["orientation_error"]):.2f}deg reason={info["termination_reason"]}',flush=True)
            if not args.headless:
                duration=config['simulation']['timestep']*config['simulation']['substeps']/args.playback_speed
                time.sleep(max(0.,duration-(time.perf_counter()-start)))
            if done or truncated:break
        result={'seed':args.seed,'reason':info['termination_reason'],'steps':env.env.steps,
                'obstacle_count':len(env.env.obstacle_ids),'obstacle_motion':'dynamic' if env.env.obstacle_motion else 'static',
                'scene_source':'frozen_bank' if args.scene_bank else 'fresh_random_scene',
                'all_obstacles_on_nominal_path':case['scene'].get('on_path_placement',{}).get('all_obstacles_intersect_nominal'),
                'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                'reward':total,'online_avoidance_planner':controller=='rrt','model':str(args.model.resolve()) if controller=='ppo' else None,
                'controller':controller,'planning_seconds':expert.planning_time if expert else 0.,
                'execution_wrapper':factory.__name__,
                'conditional_recovery_attempts':int(info.get('conditional_recovery_attempts',0)),
                'conditional_recoveries':int(info.get('conditional_recovery_count',0))}
        if profile:
            result.update(motion_bounds=profile['motion'],planning_pauses_time_checked=True,
                dynamic_planning_mode=config['rrt']['dynamic_mode'],
                motion_anchors_on_nominal_path=case['scene'].get('motion_anchor_placement',{}).get('all_obstacles_intersect_nominal'),
                replans=expert.replans if expert else 0,scheduled_waits=expert.scheduled_waits if expert else 0,
                physics_elapsed_s=env.env.sim_time,training_performed=False)
        print(json.dumps(result,indent=2),flush=True)
        if not args.headless:time.sleep(args.hold_seconds)
    finally:env.close()


if __name__=='__main__':main()
