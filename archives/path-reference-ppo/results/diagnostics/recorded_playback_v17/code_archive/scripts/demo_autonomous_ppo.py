"""Show current PPO, RRT, or an episode-level random mixture.

Default: retained initialization, four static boxes and a fresh random scene.
An obstacle-free reference is prepared before execution (or loaded from a bank);
PPO has no online avoidance planner. Explicit RRT/mixture modes label ownership.
The complete episode is recorded headlessly before60 FPS playback by default.
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


def main(default_controller='ppo', *, default_report=None, manifest_name='manifest.json',motion_profile=None,predictive_selection=None):
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
    parser.add_argument('--playback-mode',choices=['recorded','live'],default='recorded',
                        help='Record headlessly before smooth playback (default), or display during computation')
    parser.add_argument('--fps',type=float,default=60.,help='Target frame rate for recorded playback')
    parser.add_argument('--trajectory-output',type=Path,default=None,help='Self-contained JSON recording of configuration, scene, result and real states')
    choices=['ppo','rrt','mixture']+(['ppo_predictive','mixture_predictive','ppo_progress','mixture_progress'] if predictive_selection else [])
    parser.add_argument('--controller',choices=choices,default=default_controller)
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
    if not np.isfinite([args.hold_seconds,args.start_hold_seconds]).all() or min(args.hold_seconds,args.start_hold_seconds)<0:parser.error('Hold times must be finite and nonnegative')
    if not np.isfinite(args.fps) or args.fps<=0:parser.error('FPS must be finite and positive')
    if not np.isfinite(args.expert_probability) or not 0<=args.expert_probability<=1:parser.error('Invalid episode probability')
    live=not args.headless and args.playback_mode=='live'
    record_states=args.trajectory_output is not None or args.playback_mode=='recorded'
    recording=None
    if args.playback_mode=='recorded':
        print('Stage 1: recording the complete episode headlessly. Prediction may take time; playback starts after completion.',flush=True)
    torch.set_num_threads(1);manifest=read(args.manifest)
    config=copy.deepcopy(manifest['configuration'])
    settings=manifest.get('evaluation_settings',manifest.get('settings'))
    if settings is None:parser.error('Manifest must contain evaluation_settings or settings')
    if profile:config,settings=dynamic_configuration(config,settings,profile)
    model=PPO.load(args.model,device='cpu')
    def adjust_scene(case):
        if profile and profile.get('demo_obstacle_height_offsets_m') is not None:
            from wall_panel_rl.rl.demo_scene_adjustments import adjust_obstacle_heights
            return adjust_obstacle_heights(config,case,profile['demo_obstacle_height_offsets_m'])
        return case
    if args.scene_bank is None:
        if profile and profile.get('motion_aware_placement'):
            from wall_panel_rl.rl.motion_aware_placement import fresh_motion_aware_case
            rejects=[];scene_rng=np.random.default_rng()
            for _ in range(1 if args.seed is not None else profile['placement_attempts']):
                current=args.seed if args.seed is not None else int(scene_rng.integers(2000000,2000000000))
                try:
                    case=adjust_scene(fresh_motion_aware_case(config,settings,profile,current,model));break
                except (RuntimeError,ValueError) as exc:
                    rejects.append({'seed':current,'reason':str(exc)})
                    if args.seed is not None:raise
            else:raise RuntimeError(f'Could not construct a valid dynamic layout: {rejects}')
        elif profile:case,rejects=fresh_dynamic_case(config,settings,profile,args.seed,model=model)
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
        case=adjust_scene(case)
    args.seed=int(case['scene']['seed'])
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):parser.error('Expected the current 77-input / 7-action model')
    factory=SlowDynamicFourEnv if profile else (ConditionalRecoveryEnv if 'conditional_recovery' in settings else StagedAvoidanceEnv)
    env=factory(config,[],settings)
    if live:
        # Replace only the rendering connection, retaining the exact evaluated
        # reset/step/cursor/projection implementation in StagedAvoidanceEnv.
        env.env.close();env.env=WallPanelEnv(config,'human')
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=env.env.client)
    total=0.;previous=None
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        controller=(EpisodeChoice(args.expert_probability,args.policy_seed).begin_episode()
                    if args.controller in ('mixture','mixture_predictive','mixture_progress') else args.controller)
        if args.controller=='mixture_predictive' and controller=='ppo':controller='ppo_predictive'
        if args.controller=='mixture_progress' and controller=='ppo':controller='ppo_progress'
        if profile:before=planning_snapshot(env)
        expert=make_rrt(env,args.seed+100000) if controller=='rrt' else None
        safety=None
        if controller in ('ppo_predictive','ppo_progress'):
            from wall_panel_rl.rl.predictive_safety import PredictiveSafetyController
            selection=read(predictive_selection)
            gains=selection.get('gains')
            if gains is None:gains=read(Path(predictive_selection).parent/'protocol.json')['spec']['gains']
            if controller=='ppo_progress':
                from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
                if 'recovery' not in selection:parser.error('Progress recovery requires a controller config with recovery settings')
                safety=ProgressRecoveryController(model,env,selection['selected']['horizon'],gains,selection['recovery'])
            else:safety=PredictiveSafetyController(model,env,selection['selected']['horizon'],gains)
        if profile:assert_planning_paused(env,before)
        print(f'Episode controller: {controller} | seed={args.seed} | obstacles={len(env.env.obstacles)} | noise={config["noise"]}',flush=True)
        print(f'Model: {args.model.resolve()} | playback={args.playback_speed:g}x | online avoidance controller={bool(expert or safety)}',flush=True)
        if safety:print(f'PPO + predictive execution safety | horizon={safety.horizon} | weights unchanged',flush=True)
        if controller=='ppo_progress':print('Progress-aware local recovery enabled; simulation pauses during predictive computation',flush=True)
        print(f'Execution wrapper: {factory.__name__}',flush=True)
        if profile:
            print('Slow 3D motion:',json.dumps(profile['motion']),'| known motion | planning pauses simulation time',flush=True)
            if case['scene'].get('demo_height_offsets_m'):
                print('Demo anchor height offsets O1/O2/O3/O4 (m):',case['scene']['demo_height_offsets_m'],flush=True)
        if expert is not None and not expert.planning_success:
            print(json.dumps({'seed':args.seed,'reason':'planning_failure','controller':'rrt',
                'steps':0,'online_avoidance_planner':True,'fallback_used':False}),flush=True)
            return
        if live:
            p.resetDebugVisualizerCamera(2.8,-50,-25,[0,0,.58],physicsClientId=env.env.client)
            nominal=case['scene'].get('nominal_motion',{}).get('panel_positions',[])
            for a,b in zip(nominal[:-1],nominal[1:]):
                p.addUserDebugLine(a,b,[.2,.8,.85],1.5,physicsClientId=env.env.client)
            if profile and profile.get('height_stagger'):
                for i,motion in enumerate(case['scene'].get('obstacle_motion',[])):
                    anchor=motion['anchor']
                    label=[anchor[0],anchor[1],anchor[2]+.16]
                    p.addUserDebugText(f'O{i+1}: anchor z={anchor[2]:.2f}m',label,
                                      [1,.55,.1],textSize=1.,physicsClientId=env.env.client)
                p.addUserDebugText('Cyan: placement route | Blue: executed panel path',
                                  [0,0,1.7],[.2,.6,.9],textSize=1.,physicsClientId=env.env.client)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1,physicsClientId=env.env.client)
            time.sleep(args.start_hold_seconds)
        trajectory=[]
        def record_actual():
            if record_states:
                from wall_panel_rl.visualization.episode_replay import capture_frame
                trajectory.append(capture_frame(env.env))
        record_actual()
        while True:
            start=time.perf_counter()
            if profile:before=planning_snapshot(env)
            action=expert.action(obs) if expert else (safety.action(obs) if safety else model.predict(obs,deterministic=True)[0])
            if profile:assert_planning_paused(env,before)
            obs,reward,done,truncated,info=env.step(action);total+=reward
            record_actual()
            position=list(env.env.panel_pose()[0])
            if live and previous is not None:
                p.addUserDebugLine(previous,position,[.05,.4,.95],2,physicsClientId=env.env.client)
            previous=position
            if env.env.steps%10==0 or done or truncated:
                print(f'step={env.env.steps} position={info["position_error"]:.4f}m orientation={np.rad2deg(info["orientation_error"]):.2f}deg reason={info["termination_reason"]}',flush=True)
            if live:
                duration=config['simulation']['timestep']*config['simulation']['substeps']/args.playback_speed
                time.sleep(max(0.,duration-(time.perf_counter()-start)))
            if done or truncated:break
        result={'seed':args.seed,'reason':info['termination_reason'],'steps':env.env.steps,
                'obstacle_count':len(env.env.obstacle_ids),'obstacle_motion':'dynamic' if env.env.obstacle_motion else 'static',
                'scene_source':'frozen_bank' if args.scene_bank else 'fresh_random_scene',
                'all_obstacles_on_nominal_path':case['scene'].get('on_path_placement',{}).get('all_obstacles_intersect_nominal'),
                'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                'reward':total,'online_avoidance_planner':bool(expert or safety),'model':str(args.model.resolve()) if controller.startswith('ppo') else None,
                'controller':controller,'planning_seconds':expert.planning_time if expert else (safety.seconds if safety else 0.),
                'execution_wrapper':factory.__name__,
                'conditional_recovery_attempts':int(info.get('conditional_recovery_attempts',0)),
                'conditional_recoveries':int(info.get('conditional_recovery_count',0))}
        if profile:
            result.update(motion_bounds=profile['motion'],planning_pauses_time_checked=True,
                dynamic_planning_mode=config['rrt']['dynamic_mode'],
                motion_anchors_on_nominal_path=case['scene'].get('motion_anchor_placement',{}).get('all_obstacles_intersect_nominal'),
                replans=expert.replans if expert else 0,scheduled_waits=expert.scheduled_waits if expert else 0,
                physics_elapsed_s=env.env.sim_time,training_performed=False)
            if case['scene'].get('demo_height_offsets_m'):
                result['demo_height_offsets_m']=case['scene']['demo_height_offsets_m']
        if safety:result.update(prediction_horizon=safety.horizon,predictive_interventions=safety.interventions,
            discarded_prediction_steps=safety.prediction_steps,uses_global_rrt=False,weights_changed=False)
        if controller=='ppo_progress':result.update(recovery_actions=safety.recovery_actions,local_rescues=safety.rescues,
            baseline_success_protected=safety.protected,forecast_cache_hits=safety.forecast_cache_hits)
        if record_states:
            from wall_panel_rl.visualization.episode_replay import recording_payload
            recording=recording_payload(config,case,result,trajectory)
            output=args.trajectory_output or ROOT/'results/replays'/f'episode_{args.seed}_{time.time_ns()}.json'
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(recording,indent=2)+'\n')
            print(f'Recorded episode saved: {output.resolve()}',flush=True)
        print(json.dumps(result,indent=2),flush=True)
        if live:time.sleep(args.hold_seconds)
    finally:env.close()
    if args.playback_mode=='recorded' and not args.headless and recording is not None:
        from wall_panel_rl.visualization.episode_replay import replay_episode
        print(f'Stage 2: recorded playback at {args.fps:g} FPS / {args.playback_speed:g}x; no planning or inference during playback.',flush=True)
        stats=replay_episode(recording,fps=args.fps,speed=args.playback_speed,
                             start_hold=args.start_hold_seconds,end_hold=args.hold_seconds)
        print('Playback:',json.dumps({k:v for k,v in stats.items() if k!='recorded_result'}),flush=True)


if __name__=='__main__':main()
