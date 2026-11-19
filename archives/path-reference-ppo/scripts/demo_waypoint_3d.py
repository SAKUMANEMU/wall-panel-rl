"""Render a real PyBullet RRT+PPO rollout to MP4, optionally with a live GUI.

Only model inference is performed. Visual trails have no collision shapes.
"""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
import torch
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.waypoint_tracking import RouteCursor, reference_observation, bounded_action
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.utils.config import ROOT, load_config

WIDTH, HEIGHT, FPS = 1280, 800, 30


def main():
    release=ROOT/'releases/path_reference_bc_20261025'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=ROOT/'models/waypoint_ppo_pilot_seed61/best_model.zip')
    parser.add_argument('--scene',type=Path,default=release/'scenes/original_validation_81009.json')
    parser.add_argument('--config',type=Path,default=release/'config.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'results/demos/waypoint_ppo_3d.mp4')
    parser.add_argument('--playback-speed',type=float,default=.6)
    parser.add_argument('--gui',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.playback_speed) or args.playback_speed<=0:
        parser.error('Playback speed must be positive and finite')
    if args.output.exists():parser.error('Output exists; choose a new filename')
    torch.set_num_threads(1)
    config=load_config(args.config)
    if config['obstacles'].get('dynamic',{}).get('enabled',False):
        parser.error('This demo supports the validated static-obstacle controller')
    scene=json.loads(args.scene.read_text());seed=scene['seed']
    model=PPO.load(args.model,device='cpu')
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,):parser.error('A 77-input, 7-output waypoint model is required')
    caption=f'Model: {args.model.name}'
    selection=args.model.parent/'selection.json'
    if args.model.name=='best_model.zip' and selection.exists():
        added=json.loads(selection.read_text())['additional_steps']
        caption=f'Selected model: +{added:,} PPO transitions after BC initialization'
    env=WallPanelEnv(config,'human' if args.gui else 'rgb_array')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    writer=None
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    regular=ImageFont.truetype(font_path,19)
    title=ImageFont.truetype(font_path,24)
    small=ImageFont.truetype(font_path,16)
    error_log=tempfile.TemporaryFile()
    try:
        if args.gui:p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=env.client)
        obs,info=env.reset(seed=seed,options={'scenario':scene,'noise_seed':seed+1000000})
        initial_error=float(info['position_error'])
        planner=ClassicalController(env,seed)
        cursor=RouteCursor(planner.path,config['rrt']['waypoint_tolerance']) if planner.planning_success else None
        path_positions=[]
        if planner.planning_success:
            for q in planner.path:
                with env.collision.at_configuration(q):
                    path_positions.append(np.array(env.panel_pose()[0]))
        def dot(position,color,radius):
            shape=p.createVisualShape(p.GEOM_SPHERE,radius=radius,rgbaColor=color,physicsClientId=env.client)
            p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,baseVisualShapeIndex=shape,
                              basePosition=position,physicsClientId=env.client)
        for position in path_positions[::2]:dot(position,[.90,.66,.10,1],.0045)
        ground=p.createVisualShape(p.GEOM_BOX,halfExtents=[1.7,1.7,.002],
                                 rgbaColor=[.93,.95,.97,1],physicsClientId=env.client)
        p.createMultiBody(baseMass=0,baseCollisionShapeIndex=-1,baseVisualShapeIndex=ground,
                         basePosition=[0,0,-.009],physicsClientId=env.client)
        if args.gui:p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1,physicsClientId=env.client)
        writer=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo',
            '-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}','-r',str(FPS),'-i','-',
            '-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(args.output)],stdin=subprocess.PIPE,stderr=error_log)
        reason='planning_failure' if not planner.planning_success else 'ready'
        frames=0; trajectory=[list(env.panel_pose()[0])]
        def frame(phase,yaw=-50):
            view=p.computeViewMatrixFromYawPitchRoll([0,0,.58],2.8,yaw,-25,0,2)
            projection=p.computeProjectionMatrixFOV(52,WIDTH/620,.04,7)
            raw=p.getCameraImage(WIDTH,620,view,projection,renderer=p.ER_TINY_RENDERER,
                                physicsClientId=env.client,shadow=1)[2]
            canvas=Image.new('RGB',(WIDTH,HEIGHT),'#f7f9fc')
            canvas.paste(Image.fromarray(np.asarray(raw,dtype=np.uint8)[:,:,:3]),(0,100))
            draw=ImageDraw.Draw(canvas)
            draw.rectangle((0,0,WIDTH,99),fill='#13243a')
            draw.text((28,13),'3D SIMULATION  |  RRT + PPO waypoint tracking',font=title,fill='white')
            draw.text((28,53),f'{len(env.obstacles)} STATIC obstacles  |  seed {seed}  |  motion playback {args.playback_speed:g}x',font=regular,fill='#c5d8ee')
            draw.text((28,77),caption,font=small,fill='#a8bed8')
            draw.rectangle((0,720,WIDTH,HEIGHT),fill='#13243a')
            draw.text((28,731),f'Step {env.steps:03d}   |   simulation {env.sim_time:.2f}s   |   position {info["position_error"]*1000:.2f} mm   |   orientation {np.rad2deg(info["orientation_error"]):.2f} deg',font=regular,fill='white')
            draw.text((28,764),'BLUE: executed panel path     YELLOW: RRT route     RED: obstacles     GREEN: target',font=small,fill='#bed2e8')
            color='#247a49' if phase=='SUCCESS' else '#a13a35' if phase=='FAILED' else '#264f80'
            draw.rounded_rectangle((WIDTH-285,115,WIDTH-24,158),radius=8,fill=color)
            draw.text((WIDTH-269,124),phase,font=regular,fill='white')
            return canvas
        def emit(canvas,count):
            nonlocal frames
            data=np.asarray(canvas,dtype=np.uint8).tobytes()
            for _ in range(count):
                writer.stdin.write(data);frames+=1
                if args.gui:time.sleep(1/FPS)
        emit(frame('READY - paused'),FPS*2)
        motion_frames=0;poster_saved=False
        if cursor:
            for _ in range(config['simulation']['max_steps']):
                q,_=env.joint_state();ref=cursor.reference(q)
                x=reference_observation(obs,ref,env.lower,env.upper)
                action=model.predict(x,deterministic=True)[0]
                action=bounded_action(q,action,env.lower,env.upper,env.max_increment,model.policy.joint_margin)
                obs,_,done,truncated,info=env.step(action)
                position=list(env.panel_pose()[0]);trajectory.append(position)
                dot(position,[.10,.43,.87,1],.006)
                current=frame('RUNNING')
                if env.steps==20:
                    current.save(args.output.with_suffix('.png'));poster_saved=True
                expected=round(env.steps*config['simulation']['substeps']*config['simulation']['timestep']/args.playback_speed*FPS)
                emit(current,max(0,expected-motion_frames));motion_frames=expected
                if done or truncated:break
            reason='success' if info['success'] else 'collision' if info['collision'] else 'joint_limit' if info['joint_limit_violation'] else 'timeout'
        final=frame('SUCCESS' if reason=='success' else 'FAILED')
        if not poster_saved:final.save(args.output.with_suffix('.png'))
        final.save(args.output.with_name(args.output.stem+'_final.png'))
        emit(final,FPS*2)
        # Orbit the paused final physical state; simulation does not advance.
        for yaw in np.linspace(-50,130,FPS*3):
            emit(frame('SUCCESS' if reason=='success' else 'FAILED',float(yaw)),1)
        writer.stdin.close()
        code=writer.wait()
        if code:
            error_log.seek(0);raise RuntimeError(error_log.read().decode())
        result={'method':'RRT + PPO waypoint tracking','seed':seed,'reason':reason,
            'steps':env.steps,'simulated_seconds':env.sim_time,'playback_speed':args.playback_speed,
            'video_seconds':frames/FPS,'width':WIDTH,'height':HEIGHT,'fps':FPS,
            'initial_position_error_m':initial_error,'position_error_m':float(info['position_error']),
            'orientation_error_deg':float(np.rad2deg(info['orientation_error'])),
            'planning_seconds':planner.planning_time,'obstacle_count':len(env.obstacles),
            'obstacles_dynamic':False,'noise':config['noise'],'model':str(args.model.resolve()),
            'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'scene':str(args.scene.resolve()),'panel_positions':trajectory,
            'final_orbit_simulation_paused':True}
        args.output.with_suffix('.json').write_text(json.dumps(result,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k!='panel_positions'},indent=2))
    finally:
        if writer and writer.poll() is None:
            if writer.stdin and not writer.stdin.closed:writer.stdin.close()
            writer.wait()
        error_log.close();env.close()


if __name__=='__main__':main()
