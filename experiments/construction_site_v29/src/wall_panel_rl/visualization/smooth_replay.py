"""Interactive recorded-pose playback with lower IPC cost and smooth pacing.

No policy, planner or simulation steps are run. On a slow display, smooth mode
slows wall-clock playback rather than dropping poses or rapidly catching up.
The original replay module stays unchanged for experiment reproducibility.
"""
import copy,math,time
import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.visualization.episode_replay import RecordedEpisode,_hold
from wall_panel_rl.envs.construction_site import update_site_visuals


def playback_times(duration,fps,speed):
    if not np.isfinite([duration,fps,speed]).all() or duration<0 or min(fps,speed)<=0:
        raise ValueError('Nonnegative duration and positive finite FPS/speed required')
    count=duration*fps/speed;nearest=round(count)
    steps=int(nearest if abs(count-nearest)<1e-8 else math.ceil(count))
    result=np.minimum(np.arange(steps+1)*speed/fps,duration)
    result[-1]=duration
    return result


def prepare_frames(episode,fps,speed):
    times=playback_times(episode.duration,fps,speed)
    prepared=[]
    for when in times:
        frame=episode.sample(when)
        prepared.append({'joints':[[float(x)] for x in frame['joints']],
            'panel':frame['panel'].tolist(),'orientation':frame['panel_orientation'].tolist(),
            'centers':frame['obstacle_centers'].tolist()})
    return times,prepared


def apply_visual_frame(env,frame):
    # One IPC call for all seven joints rather than seven individual calls.
    p.resetJointStatesMultiDof(env.robot,env.joints,frame['joints'],
        [[0.] for _ in env.joints],physicsClientId=env.client)
    p.resetBasePositionAndOrientation(env.panel,frame['panel'],frame['orientation'],physicsClientId=env.client)
    for body,center in zip(env.obstacle_ids,frame['centers']):
        p.resetBasePositionAndOrientation(body,center,[0,0,0,1],physicsClientId=env.client)
    if env.config.get('construction_site',{}).get('enabled') and env.render_mode=='human':
        # Recorded obstacle centers are the display truth; no motion is rerun.
        for i,center in enumerate(frame['centers']):env.obstacles[i]=(np.asarray(center),env.obstacles[i][1])
        update_site_visuals(env)


def choose_fps(requested,cost_seconds):
    supported=1/max(cost_seconds*1.5,1e-6)
    for candidate in [requested,30.,24.,20.,15.,10.]:
        if candidate<=requested and candidate<=supported:return float(candidate)
    return float(min(requested,max(5.,math.floor(supported))))


def replay_episode(episode,*,fps=60.,speed=.6,headless=False,start_hold=2.,end_hold=30.,
                   real_time=None,pacing='smooth',adaptive_fps=True,show_path=False,shadows=False):
    if not isinstance(episode,RecordedEpisode):episode=RecordedEpisode(episode)
    times,frames=prepare_frames(episode,fps,speed)
    if not np.isfinite([start_hold,end_hold]).all() or min(start_hold,end_hold)<0:
        raise ValueError('Hold times must be finite and nonnegative')
    if pacing not in ('smooth','realtime'):raise ValueError('Unknown playback timing mode')
    real_time=not headless if real_time is None else real_time
    env=WallPanelEnv(copy.deepcopy(episode.configuration),None if headless else 'human')
    effective=fps;costs=[];presented=skipped=0;starts=[];warm=[]
    try:
        p.setRealTimeSimulation(0,physicsClientId=env.client)
        if not headless:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0,physicsClientId=env.client)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,0,physicsClientId=env.client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,int(shadows),physicsClientId=env.client)
        env.reset(seed=int(episode.case['scene']['seed']),options={'scenario':episode.case['scene']})
        p.resetBaseVelocity(env.panel,[0,0,0],[0,0,0],physicsClientId=env.client)
        apply_visual_frame(env,frames[0])
        if not headless:
            camera=episode.configuration.get('demo',{}) if episode.configuration.get('construction_site',{}).get('enabled') else {}
            p.resetDebugVisualizerCamera(camera.get('camera_distance',2.8),camera.get('camera_yaw',-50),
                camera.get('camera_pitch',-25),camera.get('camera_target',[0,0,.58]),physicsClientId=env.client)
            # Static paths are optional. Never allocate a growing trail per frame.
            if show_path:
                measured=episode.positions
                for a,b in zip(measured[:-1],measured[1:]):
                    if np.linalg.norm(b-a)>1e-6:p.addUserDebugLine(a,b,[.05,.4,.95],2,physicsClientId=env.client)
            for i,motion in enumerate(episode.case['scene'].get('obstacle_motion',[])):
                anchor=motion['anchor']
                p.addUserDebugText(f'O{i+1}',[anchor[0],anchor[1],anchor[2]+.16],
                    [1,.55,.1],textSize=1.,physicsClientId=env.client)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1,physicsClientId=env.client)
            if adaptive_fps:
                for _ in range(12):
                    begin=time.perf_counter();apply_visual_frame(env,frames[0]);warm.append(time.perf_counter()-begin)
                effective=choose_fps(fps,float(np.percentile(warm,95)))
                if effective!=fps:times,frames=prepare_frames(episode,effective,speed)
            _hold(env,start_hold)
        begin=time.perf_counter();next_deadline=begin;index=0
        while index<len(times) and p.isConnected(env.client):
            if real_time:
                deadline=next_deadline if pacing=='smooth' else begin+times[index]/speed
                remaining=deadline-time.perf_counter()
                if remaining>0:time.sleep(remaining)
                if pacing=='realtime':
                    due=min(int(np.searchsorted(times/speed,time.perf_counter()-begin,side='right'))-1,len(times)-1)
                    if due>index:skipped+=due-index;index=due
            start=time.perf_counter();starts.append(start)
            apply_visual_frame(env,frames[index]);costs.append(time.perf_counter()-start)
            presented+=1
            if real_time and pacing=='smooth' and index+1<len(times):
                # Rebase after lateness: no skipped samples and no catch-up burst.
                interval=(times[index+1]-times[index])/speed
                next_deadline=max(next_deadline,start)+interval
            index+=1
        elapsed=time.perf_counter()-begin
        spacings=np.diff(starts)
        stats={'requested_fps':fps,'effective_fps':effective,'timing_mode':pacing,
            'adaptive_fps':adaptive_fps,'playback_speed':speed,'recorded_duration_s':episode.duration,
            'target_playback_duration_s':episode.duration/speed,'playback_wall_s':elapsed,
            'actual_playback_speed':episode.duration/elapsed if elapsed>0 else 0.,
            'scheduled_frames':len(times),'presented_frames':presented,'skipped_frames':skipped,
            'mean_pose_update_ms':float(np.mean(costs))*1000 if costs else 0.,
            'p95_pose_update_ms':float(np.percentile(costs,95))*1000 if costs else 0.,
            'p95_frame_interval_ms':float(np.percentile(spacings,95))*1000 if len(spacings) else 0.,
            'max_frame_interval_ms':float(np.max(spacings))*1000 if len(spacings) else 0.,
            'completed':index==len(times),'physics_steps_during_playback':0,'controller_calls_during_playback':0,
            'poses_precomputed':True,'batched_joint_update':True,'real_time':real_time,
            'recorded_result':episode.result}
        if not headless and p.isConnected(env.client):_hold(env,end_hold)
        return stats
    finally:env.close()
