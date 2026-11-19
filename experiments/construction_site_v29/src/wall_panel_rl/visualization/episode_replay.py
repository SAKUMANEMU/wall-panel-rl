"""Display recorded physical states without planning or stepping the simulator.

Recording samples are actual control-step states. Interpolation between them is
only for display; it does not create new physics or collision-test results.
"""
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv

FORMAT = 'wall-panel-replay-v1'


def capture_frame(env):
    position, orientation = env.panel_pose()
    return {'step': int(env.steps), 'time_s': float(env.sim_time),
            'joints': env.joint_state()[0].tolist(), 'panel': list(position),
            'panel_orientation': list(orientation),
            'obstacle_centers': [list(center) for center, _ in env.obstacles]}


def recording_payload(configuration, case, result, trajectory):
    return {'format': FORMAT, 'configuration': copy.deepcopy(configuration),
            'case': copy.deepcopy(case), 'result': copy.deepcopy(result),
            'trajectory': trajectory,
            'display_interpolation_only': True}


class RecordedEpisode:
    """Validate a self-contained recording before opening a graphics window."""
    def __init__(self, payload):
        if payload.get('format') != FORMAT:
            raise ValueError('Not a complete replay recording; record again with the updated demo')
        self.payload = payload
        self.configuration = payload['configuration']
        self.case = payload['case']
        self.result = payload['result']
        frames = payload['trajectory']
        if not frames:
            raise ValueError('Recording has no physical states')
        n = len(self.case['scene']['initial_q'])
        count = len(self.case['scene']['obstacles'])
        self.times = np.array([f['time_s'] for f in frames], dtype=float)
        self.joints = np.array([f['joints'] for f in frames], dtype=float)
        self.positions = np.array([f['panel'] for f in frames], dtype=float)
        self.orientations = np.array([f['panel_orientation'] for f in frames], dtype=float)
        self.centers = np.array([f['obstacle_centers'] for f in frames], dtype=float).reshape(len(frames), count, 3)
        expected = [(self.times, (len(frames),)), (self.joints, (len(frames), n)),
                    (self.positions, (len(frames), 3)), (self.orientations, (len(frames), 4)),
                    (self.centers, (len(frames), count, 3))]
        if any(a.shape != shape or not np.isfinite(a).all() for a, shape in expected):
            raise ValueError('Invalid recorded state dimensions or nonfinite data')
        if self.times[0] < 0 or np.any(np.diff(self.times) <= 0):
            raise ValueError('Recorded physical times must strictly increase')
        if not np.allclose(np.linalg.norm(self.orientations, axis=1), 1., atol=1e-5):
            raise ValueError('Recorded panel orientations must be unit quaternions')
        self.duration = float(self.times[-1]-self.times[0])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def sample(self, elapsed):
        """Interpolate joints/centers and use shortest-arc panel quaternion SLERP."""
        when = float(np.clip(elapsed, 0, self.duration))+self.times[0]
        right = min(int(np.searchsorted(self.times, when, side='right')), len(self.times)-1)
        left = max(0, right-1)
        alpha = ((when-self.times[left])/(self.times[right]-self.times[left]) if right != left else 0.)
        blend = lambda values: (1-alpha)*values[left]+alpha*values[right]
        a, b = self.orientations[left], self.orientations[right].copy()
        dot = float(np.dot(a, b))
        if dot < 0:
            b = -b
            dot = -dot
        dot = np.clip(dot, -1., 1.)
        if dot > .9995:
            orientation = (1-alpha)*a+alpha*b
        else:
            angle = np.arccos(dot)
            orientation = (np.sin((1-alpha)*angle)*a+np.sin(alpha*angle)*b)/np.sin(angle)
        orientation /= np.linalg.norm(orientation)
        return {'time_s': when, 'joints': blend(self.joints), 'panel': blend(self.positions),
                'panel_orientation': orientation, 'obstacle_centers': blend(self.centers)}


def apply_frame(env, frame):
    """Set visual poses directly. Never call env.step() or p.stepSimulation()."""
    for joint, angle in zip(env.joints, frame['joints']):
        p.resetJointState(env.robot, joint, float(angle), physicsClientId=env.client)
    p.resetBasePositionAndOrientation(env.panel, frame['panel'], frame['panel_orientation'], physicsClientId=env.client)
    p.resetBaseVelocity(env.panel, [0, 0, 0], [0, 0, 0], physicsClientId=env.client)
    for body, center in zip(env.obstacle_ids, frame['obstacle_centers']):
        p.resetBasePositionAndOrientation(body, center, [0, 0, 0, 1], physicsClientId=env.client)


def frame_times(duration, fps, speed):
    if not np.isfinite([duration, fps, speed]).all() or duration < 0 or min(fps, speed) <= 0:
        raise ValueError('Nonnegative duration, positive finite FPS and speed required')
    return np.minimum(np.arange(int(math.ceil(duration*fps/speed))+1)*speed/fps, duration)


def replay_episode(episode, *, fps=60., speed=.6, headless=False,
                   start_hold=2., end_hold=30., real_time=None):
    """Use wall-clock frame deadlines, dropping late frames instead of bursting.

    DIRECT/headless mode validates all poses as fast as possible by default.
    A GUI closes after end_hold, or sooner when its window is closed by the user.
    """
    if not isinstance(episode, RecordedEpisode):
        episode = RecordedEpisode(episode)
    times = frame_times(episode.duration, fps, speed)
    if not np.isfinite([start_hold, end_hold]).all() or min(start_hold, end_hold) < 0:
        raise ValueError('Hold times must be finite and nonnegative')
    real_time = not headless if real_time is None else real_time
    env = WallPanelEnv(copy.deepcopy(episode.configuration), None if headless else 'human')
    presented, skipped, costs = 0, 0, []
    try:
        p.setRealTimeSimulation(0, physicsClientId=env.client)
        if not headless:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=env.client)
        env.reset(seed=int(episode.case['scene']['seed']), options={'scenario': episode.case['scene']})
        apply_frame(env, episode.sample(0))
        if not headless:
            p.resetDebugVisualizerCamera(2.8, -50, -25, [0, 0, .58], physicsClientId=env.client)
            nominal = episode.case['scene'].get('nominal_motion', {}).get('panel_positions', [])
            for a, b in zip(nominal[:-1], nominal[1:]):
                if np.linalg.norm(np.asarray(b)-a) > 1e-6:
                    p.addUserDebugLine(a, b, [.2, .8, .85], 1.5, physicsClientId=env.client)
            for i, motion in enumerate(episode.case['scene'].get('obstacle_motion', [])):
                anchor = motion['anchor']
                p.addUserDebugText(f'O{i+1}: anchor z={anchor[2]:.2f}m',
                                  [anchor[0], anchor[1], anchor[2]+.16], [1, .55, .1],
                                  textSize=1., physicsClientId=env.client)
            reason = episode.result.get('reason', 'unknown')
            label = f'Recorded replay | {fps:g} FPS | {speed:g}x | seed={episode.case["scene"]["seed"]} | {reason}'
            p.addUserDebugText(label, [0, 0, 1.7], [.2, .6, .9], textSize=1., physicsClientId=env.client)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.client)
            _hold(env, start_hold)
        begin = time.perf_counter()
        index, previous = 0, None
        while index < len(times) and p.isConnected(env.client):
            if real_time:
                deadline = begin+index/fps
                remaining = deadline-time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                due = min(int((time.perf_counter()-begin)*fps), len(times)-1)
                if due > index:
                    skipped += due-index
                    index = due
            start = time.perf_counter()
            frame = episode.sample(times[index])
            apply_frame(env, frame)
            if not headless and previous is not None and np.linalg.norm(frame['panel']-previous) > 1e-6:
                p.addUserDebugLine(previous, frame['panel'], [.05, .4, .95], 2, physicsClientId=env.client)
            previous = frame['panel']
            costs.append(time.perf_counter()-start)
            presented += 1
            index += 1
        elapsed = time.perf_counter()-begin
        stats = {'requested_fps': fps, 'playback_speed': speed, 'recorded_duration_s': episode.duration,
                 'target_playback_duration_s': episode.duration/speed,
                 'playback_wall_s': elapsed, 'scheduled_frames': len(times),
                 'presented_frames': presented, 'skipped_frames': skipped,
                 'mean_pose_update_ms': 1000*float(np.mean(costs)) if costs else 0.,
                 'completed': index == len(times), 'physics_steps_during_playback': 0,
                 'controller_calls_during_playback': 0, 'real_time': real_time,
                 'recorded_result': episode.result}
        if not headless and p.isConnected(env.client):
            _hold(env, end_hold)
        return stats
    finally:
        env.close()


def _hold(env, seconds):
    end = time.perf_counter()+seconds
    while p.isConnected(env.client) and time.perf_counter() < end:
        time.sleep(min(.05, max(0, end-time.perf_counter())))
