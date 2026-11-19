"""Replay preserves measured states/results and cannot invoke physical control."""
import copy
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.visualization import episode_replay as replay


@pytest.fixture
def payload():
    config = load_config()
    env = WallPanelEnv(config)
    try:
        scene = {'seed': 7, 'initial_q': env.goal_q.tolist(),
                 'target_position': env.target_pose[0].tolist(),
                 'obstacles': [{'center': [-2., 0., 1.], 'dimensions': [.1, .1, .1]}],
                 'obstacle_motion': []}
        env.reset(seed=7, options={'scenario': scene})
        first = replay.capture_frame(env)
        q = env.goal_q.copy()
        q[0] += .03
        env.set_configuration(q)
        env.steps = 1
        env.sim_time = .05
        last = replay.capture_frame(env)
        last['obstacle_centers'] = [[-1.9, .1, 1.1]]
        return replay.recording_payload(config, {'scene': scene},
                                        {'reason': 'timeout', 'steps': 1}, [first, last])
    finally:
        env.close()


def test_measured_sample_endpoints_and_antipodal_quaternions(payload):
    # q and -q represent the same orientation; avoid a spurious full rotation.
    payload['trajectory'][1]['panel_orientation'] = [-x for x in payload['trajectory'][0]['panel_orientation']]
    episode = replay.RecordedEpisode(payload)
    midpoint = episode.sample(.025)
    assert np.allclose(midpoint['joints'], (episode.joints[0]+episode.joints[-1])/2)
    assert abs(np.dot(midpoint['panel_orientation'], episode.orientations[0])) == pytest.approx(1.)
    for when, index in [(0, 0), (.05, -1)]:
        state = episode.sample(when)
        assert np.allclose(state['joints'], episode.joints[index])
        assert np.allclose(state['panel'], episode.positions[index])
    assert np.isfinite(midpoint['obstacle_centers']).all()


def test_incomplete_or_nonmonotonic_recording_is_rejected(payload):
    old = copy.deepcopy(payload)
    del old['format']
    with pytest.raises(ValueError, match='complete replay'):
        replay.RecordedEpisode(old)
    payload['trajectory'][-1]['time_s'] = 0
    with pytest.raises(ValueError, match='strictly increase'):
        replay.RecordedEpisode(payload)


def test_display_does_not_step_physics_and_preserves_failure_result(payload, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Replay must not execute any physical steps')
    monkeypatch.setattr(WallPanelEnv, 'step', forbidden)
    monkeypatch.setattr(p, 'stepSimulation', forbidden)
    actual_apply = replay.apply_frame
    measured = []
    def verify_pose(env, frame):
        before = (env.steps, env.sim_time)
        actual_apply(env, frame)
        assert (env.steps, env.sim_time) == before
        assert np.allclose(env.joint_state()[0], frame['joints'], atol=1e-12)
        pos, quat = env.panel_pose()
        assert np.allclose(pos, frame['panel'], atol=1e-12)
        assert abs(np.dot(quat, frame['panel_orientation'])) == pytest.approx(1.)
        for body, center in zip(env.obstacle_ids, frame['obstacle_centers']):
            actual = p.getBasePositionAndOrientation(body, physicsClientId=env.client)[0]
            assert np.allclose(actual, center, atol=1e-12)
        measured.append(frame)
    monkeypatch.setattr(replay, 'apply_frame', verify_pose)
    stats = replay.replay_episode(payload, headless=True, fps=60, speed=.6)
    assert stats['completed'] and stats['physics_steps_during_playback'] == 0
    assert stats['controller_calls_during_playback'] == 0
    assert stats['recorded_result']['reason'] == 'timeout'
    assert stats['scheduled_frames'] == stats['presented_frames'] == 6
    assert np.allclose(measured[-1]['joints'], payload['trajectory'][-1]['joints'])


def test_frame_deadlines_have_fixed_spacing_and_reach_exact_endpoint():
    times = replay.frame_times(1., fps=60, speed=.6)
    assert len(times) == 101
    assert np.allclose(np.diff(times)/.6, 1/60)
    assert times[-1] == 1.
    with pytest.raises(ValueError):
        replay.frame_times(1., 0, .6)


def test_late_frames_are_skipped_without_bursting(payload, monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(replay.time, 'perf_counter', lambda: clock.now)
    monkeypatch.setattr(replay.time, 'sleep', lambda t: setattr(clock, 'now', clock.now+t))
    actual_apply = replay.apply_frame
    def slow_display(env, frame):
        actual_apply(env, frame)
        clock.now += .04  # A slow renderer cannot keep up with 60 FPS.
    monkeypatch.setattr(replay, 'apply_frame', slow_display)
    stats = replay.replay_episode(payload, headless=True, real_time=True, fps=60, speed=.6)
    assert stats['skipped_frames'] > 0
    assert stats['completed']
    assert stats['presented_frames']+stats['skipped_frames'] == stats['scheduled_frames']


def test_smooth_batch_update_preserves_recording_without_physics(payload,monkeypatch):
    from wall_panel_rl.visualization import smooth_replay as smooth
    def forbidden(*args,**kwargs):raise AssertionError('Playback must not simulate')
    monkeypatch.setattr(WallPanelEnv,'step',forbidden)
    monkeypatch.setattr(p,'stepSimulation',forbidden)
    original=smooth.apply_visual_frame;observed=[]
    def check(env,frame):
        before=(env.steps,env.sim_time);original(env,frame)
        assert before==(env.steps,env.sim_time)
        assert np.allclose(env.joint_state()[0],np.asarray(frame['joints']).ravel(),atol=1e-12)
        assert np.allclose(env.panel_pose()[0],frame['panel'],atol=1e-12)
        observed.append(frame)
    monkeypatch.setattr(smooth,'apply_visual_frame',check)
    stats=smooth.replay_episode(payload,headless=True)
    assert stats['completed'] and stats['skipped_frames']==0
    assert stats['batched_joint_update'] and stats['poses_precomputed']
    assert stats['physics_steps_during_playback']==stats['controller_calls_during_playback']==0
    assert stats['recorded_result']==payload['result']
    assert np.allclose(np.asarray(observed[-1]['joints']).ravel(),payload['trajectory'][-1]['joints'])


def test_smooth_slow_display_keeps_frames_without_catchup_burst(payload,monkeypatch):
    from wall_panel_rl.visualization import smooth_replay as smooth
    clock=SimpleNamespace(now=0.)
    monkeypatch.setattr(smooth.time,'perf_counter',lambda:clock.now)
    monkeypatch.setattr(smooth.time,'sleep',lambda t:setattr(clock,'now',clock.now+t))
    original=smooth.apply_visual_frame
    def slow(env,frame):original(env,frame);clock.now+=.04
    monkeypatch.setattr(smooth,'apply_visual_frame',slow)
    stats=smooth.replay_episode(payload,headless=True,real_time=True,pacing='smooth')
    assert stats['presented_frames']==stats['scheduled_frames']==6
    assert stats['skipped_frames']==0 and stats['completed']
    assert stats['playback_wall_s']>stats['target_playback_duration_s']
    assert stats['p95_frame_interval_ms']>=39.9


def test_fractional_duration_does_not_add_duplicate_endpoint():
    from wall_panel_rl.visualization.smooth_replay import playback_times
    values=playback_times(3.7500000000000813,60,.6)
    assert len(values)==376 and np.all(np.diff(values)>1e-9)
    assert values[-1]==3.7500000000000813
