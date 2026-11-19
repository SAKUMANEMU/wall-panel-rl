"""Motion bounds, replay consistency, preserved interfaces and timed-planner routing."""
import copy
import json
import numpy as np
import pytest
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.slow_dynamic_four import (load_profile,dynamic_configuration,add_motion,
    SlowDynamicFourEnv,planning_snapshot,assert_planning_paused)


@pytest.fixture
def setup():
    spec=load_profile(ROOT/'configs/four_slow_dynamic.yaml')
    manifest=json.loads((ROOT/spec['source_report']/'candidate_manifest.json').read_text())
    config,settings=dynamic_configuration(manifest['configuration'],manifest['evaluation_settings'],spec)
    static=json.loads((ROOT/'data/four_expanded_v5_seed61/validation_cases.json').read_text())[0]
    return spec,manifest,config,settings,static


def test_profile_preserves_robot_rewards_and_interfaces_without_mutating_source(setup):
    spec,source,config,settings,_=setup
    assert source['evaluation_settings']['active_stage']['mode']=='static'
    for key in ('robot','panel','target','controller','collision','simulation','reward','observation'):
        assert config[key]==source['configuration'][key]
    assert config['noise']=={'position_std':0.,'orientation_std_deg':0.}
    assert config['rrt']['dynamic_mode']=='timed' and not config['obstacles']['freeze_motion']
    assert settings['conditional_recovery']==source['evaluation_settings']['conditional_recovery']
    assert settings['active_stage']['count']==4 and spec['expert_probability']==.5


def test_random_phase_motion_is_reproducible_three_dimensional_and_norm_bounded(setup):
    spec,_,_,_,static=setup;before=copy.deepcopy(static)
    case=add_motion(static,spec);assert case==add_motion(static,spec) and static==before
    altered=copy.deepcopy(static);altered['scene']['seed']+=1
    assert case['scene']['obstacle_motion']!=add_motion(altered,spec)['scene']['obstacle_motion']
    assert case['task']=='dynamic' and case['strength']==[.05,.02]
    for key in ('initial_q','reference_joint_path','target_position','target_orientation'):
        assert case['scene'][key]==static['scene'][key]
    for old,box,motion in zip(static['scene']['obstacles'],case['scene']['obstacles'],case['scene']['obstacle_motion']):
        assert box['dimensions']==old['dimensions'] and motion['anchor']==old['center']
        extent=np.array(motion['extent']);omega=np.array(motion['omega']);phase=np.array(motion['phase'])
        t=np.linspace(0,100,1001)[:,None];delta=extent*np.sin(t*omega+phase);velocity=extent*omega*np.cos(t*omega+phase)
        assert np.all(np.ptp(delta,axis=0)>.01)
        assert np.linalg.norm(delta,axis=1).max()<=.05+1e-12
        assert np.linalg.norm(velocity,axis=1).max()<=.02+1e-12
        assert np.allclose(box['center'],np.array(motion['anchor'])+delta[0])


def test_dynamic_reset_and_query_restore_physics_clock_reference_and_velocity_slots(setup,monkeypatch):
    from wall_panel_rl.rl.episode_imitation import make_rrt
    from wall_panel_rl.planning.timed_rrt import TimedRRT
    spec,_,config,settings,static=setup;case=add_motion(static,spec)
    env=SlowDynamicFourEnv(config,[],settings)
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        assert obs.shape==(77,) and env.action_space.shape==(7,)
        slots=obs[20:70].reshape(5,10)
        assert np.all(slots[:4,9]==1) and np.all(slots[4]==0)
        assert np.allclose(slots[:4,6:9],[env.env.obstacle_velocity(i) for i in range(4)])
        before=planning_snapshot(env)
        q=env.env.joint_state()[0];TimedRRT(env.env).edge_valid(q,q,.5)
        assert_planning_paused(env,before)
        def no_swept_planning():raise AssertionError('Dynamic planner used swept boxes')
        monkeypatch.setattr(env.env,'planning_obstacles',no_swept_planning)
        monkeypatch.setattr(TimedRRT,'plan',lambda self,start,goal:[start.copy(),start.copy()])
        expert=make_rrt(env,61)
        assert expert.timed and isinstance(expert.planner,TimedRRT)
        assert_planning_paused(env,before)
        env.step(np.zeros(7))
        assert env.env.sim_time>0 and not np.array_equal(before['centers'],planning_snapshot(env)['centers'])
        assert np.array_equal(before['reference'],env.cursor.path)
    finally:env.close()


def test_legacy_conditional_environment_still_rejects_dynamic_and_bad_phase_is_rejected(setup):
    spec,_,config,settings,static=setup;case=add_motion(static,spec)
    options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']}
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings)
    try:
        with pytest.raises(ValueError,match='restricted to static'):env.reset(options=options)
    finally:env.close()
    env=SlowDynamicFourEnv(config,[],settings)
    try:
        bad=copy.deepcopy(options);bad['scenario']['obstacles'][0]['center'][0]+=.01
        with pytest.raises(ValueError,match='phase at t=0'):env.reset(options=bad)
    finally:env.close()


def test_verification_entry_checks_only_by_default(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import verify_four_slow_dynamic as entry
    monkeypatch.setattr(entry,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Unexpected execution')))
    monkeypatch.setattr('sys.argv',['verify_four_slow_dynamic.py']);entry.main()
    assert 'check_only' in capsys.readouterr().out
