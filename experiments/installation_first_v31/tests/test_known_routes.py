"""Known geometry, true travel speed, policy migration and rollback contracts."""
import copy
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3.common.env_checker import check_env
from wall_panel_rl.envs.known_route_motion import new_motion,route_state,arc_table,OBSERVATIONS
from wall_panel_rl.rl.known_route_env import KnownRouteEnv
from wall_panel_rl.rl.variable_weaving import VariableCaseBankEnv
from wall_panel_rl.rl.predictive_safety import PredictiveSafetyController
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot,assert_planning_paused
from wall_panel_rl.submission import load_json,load_runtime,load_agent
from wall_panel_rl.utils.config import ROOT


def case(count):
    return load_json(ROOT/f'checks_site/count{count}.json')


def test_speed_changes_timing_without_changing_geometry():
    m=new_motion([0,0,.5],[.09,.1,.1],np.random.default_rng(12))
    slow=copy.deepcopy(m);fast=copy.deepcopy(m)
    slow['speed_m_s']=.03;fast['speed_m_s']=.18
    for seconds in np.linspace(0,30,50):
        ps,vs,_=route_state(slow,seconds)
        pf,vf,_=route_state(fast,seconds/6)
        assert np.allclose(ps,pf,atol=1e-12)
        assert np.linalg.norm(vs)==pytest.approx(.03)
        assert np.linalg.norm(vf)==pytest.approx(.18)


def test_analytic_velocity_matches_measured_positions_and_closed_period():
    m=new_motion([0,0,.5],[.09,.1,.1],np.random.default_rng(8))
    m['speed_m_s']=.18
    _,arc=arc_table(m)
    for seconds in np.linspace(.013,25,60):
        position,velocity,_=route_state(m,seconds)
        numerical=(route_state(m,seconds+1e-5)[0]-route_state(m,seconds-1e-5)[0])/2e-5
        assert np.linalg.norm(numerical-velocity)<.002
        assert np.allclose(position,route_state(m,seconds+arc[-1]/.18)[0],atol=1e-10)


@pytest.mark.parametrize('count',[2,3,4,5])
def test_geometry_information_and_masks_are_correct(count):
    rt=load_runtime();c=case(count);env=KnownRouteEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':c['scene'],'task':'dynamic','strength':c['strength']})
        assert obs.shape==(OBSERVATIONS,) and obs.dtype==np.float32
        old=obs[20:70].reshape(5,10);routes=obs[77:].reshape(5,18)
        assert np.all(old[count:]==0) and np.all(routes[count:]==0)
        for slot in range(count):
            f=routes[slot];motion=c['scene']['obstacle_motion'][slot]
            progress=np.arctan2(f[15],f[16])
            phase=np.arctan2(f[9:12],f[12:15])
            reconstructed=f[:3]+f[3:6]*.18*np.sin(f[6:9]*3*progress+phase)
            assert np.allclose(reconstructed,old[slot,:3],atol=2e-6)
            assert f[17]*.18==pytest.approx(motion['speed_m_s'])
        speeds=[m['speed_m_s'] for m in env.env.obstacle_motion]
        env.step(np.zeros(7,np.float32))
        assert speeds==[m['speed_m_s'] for m in env.env.obstacle_motion]
    finally:env.close()


def test_selected_policy_distribution_survives_reload(tmp_path):
    new=load_agent()
    rt=load_runtime();c=case(5);env=KnownRouteEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':c['scene'],'task':'dynamic','strength':c['strength']})
        path=tmp_path/'selected.zip';new.save(path);old=load_agent(path)
        assert np.array_equal(old.predict(obs,deterministic=True)[0],new.predict(obs,deterministic=True)[0])
        with torch.no_grad():
            a=old.policy.get_distribution(torch.tensor(obs[None])).distribution
            b=new.policy.get_distribution(torch.tensor(obs[None])).distribution
            assert torch.equal(a.loc,b.loc) and torch.equal(a.scale,b.scale)
    finally:env.close()


def test_forecast_rolls_back_route_progress_and_constant_speeds():
    rt=load_runtime();c=case(2);env=KnownRouteEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':c['scene'],'task':'dynamic','strength':c['strength']})
        before=planning_snapshot(env);motions=copy.deepcopy(env.env.obstacle_motion)
        controller=PredictiveSafetyController(load_agent(),env,horizon=2)
        action=controller.action(obs)
        assert action.shape==(7,)
        assert_planning_paused(env,before)
        for old,new in zip(motions,env.env.obstacle_motion):
            for key in ('anchor','extent','harmonics','phase','speed_m_s','start_distance_m'):
                assert np.array_equal(old[key],new[key])
            assert np.array_equal(route_state(old,env.env.sim_time)[0],route_state(new,env.env.sim_time)[0])
    finally:env.close()


def test_local_recovery_trials_preserve_known_routes_and_reference():
    from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController
    from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot
    rt=load_runtime();c=case(4);env=KnownRouteEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':c['scene'],'task':'dynamic','strength':c['strength']})
        spec=copy.deepcopy(rt['controller']);spec['recovery']['horizon']=4
        controller=TimeoutRecoveryController(load_agent(),env,2,spec['gains'],spec['recovery'])
        before=planning_snapshot(env)
        for kind,value in [('bias',(5,.4)),('lookahead',(3,.7)),('persistent',([1,3],[.28,-.28]))]:
            with EpisodeSnapshot(env):
                _,_,_,actions=controller._candidate(kind,value,obs.copy())
                assert actions and all(np.isfinite(a).all() for a in actions)
            assert_planning_paused(env,before)
    finally:env.close()


def test_gym_and_new_route_sampling_are_reproducible():
    rt=load_runtime();spec=yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())
    env=VariableCaseBankEnv(rt,[case(n) for n in (2,3,4,5)],spec['training'],918)
    try:
        check_env(env,warn=True)
        first,_=env.reset(seed=918);shapes=copy.deepcopy(env.env.obstacle_motion)
        same,_=env.reset(seed=918)
        assert np.array_equal(first,same)
        next_obs,_=env.reset()
        assert not np.array_equal(same,next_obs)
        assert all(.04<=m['speed_m_s']<=.22 for m in env.env.obstacle_motion)
        assert all('omega' not in m for m in env.env.obstacle_motion)
    finally:env.close()


def test_declared_timed_conflicts_use_real_boxes():
    import pybullet as p
    from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
    rt=load_runtime()
    for count in (2,3,4,5):
        c=case(count);s=c['scene'];env=WallPanelEnv(rt['configuration'])
        try:
            env.reset(seed=s['seed'],options={'scenario':s})
            for slot,r in enumerate(s['motion_anchor_placement']['obstacles']):
                if not r['timed_conflict']:continue
                center=route_state(s['obstacle_motion'][slot],r['guide_time_s'])[0]
                assert np.allclose(center,s['obstacle_motion'][slot]['anchor'],atol=1e-12)
                p.resetBasePositionAndOrientation(env.obstacle_ids[slot],center,[0,0,0,1],physicsClientId=env.client)
                with env.collision.at_configuration(r['nominal_joint_position']):
                    assert (p.getClosestPoints(env.panel,env.obstacle_ids[slot],0,physicsClientId=env.client) or
                            p.getClosestPoints(env.robot,env.obstacle_ids[slot],0,physicsClientId=env.client))
        finally:env.close()
