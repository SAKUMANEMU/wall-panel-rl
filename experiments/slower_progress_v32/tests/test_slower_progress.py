"""V32 physical speed bounds and reversible early reference rejoining."""
import copy
import numpy as np
import pytest
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_runtime,load_json,load_agent,make_controller
from wall_panel_rl.rl.variable_weaving import VariableDynamicEnv,VariableCaseBankEnv
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot,assert_planning_paused


def slow_case(count):
    case=copy.deepcopy(load_json(ROOT/f'checks_site/count{count}.json'))
    for motion in case['scene']['obstacle_motion']:
        motion['speed_m_s']=min(motion['speed_m_s'],.18)
    case['strength'][1]=.18
    return case


@pytest.mark.parametrize('count',[2,3,4,5])
def test_speed_bounds_reach_actual_physics_and_neural_observations(count):
    runtime=load_runtime();case=slow_case(count)
    env=VariableDynamicEnv(runtime['configuration'],[],runtime['settings'])
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        assert obs.shape==(167,) and env.action_space.shape==(7,)
        velocities=obs[77:].reshape(5,18)[:count,17]*.18
        assert np.allclose(velocities,[m['speed_m_s'] for m in env.env.obstacle_motion])
        before=env.env.sim_time
        env.step(np.zeros(7,np.float32))
        assert env.env.sim_time-before==pytest.approx(.05)
        assert all(.04<=m['speed_m_s']<=.18 for m in env.env.obstacle_motion)
    finally:env.close()


def test_random_training_resets_use_slower_speed_without_changing_task_limits():
    import yaml
    runtime=load_runtime();spec=yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())
    env=VariableCaseBankEnv(runtime,[slow_case(n) for n in [2,3,4,5]],spec['training'],32)
    try:
        for seed in [32,33,34,35]:
            env.reset(seed=seed)
            assert all(.04<=m['speed_m_s']<=.18 for m in env.env.obstacle_motion)
        assert env.env.config['simulation']['max_steps']==240
        assert env.env.config['target']['position_tolerance']==.01
        assert env.env.config['target']['orientation_tolerance_deg']==2.
        assert env.env.config['construction_site']['floor_z']==.1
        assert env.env.config['construction_site']['ceiling_z']==1.1
    finally:env.close()


def test_rejoin_search_restores_time_routes_cursor_and_joints_at_success():
    runtime=load_runtime();case=slow_case(3)
    env=VariableDynamicEnv(runtime['configuration'],[],runtime['settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        env.env.set_configuration(env.env.goal_q)
        env.cursor.index=len(env.cursor.path)-1
        before=planning_snapshot(env);path=env.cursor.path.copy()
        motion=copy.deepcopy(env.env.obstacle_motion)
        controller=make_controller(load_agent(),env,runtime['controller'])
        plan=controller._find_rejoin_plan(obs)
        assert plan is not None and plan[0]=='installation_rejoin'
        assert_planning_paused(env,before)
        assert np.array_equal(env.cursor.path,path)
        for actual,expected in zip(env.env.obstacle_motion,motion):
            assert actual.keys()==expected.keys()
            assert all(np.array_equal(actual[k],expected[k]) for k in actual)
        _,_,terminated,truncated,info=env.step(plan[3][0])
        assert terminated and info['termination_reason']=='success' and not info['collision']
    finally:env.close()


def test_source_controller_keeps_previous_search_order_and_guard_thresholds():
    source=load_json(ROOT/'configs/source_controller.json')
    changed=load_runtime()['controller']
    assert not source['installation'].get('route_first',False)
    assert changed['installation']['route_first']
    assert changed['installation']['wait_steps']==[0,2,4]
    assert source['installation']['wait_steps']==[0,4,8,12]
    assert changed['recovery']['minimum_step']<source['recovery']['minimum_step']
