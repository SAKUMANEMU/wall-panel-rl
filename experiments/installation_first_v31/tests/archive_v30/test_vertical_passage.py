"""The new controller must keep prediction state and physical boundaries intact."""
import copy
import numpy as np
from wall_panel_rl.submission import load_runtime,load_agent,load_json,make_controller
from wall_panel_rl.rl.variable_weaving import VariableDynamicEnv,VariableCaseBankEnv
from wall_panel_rl.rl.vertical_passage import VerticalPassageController
from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot,assert_planning_paused
from wall_panel_rl.utils.config import ROOT
import yaml


def test_vertical_candidate_search_restores_physics_and_routes():
    rt=load_runtime();case=load_json(ROOT/'checks_site/count3.json')
    env=VariableDynamicEnv(rt['configuration'],[],rt['settings'])
    try:
        env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        controller=make_controller(load_agent(),env,rt['controller'])
        assert isinstance(controller,VerticalPassageController)
        before=planning_snapshot(env);original=copy.deepcopy(env.env.obstacle_motion)
        q,velocity=env.env.joint_state();path=env.cursor.path.copy()
        controller._vertical_plan(case['scene']['motion_anchor_placement']['route_slot_order'][0])
        assert_planning_paused(env,before)
        assert np.allclose(env.env.joint_state()[0],q,atol=1e-12)
        assert np.allclose(env.env.joint_state()[1],velocity,atol=1e-12)
        assert np.array_equal(env.cursor.path,path)
        for actual,expected in zip(env.env.obstacle_motion,original):
            assert actual.keys()==expected.keys()
            for key in actual:
                assert np.array_equal(actual[key],expected[key])
        assert not env.env.collision.site_contact_types()
    finally:env.close()


def test_disabled_passage_extension_preserves_timeout_controller_action():
    rt=load_runtime();case=load_json(ROOT/'checks_site/count2.json')
    env=VariableDynamicEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        model=load_agent();c=rt['controller']
        disabled=VerticalPassageController(model,env,c['selected']['horizon'],c['gains'],c['recovery'],passage={'enabled':False})
        baseline=TimeoutRecoveryController(model,env,c['selected']['horizon'],c['gains'],c['recovery'])
        first=disabled.action(obs)
        second=baseline.action(obs)
        assert np.array_equal(first,second)
    finally:env.close()


def test_terminal_potential_uses_zero_absorbing_value():
    rt=load_runtime();spec=yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())['training']
    cases=[load_json(ROOT/f'checks_site/count{n}.json') for n in (2,3,4,5)]
    env=VariableCaseBankEnv(rt,cases,spec,31)
    try:
        env.reset(seed=31);env.env.steps=env.env.config['simulation']['max_steps']-1
        previous=env.vertical_potential
        _,_,done,truncated,info=env.step(np.zeros(7,np.float32))
        assert done or truncated
        assert env.vertical_potential==0.
        assert np.isclose(info['reward_vertical_potential'],-spec['vertical_potential_weight']*previous)
    finally:env.close()


def test_successful_finish_forecast_restores_state_and_executes_real_success():
    rt=load_runtime();case=load_json(ROOT/'checks_site/count2.json')
    env=VariableDynamicEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        env.env.set_configuration(env.env.goal_q)
        controller=make_controller(load_agent(),env,rt['controller'])
        before=planning_snapshot(env)
        finish=controller._finish_plan(obs)
        assert finish is not None
        assert_planning_paused(env,before)
        actions,_,_,_=finish
        for action in actions:
            _,_,done,truncated,info=env.step(action)
            if done or truncated:break
        assert done and info['termination_reason']=='success'
        assert not info['collision']
    finally:env.close()
