"""Installation priority, unchanged baseline, reversible repair, and data gates."""
import json
import sys
import copy
import numpy as np
import pytest
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_runtime,load_agent,load_json,make_controller
from wall_panel_rl.rl.variable_weaving import VariableDynamicEnv
from wall_panel_rl.rl.installation_recovery import InstallationRecoveryController
from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot,assert_planning_paused
from wall_panel_rl.rl.timeout_diagnostics import diagnose
sys.path.insert(0,str(ROOT/'scripts'))
import run_variable_training as training


@pytest.fixture
def physical():
    rt=load_runtime(ROOT/'configs/installation_runtime.json')
    case=load_json(ROOT/'checks_site/count3.json')
    env=VariableDynamicEnv(rt['configuration'],[],rt['settings'])
    obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
    try:yield rt,case,env,obs
    finally:env.close()


def test_default_restores_old_controller_and_new_profile_is_explicit(physical):
    _,_,env,_=physical;model=load_agent()
    assert type(make_controller(model,env,load_runtime()['controller'])) is TimeoutRecoveryController
    c=make_controller(model,env,load_runtime(ROOT/'configs/installation_runtime.json')['controller'])
    assert type(c) is InstallationRecoveryController


def test_disabled_repairs_preserve_original_action(physical):
    rt,_,env,obs=physical;model=load_agent();spec=rt['controller']
    old=TimeoutRecoveryController(model,env,spec['selected']['horizon'],spec['gains'],spec['recovery'])
    fixed=InstallationRecoveryController(model,env,spec['selected']['horizon'],spec['gains'],spec['recovery'],installation={'enabled':False})
    assert np.array_equal(old.action(obs),fixed.action(obs))


def test_old_accepted_plan_is_never_replaced(physical,monkeypatch):
    rt,_,env,obs=physical;c=make_controller(load_agent(),env,rt['controller'])
    original=('existing',.7,.1,[np.zeros(7,np.float32)])
    monkeypatch.setattr(TimeoutRecoveryController,'_find_plan',lambda *args:original)
    monkeypatch.setattr(c,'_loses_reference',lambda *args:False)
    assert c._find_plan(obs,1.) is original


def test_finish_forecast_restores_state_and_predicts_true_installation(physical):
    rt,_,env,obs=physical;c=make_controller(load_agent(),env,rt['controller'])
    env.env.set_configuration(env.env.goal_q)
    before=planning_snapshot(env);motions=copy.deepcopy(env.env.obstacle_motion)
    found=c._finish_plan(obs)
    assert found is not None
    assert_planning_paused(env,before)
    for actual,expected in zip(env.env.obstacle_motion,motions):
        for key in actual:assert np.array_equal(actual[key],expected[key])
    _,_,_,actions=found
    for action in actions:
        _,_,done,truncated,info=env.step(action)
        if done or truncated:break
    assert done and info['termination_reason']=='success' and not info['collision']


def test_wait_and_rejoin_forecast_restores_reference_and_route_time(physical):
    rt,_,env,obs=physical;c=make_controller(load_agent(),env,rt['controller'])
    before=planning_snapshot(env);reference=env.cursor.path.copy()
    from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot
    with EpisodeSnapshot(env):
        c._candidate('installation_rejoin',(env.cursor.index,.7,4),obs)
    assert_planning_paused(env,before)
    assert np.array_equal(env.cursor.path,reference)


def test_actual_training_reference_escape_is_rejected_and_rolled_back():
    fixture=load_json(ROOT/'tests/fixtures/reference_escape.json')
    rt=load_runtime(ROOT/'configs/installation_runtime.json');case=fixture['case']
    env=VariableDynamicEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        for action in fixture['commands']:obs,_,_,_,_=env.step(action)
        controller=make_controller(load_agent(),env,rt['controller'])
        before=planning_snapshot(env)
        plan=(fixture['candidate_mode'],fixture['candidate_value'],0.,[])
        assert controller._loses_reference(plan,obs)
        assert_planning_paused(env,before)
    finally:env.close()


def test_selection_prefers_installation_success_over_weaving():
    def result(success,collision,weaving):
        return {'metrics':{'all':{'success':success,'collision':collision,'joint_limit':0,
                                'success_with_weaving':weaving,'mean_task_error':.1}}}
    assert training.score(result(5,1,0))>training.score(result(4,0,4))
    assert training.score(result(5,0,0))>training.score(result(5,1,5))


def test_teacher_retry_log_is_serializable_and_contains_only_success_labels(physical,monkeypatch):
    rt,case,_,_=physical
    monkeypatch.setitem(training.CACHE,'training',{'teacher_phase_retries':1})
    monkeypatch.setitem(training.CACHE,'runtime',rt)
    attempts=[]
    def attempt(selected,index,route_seed):
        attempts.append(index);success=index==1
        row={'seed':selected['scene']['seed'],'accepted_for_bc':success,'reason':'success' if success else 'timeout'}
        n=3 if success else 0
        return row,np.zeros((n,167),np.float32),np.zeros((n,7),np.float32),np.ones(n,np.float32)
    monkeypatch.setattr(training,'teacher_attempt',attempt)
    row,x,y,w=training.collect_teacher(case)
    json.dumps(row,allow_nan=False)
    assert attempts==[0,1] and row['accepted_for_bc'] and len(row['attempts'])==2
    assert x.shape==(3,167) and y.shape==(3,7) and len(w)==3


def test_timeout_diagnosis_distinguishes_waiting_and_precision():
    frames=[{'joints':[0.]*7,'reference_index':2,'reference_distance_rad':.3,
             'command_action':[0.]*7} for _ in range(41)]
    payload={'result':{'reason':'timeout','position_m':.6,'orientation_deg':80.},'trajectory':frames}
    row=diagnose(payload)
    assert row['primary']=='waiting' and row['flags']['stale_reference']
    payload['result'].update(position_m=.005,orientation_deg=5.)
    assert diagnose(payload)['primary']=='terminal_precision'


def test_no_improvement_keeps_old_bundle_and_never_starts_ppo(tmp_path,monkeypatch):
    """Exercise the real orchestration with failed teachers and no PPO resources."""
    from concurrent.futures import Future
    from types import SimpleNamespace
    import yaml
    scene_dir=tmp_path/'scenes';scene_dir.mkdir()
    for offset,split in enumerate(['train','validation','test']):
        cases=[{'scene':{'seed':100*offset+n,'obstacles':[{}]*n}} for n in [2,3,4,5]]
        (scene_dir/f'{split}_cases.json').write_text(json.dumps(cases))
    spec=yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())
    spec['training']['teacher_cases_by_count']={str(n):1 for n in [2,3,4,5]}
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(spec))
    class Pool:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,case):
            future=Future();future.set_result(({'accepted_for_bc':False,'reason':'timeout'},
                np.empty((0,167),np.float32),np.empty((0,7),np.float32),np.empty(0,np.float32)))
            return future
    evaluations=[]
    def evaluate(pool,path,cases,output,status,label,**kwargs):
        variant=kwargs.get('controller_variant');evaluations.append((label,variant))
        count=4 if variant=='v29_controller' else 1
        rows=[{'seed':i,'obstacle_count':2+i//2,'reason':'success' if i<count else 'timeout',
               'passage_count':0,'position_m':.1,'orientation_deg':1.} for i in range(8)]
        result={'metrics':training.summary(rows),'complete':True}
        training.save(output/'evaluations'/f'{label}.json',result)
        return result
    def forbidden(*args,**kwargs):raise AssertionError('PPO must not start without BC improvement')
    monkeypatch.setattr(training,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(training,'evaluate',evaluate)
    monkeypatch.setattr(training,'SubprocVecEnv',forbidden)
    monkeypatch.setattr(training,'plot_history',lambda *args:None)
    output=tmp_path/'training'
    training.run(SimpleNamespace(output=output,config=config,scenes=scene_dir))
    result=json.loads((output/'summary.json').read_text())
    assert result['controller_variant']=='v29_controller' and not result['ppo_started']
    assert not json.loads((output/'bc_gate.json').read_text())['passed']
    assert json.loads((output/'status.json').read_text())['ppo_steps_completed']==0
    assert json.loads((output/'selected_runtime.json').read_text())['controller']['recovery_strategy']=='timeout_v24'
    assert evaluations[-1]==('independent_test_enhanced','v29_controller')
