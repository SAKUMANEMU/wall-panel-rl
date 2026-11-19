"""Dynamic-only gates, real stagger geometry, train-only BC source preservation."""
import copy,json
import numpy as np
import pybullet as pb
import pytest,torch,yaml
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors
from wall_panel_rl.rl.slow_dynamic_four import dynamic_configuration,validate_placement
from wall_panel_rl.rl.dynamic_four_training import DynamicFourBankEnv,curriculum_gate
from stable_baselines3 import PPO

@pytest.fixture
def setup():
    spec=yaml.safe_load((ROOT/'configs/four_staggered_dynamic_preparation.yaml').read_text())
    source=json.loads((ROOT/spec['source_report']/'candidate_manifest.json').read_text())
    profile=yaml.safe_load((ROOT/spec['motion_profile']).read_text())
    config,settings=dynamic_configuration(source['configuration'],source['evaluation_settings'],profile)
    old=json.loads((ROOT/spec['source_train_cases']).read_text())[0]
    return spec,profile,config,settings,old

def test_stagger_is_alternating_with_true_on_path_contacts_and_full_bounds(setup):
    spec,profile,config,settings,old=setup;saved=copy.deepcopy(old)
    case=fit_motion_anchors(config,settings,profile,old);assert old==saved
    assert case==fit_motion_anchors(config,settings,profile,old)
    scene=case['scene'];records=scene['motion_anchor_placement'];order=records['route_slot_order']
    offsets=np.array([records['obstacles'][i]['vertical_offset_m'] for i in order])
    assert np.all(offsets[:-1]*offsets[1:]<0) and np.all(np.abs(offsets)>=.03) and np.all(np.abs(offsets)<=.07)
    heights=np.array([scene['obstacle_motion'][i]['anchor'][2] for i in order]);assert heights.ptp()>=.15 and np.min(np.abs(np.diff(heights)))>=.04
    for key in ['initial_q','reference_joint_path','nominal_motion']:assert scene[key]==old['scene'][key]
    validate_placement(config,case)
    for m in scene['obstacle_motion']:
        assert np.isclose(np.linalg.norm(m['extent']),.18) and np.linalg.norm(np.array(m['extent'])*m['omega'])<=.12+1e-12
    from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
    env=WallPanelEnv(config)
    try:
        empty=copy.deepcopy(scene);empty.update(obstacles=[],obstacle_motion=[]);env.reset(options={'scenario':empty})
        for box,m,w in zip(scene['obstacles'],scene['obstacle_motion'],records['obstacles']):
            body=env.box(box['dimensions'],m['anchor'],[1,0,0,0])
            with env.collision.at_configuration(w['nominal_joint_position']):
                assert pb.getClosestPoints(env.panel,body,0,physicsClientId=env.client) or pb.getClosestPoints(env.robot,body,0,physicsClientId=env.client)
            pb.removeBody(body,physicsClientId=env.client)
    finally:env.close()

def test_all_reset_tasks_are_dynamic_and_level_change_waits_until_reset(setup):
    spec,profile,config,settings,old=setup;case=fit_motion_anchors(config,settings,profile,old)
    env=DynamicFourBankEnv(config,settings,[case],spec,seed=91)
    try:
        env.reset();before=copy.deepcopy(env.last_scene);env.set_curriculum(2)
        assert env.last_scene==before and env.amplitude==.15 and env.speed==.1
        for _ in range(10):
            env.reset();assert env.task=='dynamic' and env.amplitude==.18 and env.speed==.12 and len(env.last_scene['obstacle_motion'])==4
    finally:env.close()

def test_dynamic_only_selection_and_curriculum_need_no_static_metrics(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import train_four_dynamic_ppo as trainer
    m=dict(episodes=64,success=50,collision=10,timeout=4,joint_limit=0)
    assert trainer.select_score(m,m,None,None,None,None)==[1,50,-10,-4]
    assert trainer.select_score(dict(m,collision=11),m,None,None,None,None)[0]==0
    assert curriculum_gate(m,m,None,None,m,m)['passed']
    assert not curriculum_gate(m,m,None,None,dict(m,success=49),m)['passed']

def test_collision_diagnosis_ignores_expected_adjacent_links(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import run_staggered_dynamic_warmup as warm
    from types import SimpleNamespace
    e=SimpleNamespace(config={'collision':{'margin':0}},client=1,obstacle_ids=[20],wall=30,robot=1,panel=2,ee=6)
    def contact(a,b,margin,**kw):
        if a==b:return [(0,0,0,2,3,0,0,0,-.01),(0,0,0,2,5,0,0,0,-.02)]
        if (a,b)==(1,2):return [(0,0,0,6,-1,0,0,0,-.01)]
        if (a,b)==(2,20):return [(0,0,0,-1,-1,0,0,0,-.03)]
        return []
    monkeypatch.setattr(warm.pb,'getClosestPoints',contact)
    pairs=warm.collision_pairs(e);assert len(pairs)==2 and {p['other'] for p in pairs}=={'arm','obstacle_0'}

def test_dynamic_only_entry_refuses_static_sampling_and_unverified_old_pool(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import train_four_dynamic_ppo as trainer
    spec=yaml.safe_load((ROOT/'configs/four_large_motion_training.yaml').read_text());spec['dynamic_only']=True
    with pytest.raises(ValueError,match='cannot sample static'):trainer.inputs(spec)
    spec['static_probability']=0
    with pytest.raises(ValueError,match='dynamic-only teacher pool'):trainer.inputs(spec)
