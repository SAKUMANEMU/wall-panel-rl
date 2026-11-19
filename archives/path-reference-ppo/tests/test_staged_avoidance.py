"""Code-only curriculum/teacher/gating checks. No learning or expert collection."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl import avoidance_curriculum as c
from wall_panel_rl.rl.avoidance_imitation import imitation_split
from wall_panel_rl.utils.config import ROOT,load_config

@pytest.fixture
def settings(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from staged_avoidance import resolve,check
    s=resolve(yaml.safe_load((ROOT/'configs/staged_avoidance.yaml').read_text()),'one_static')
    check(s);return s

@pytest.fixture
def scene():
    return json.loads((ROOT/'data/dynamic_autonomous_pilot_seed61/train.json').read_text())[0]


def test_gated_policy_transfer_freeze_std_and_reload(settings,tmp_path):
    torch.set_num_threads(1)
    model=c.initialize_gated(ROOT/settings['source_model'],settings)
    old=PPO.load(ROOT/settings['source_model'],device='cpu')
    x=torch.zeros((2,77));x[:,70:]=.02
    with torch.no_grad():
        # No obstacle slots: learned gate/residual cannot turn off free-space tracking.
        assert torch.allclose(model.policy.tracking_mean(x),old.policy.tracking_mean(x),atol=1e-7)
        assert torch.allclose(model.policy.log_std.exp(),torch.full((7,),.1))
    assert not model.policy.tracking_bias.requires_grad
    assert all(p.requires_grad for p in model.policy.prior_gate.parameters())
    path=tmp_path/'model.zip';model.save(path);loaded=PPO.load(path,device='cpu')
    assert isinstance(loaded.policy,c.ReferencePreservingPolicy)
    assert not loaded.policy.log_std.requires_grad
    assert torch.equal(loaded.policy.prior_gate[-1].bias,model.policy.prior_gate[-1].bias)
    assert model.rollout_buffer.n_envs==4


def test_stage_counts_static_motion_and_unchanged_bank(scene,settings):
    original=copy.deepcopy(scene)
    for stage in settings['stages']:
        selected=c.stage_scene(scene,stage)
        assert len(selected['obstacles'])==stage['count']
        assert len(selected['obstacle_motion'])==(0 if stage['mode']=='static' else stage['count'])
    assert scene==original
    single=c.stage_scene(scene,settings['stages'][0])
    assert single['obstacle_original_indices']==[2]


def test_distance_feedback_all_arm_links_and_payload(monkeypatch):
    queried=[]
    class Physics:
        @staticmethod
        def getClosestPoints(body,obstacle,distance,**kw):
            queried.append((body,obstacle))
            value=.04 if body==1 else .06
            return [(None,)*8+(value,)]
    monkeypatch.setattr(c,'p',Physics)
    env=type('Env',(),{'obstacle_ids':[3,4],'robot':1,'panel':2,'client':0})()
    penalty,distance=c.clearance_feedback(env,.08,.5)
    assert set(queried)=={(1,3),(1,4),(2,3),(2,4)}
    assert np.isclose(penalty,-.125) and distance==.04
    env.obstacle_ids=[]
    assert c.clearance_feedback(env,.08,.5)==(0.,None)


def test_runtime_no_oracle_and_reward_composition(scene,settings,monkeypatch):
    from wall_panel_rl.baselines.classical import ClassicalController
    from wall_panel_rl.planning.rrt_connect import RRTConnect
    def forbidden(*a,**k):raise AssertionError('Expert/planner must not run at policy execution')
    monkeypatch.setattr(ClassicalController,'__init__',forbidden)
    monkeypatch.setattr(RRTConnect,'plan',forbidden)
    env=c.StagedAvoidanceEnv(load_config(ROOT/settings['environment']),[scene],settings)
    try:
        for stage in settings['stages'][:3]:
            env.settings=copy.deepcopy(settings);env.settings['active_stage']=stage
            env.settings['mixture']={'alignment':0.,'static':0.,'dynamic':1.}
            obs,info=env.reset(seed=42)
            assert obs.shape==(77,) and len(env.env.obstacles)==stage['count']
            _,reward,_,_,info=env.step(np.zeros(7))
            assert not info['online_avoidance_planner']
            assert np.isclose(reward,info['reward_base']+info['reward_waypoint_progress']+info['reward_obstacle_clearance'])
            if stage['mode']=='static':assert env.env.obstacle_motion==[]
    finally:env.close()


def metrics(success=18,collision=1):
    return {'stage_target':{'episodes':20,'success':success,'collision':collision,
        'joint_limit':0,'timeout':20-success-collision},
        'alignment_far':{'episodes':10,'success':10},'alignment_near':{'episodes':10,'success':9}}


def test_acceptance_requires_samples_retention_and_distinct_updates(settings):
    gate=settings['gate'];m=metrics()
    assert c.gate_result(m,gate)['passed']
    short=copy.deepcopy(m);short['stage_target'].update(episodes=10,success=9,collision=1,timeout=0)
    assert not c.gate_result(short,gate)['passed']
    forgotten=copy.deepcopy(m);forgotten['alignment_far']['success']=8
    assert not c.gate_result(forgotten,gate)['passed']
    history=[{'kind':'baseline','additional_steps':0,'metrics':{'dynamic_ppo':m}},
             {'kind':'post_update','additional_steps':5000,'metrics':{'dynamic_ppo':m}}]
    assert not c.assess_history(history,gate)['passed']
    final={'kind':'final_saved_reload','additional_steps':10000,'metrics':{'dynamic_ppo':m}}
    assert c.assess_history(history+[final],gate)['passed']
    final['additional_steps']=5000
    assert not c.assess_history(history+[final],gate)['passed']


def test_imitation_split_groups_same_layout_seeds():
    seeds=np.array([1,1,2,2,3,3,4,4,5,5]);offsets=np.arange(11)*2
    train,held=imitation_split(offsets,seeds,42)
    assert not set(seeds[train])&set(seeds[held])
    assert len(train)+len(held)==10


def test_stage_promotion_blocks_missing_or_failed_previous(settings,monkeypatch,tmp_path):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from staged_avoidance import resolve,require_previous
    root=copy.deepcopy(settings);root['output_root']=str(tmp_path)
    second=resolve(root,'one_dynamic')
    with pytest.raises(FileNotFoundError):require_previous(second)
    previous=tmp_path/'one_static/ppo';previous.mkdir(parents=True)
    (previous/'assessment.json').write_text(json.dumps({'passed':False,'stage':'one_static','thresholds':settings['gate']}))
    with pytest.raises(ValueError,match='not passed'):require_previous(second)


def test_check_mode_cannot_train_or_collect(settings,monkeypatch,tmp_path,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import staged_avoidance as entry
    root=yaml.safe_load((ROOT/'configs/staged_avoidance.yaml').read_text())
    root['output_root']=str(tmp_path/'models');root['data_root']=str(tmp_path/'data')
    root['source_first']=str(ROOT/settings['source_model'])
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(root))
    def forbidden(*a,**k):raise AssertionError('Expensive action in check mode')
    monkeypatch.setattr(entry,'prepare_stage',forbidden)
    monkeypatch.setattr(entry,'collect_teacher',forbidden)
    monkeypatch.setattr(entry,'pretrain_actor',forbidden)
    monkeypatch.setattr(PPO,'learn',forbidden)
    monkeypatch.setattr(sys,'argv',['staged_avoidance.py','--config',str(path)])
    entry.main()
    assert 'check_only' in capsys.readouterr().out
    assert not (tmp_path/'models').exists() and not (tmp_path/'data').exists()


def test_offline_teacher_keeps_policy_reference_and_logs_failures(scene,settings,monkeypatch,tmp_path):
    from wall_panel_rl.rl import avoidance_imitation as module
    class Env:
        def __init__(self,*args):
            self.env=type('Physical',(),{'steps':0})();self.cursor=type('Cursor',(),{'recoveries':0})()
        def reset(self,**kwargs):
            self.env.steps=0
            obs=np.zeros(77,np.float32);obs[70:]=.123
            return obs,{}
        def step(self,action):
            self.env.steps=1
            return np.zeros(77,np.float32),0.,True,False,{
                'termination_reason':'success','executed_action':[.25]*7}
        def close(self):pass
    class Expert:
        attempts=0
        def __init__(self,*args):
            self.planning_success=Expert.attempts>0;Expert.attempts+=1
            self.timed=False;self.replans=0;self.planning_time=0.
            self.path=np.full((2,7),9.)  # Must NEVER become the policy's reference input.
        def action(self,*args):return np.full(7,.5)
    monkeypatch.setattr(module,'StagedAvoidanceEnv',Env)
    monkeypatch.setattr(module,'make_teacher',lambda *args:Expert())
    s=copy.deepcopy(settings);s['teacher'].update(alignment_episodes=1,avoidance_episodes=1,maximum_attempts=3)
    module.collect_teacher({},[scene],s,tmp_path/'teacher')
    with np.load(tmp_path/'teacher/teacher.npz',allow_pickle=False) as d:
        assert np.allclose(d['obs'][:,70:],.123)
        assert np.allclose(d['actions'],.25) and np.allclose(d['raw_actions'],.5)
        assert len(d['offsets'])==3
        assert d['terminal_obs'].shape==(2,77) and d['terminal_actions'].shape==(2,7)
    attempts=json.loads((tmp_path/'teacher/attempts.json').read_text())
    assert len(attempts)==3 and attempts[0]['reason']=='planning_failure'
    assert json.loads((tmp_path/'teacher/metadata.json').read_text())['completed']


def test_incomplete_teacher_cannot_start_optimization(settings,monkeypatch,tmp_path):
    from wall_panel_rl.rl import avoidance_imitation as module
    folder=tmp_path/'teacher';folder.mkdir()
    (folder/'metadata.json').write_text(json.dumps({'completed':False}))
    def forbidden(*a,**k):raise AssertionError('Optimizer must not be created')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    with pytest.raises(ValueError,match='incomplete'):
        module.pretrain_actor(None,folder,settings,tmp_path/'bc')
    assert not (tmp_path/'bc').exists()
