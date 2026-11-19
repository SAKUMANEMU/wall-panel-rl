"""Autonomy, warm-start and curriculum checks without any optimizer updates."""
import copy
import json
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl import autonomous_dynamic_ppo as a
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import ROOT,load_config
from wall_panel_rl.utils.metrics import capture_scenario

@pytest.fixture
def settings():return yaml.safe_load((ROOT/'configs/dynamic_autonomous_pilot.yaml').read_text())

@pytest.fixture
def model(settings):
    torch.set_num_threads(1)
    return a.initialize_from_tracking(ROOT/settings['source'],settings)

@pytest.fixture(scope='module')
def scene():
    env=WallPanelEnv(load_config(ROOT/'configs/ppo_dynamic_clean.yaml'))
    try:
        env.reset(seed=10002);return capture_scenario(env,10002)
    finally:env.close()


def test_warm_start_and_obstacle_feature_learning(model,settings):
    old=PPO.load(ROOT/settings['source'],device='cpu')
    with np.load(ROOT/'releases/path_reference_bc_20261025/training/teacher_arrays.npz') as d:
        x=torch.as_tensor(d['obs'][:32].copy())
    with torch.no_grad():
        assert torch.equal(model.policy.tracking_mean(x),old.policy.tracking_mean(x))
    assert model.num_timesteps==old.num_timesteps==5000
    assert model.rollout_buffer.n_envs==settings['workers']
    assert not model.policy.tracking_gain_raw.requires_grad
    assert all(p.requires_grad for p in model.policy.residual.parameters())
    # Only a temporary test network is modified to expose the new sensor pathway.
    with torch.no_grad():model.policy.residual[-1].weight.fill_(.01)
    x=torch.zeros((1,77),requires_grad=True)
    model.policy.tracking_mean(x).sum().backward()
    assert torch.any(x.grad[:,20:70]!=0)  # Obstacle slots affect actor decisions.
    assert torch.any(x.grad[:,26:29]!=0)  # Velocity features are connected too.
    assert model.policy.tracking_gain_raw.grad is None


def test_reload_preserves_residual_and_mask(model,tmp_path):
    path=tmp_path/'model.zip';model.save(path)
    loaded=PPO.load(path,device='cpu')
    assert isinstance(loaded.policy,a.DynamicResidualPolicy)
    assert loaded.policy.residual_scale==2
    assert not loaded.policy.tracking_bias.requires_grad
    assert not loaded.policy.log_std.requires_grad
    assert not loaded.policy.tracking_velocity.requires_grad
    assert all(p.requires_grad for p in loaded.policy.residual.parameters())


def test_motion_schedule_and_bounds(scene):
    assert a.difficulty(0)==a.difficulty(.2)==(.1,.05)
    assert np.allclose(a.difficulty(.8),(.4,.4))
    assert np.allclose(a.difficulty(1),(.4,.4))
    weak=a.scenario_at_strength(scene,.1,.05,np.random.default_rng(42))
    for box,spec in zip(weak['obstacles'],weak['obstacle_motion']):
        extent=np.asarray(spec['extent']);omega=np.asarray(spec['omega'])
        assert np.linalg.norm(extent)<=.100000001
        assert np.linalg.norm(extent*omega)<=.050000001
        assert np.allclose(box['center'],np.asarray(spec['anchor'])+extent*np.sin(spec['phase']))
    assert np.allclose(scene['obstacle_motion'][0]['extent'],np.full(3,.4/np.sqrt(3)))


def test_no_online_planner_and_static_dynamic_switch(scene,settings,monkeypatch):
    from wall_panel_rl.planning.rrt_connect import RRTConnect
    from wall_panel_rl.baselines.classical import ClassicalController
    def forbidden(*args,**kwargs):raise AssertionError('Online planner must never be invoked')
    monkeypatch.setattr(RRTConnect,'plan',forbidden)
    monkeypatch.setattr(ClassicalController,'__init__',forbidden)
    env=a.AutonomousDynamicEnv(load_config(ROOT/'configs/ppo_dynamic_clean.yaml'),[scene],settings)
    try:
        for task in ['static','dynamic','alignment','dynamic']:
            current=a.without_obstacles(scene,.1) if task=='alignment' else scene
            obs,info=env.reset(options={'scenario':current,'task':task,'strength':(.4,.4)})
            assert obs.shape==(77,) and not info['online_avoidance_planner']
            before=np.asarray([c.copy() for c,d in env.env.obstacles])
            obs,_,_,_,info=env.step(np.zeros(7))
            after=np.asarray([c for c,d in env.env.obstacles])
            if task=='dynamic':assert not np.array_equal(before,after)
            if task=='static':assert np.array_equal(before,after) and np.all(obs[20:70].reshape(5,10)[:,6:9]==0)
            if task=='alignment':assert len(after)==0 and np.all(obs[20:70]==0)
            assert not info['online_avoidance_planner']
    finally:env.close()


def test_reference_cursor_can_advance_after_detour():
    path=np.array([np.zeros(7),np.full(7,.06),np.full(7,.12)])
    cursor=a.ReferenceCursor(path)
    assert np.allclose(cursor.reference(np.full(7,.07)),path[-1])


def test_post_update_final_validation(model,settings,tmp_path,monkeypatch):
    seen=[]
    def evaluate(candidate,*_):
        seen.append(candidate.num_timesteps)
        m={s:{'success':1,'collision':0,'mean_reward':1.} for s in
           ('dynamic_full','dynamic_easy','static','alignment_far','alignment_near')}
        return {'dynamic_ppo':m},[]
    monkeypatch.setattr(a,'evaluate',evaluate)
    callback=a.DynamicValidation(tmp_path,None,{},settings,[],5000);callback.model=model
    callback._on_training_start()
    model.num_timesteps=10000;callback._on_step();callback._on_rollout_end()
    assert seen==[5000]
    callback._on_rollout_start();callback._on_rollout_start();assert seen==[5000,10000]
    model.num_timesteps=15000;callback._on_training_end();assert seen==[5000,10000,15000]
    assert json.loads((tmp_path/'progress.json').read_text())['kind']=='final_saved_reload'
