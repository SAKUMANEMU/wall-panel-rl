"""Interface checks only: no PPO.learn or optimizer.step."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl import waypoint_ppo as w
from wall_panel_rl.utils.config import ROOT,load_config

RELEASE=ROOT/'releases/path_reference_bc_20261025'

@pytest.fixture
def settings():return yaml.safe_load((ROOT/'configs/waypoint_ppo_pilot.yaml').read_text())

@pytest.fixture
def model(settings):
    torch.set_num_threads(1)
    return w.load_for_finetuning(RELEASE/'tracker_bc.zip',settings)


def test_raw_interface_and_reload_freeze(model,tmp_path):
    with np.load(RELEASE/'training/teacher_arrays.npz',allow_pickle=False) as data:
        obs=torch.as_tensor(data['obs'][:16].copy())
    with torch.no_grad():
        forward,_,_=model.policy.forward(obs,deterministic=True)
        predicted=model.policy._predict(obs,deterministic=True)
    assert torch.equal(forward,predicted)
    for name,p in model.policy.named_parameters():
        assert p.requires_grad == (name in ('tracking_gain_raw','tracking_bias') or
              name.startswith('mlp_extractor.value_net.') or name.startswith('value_net.'))
    model.save(tmp_path/'model.zip')
    loaded=PPO.load(tmp_path/'model.zip',device='cpu')
    assert isinstance(loaded.policy,w.RawWaypointPolicy)
    assert not loaded.policy.tracking_velocity.requires_grad
    assert not loaded.policy.log_std.requires_grad
    assert torch.equal(loaded.policy.tracking_gain_raw,model.policy.tracking_gain_raw)
    assert model.learning_rate==1e-5 and model.n_epochs==3 and model.batch_size==100
    assert model.target_kl==.005 and np.isclose(model.clip_range(1),.05)


def test_wrapper_reward_uses_same_waypoint():
    env=w.WaypointPPOEnv(load_config(RELEASE/'config.yaml'))
    scene=json.loads((RELEASE/'scenes/original_validation_81009.json').read_text())
    try:
        obs,_=env.reset(options={'scenario':scene,'scene_seed':81009})
        assert obs.shape==(77,) and env.planning_success
        q,_=env.env.joint_state();ref=env.ref.copy()
        before=np.linalg.norm(ref-q)
        obs,reward,done,truncated,info=env.step(np.clip((ref-q)/env.env.max_increment,-1,1))
        after=np.linalg.norm(ref-env.env.joint_state()[0])
        expected=2*(before-after)/env.env.max_increment
        assert np.isclose(info['reward_waypoint_progress'],expected)
        assert np.isclose(reward,info['reward_base']+expected)
        assert np.isfinite(obs).all()
    finally:env.close()


def test_planning_failure_ends_without_physics(monkeypatch):
    class Failure:
        def __init__(self,*_):self.planning_success=False;self.planning_time=0
    monkeypatch.setattr(w,'ClassicalController',Failure)
    env=w.WaypointPPOEnv(load_config(RELEASE/'config.yaml'))
    scene=json.loads((RELEASE/'scenes/original_validation_81009.json').read_text())
    try:
        env.reset(options={'scenario':scene,'scene_seed':81009})
        q=env.env.joint_state()[0].copy()
        _,reward,done,truncated,info=env.step(np.ones(7))
        assert done and not truncated and reward==0 and info['planning_failure']
        assert not info['success'] and env.env.steps==0 and env.env.sim_time==0
        assert np.array_equal(q,env.env.joint_state()[0])
        with pytest.raises(RuntimeError):env.step(np.zeros(7))
    finally:env.close()


def test_dynamic_config_rejected():
    c=load_config(RELEASE/'config.yaml');c['obstacles']['dynamic']['enabled']=True
    with pytest.raises(ValueError,match='static'):w.WaypointPPOEnv(c)


def test_validation_timing_and_final_reload(model,tmp_path,monkeypatch):
    seen=[]
    def comparison(candidate,*_):
        seen.append(candidate.num_timesteps)
        return []
    metrics={'ppo':{s:{'success_rate':1.,'collision_rate':0.,'successful_mean_steps':5}
                    for s in ('static','alignment_far','alignment_near')}}
    monkeypatch.setattr(w,'compare_policies',comparison)
    monkeypatch.setattr(w,'summarize',lambda _:metrics)
    cb=w.TrackerValidation(tmp_path,model,[],RELEASE,0,1000,2.)
    cb.model=model;model.num_timesteps=0;cb._on_training_start()
    model.num_timesteps=1000;cb._on_step();cb._on_rollout_end()
    assert seen==[0]
    cb._on_rollout_start();cb._on_rollout_start();assert seen==[0,1000]
    model.num_timesteps=2000;cb._on_training_end();assert seen==[0,1000,2000]
    record=json.loads((tmp_path/'progress.json').read_text())
    assert record['kind']=='final_saved_reload'


def test_timeout_metrics_conditional():
    rows=[]
    for reason,pos in [('collision',2),('timeout',.02),('success',.005)]:
        rows.append(dict(method='ppo',suite='original_validation',reason=reason,
                         success=reason=='success',position_error_m=pos,
                         orientation_error_deg=5,steps=20,reward=0))
    m=w.summarize(rows)['ppo']['static']
    assert m['timeout_mean_position_m']==.02 and m['success_rate']==1/3
