"""Gated motion curriculum, genuine PPO buffers and physically replayed teachers."""
import copy
import json
from types import SimpleNamespace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl import dynamic_four_training as module
from wall_panel_rl.rl.episode_imitation import AuxiliaryImitation
from wall_panel_rl.rl.four_ppo_continuation import load_continuation


@pytest.fixture
def setup():
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_ppo.yaml').read_text())
    p=json.loads((ROOT/spec['preparation_report']/'protocol.json').read_text())
    cases=json.loads((ROOT/spec['data_root']/'train_cases.json').read_text())[:2]
    return spec,p,cases


def test_speed_changes_preserve_scene_phase_geometry_and_immutable_reference(setup):
    _,_,cases=setup;a=copy.deepcopy(cases[0]);b=module.case_at_speed(a,.005)
    assert a==cases[0] and b['task']=='dynamic' and b['strength']==[.05,.005]
    for key in ('initial_q','reference_joint_path','obstacles'):assert a['scene'][key]==b['scene'][key]
    for x,y in zip(a['scene']['obstacle_motion'],b['scene']['obstacle_motion']):
        for key in ('anchor','extent','phase'):assert x[key]==y[key]
        np.testing.assert_allclose(np.array(x['omega'])*.25,y['omega'])
    frozen=module.case_at_speed(a,.02,static=True)
    assert frozen['task']=='static' and frozen['scene']['obstacles']==a['scene']['obstacles'] and not frozen['scene']['obstacle_motion']


def test_curriculum_only_changes_future_episodes_and_rng_streams_are_reproducible(setup):
    spec,p,cases=setup;spec=copy.deepcopy(spec);spec['static_probability']=0.
    a=module.DynamicFourBankEnv(p['configuration'],p['evaluation_settings'],cases,spec,seed=61)
    b=module.DynamicFourBankEnv(p['configuration'],p['evaluation_settings'],cases,spec,seed=61)
    try:
        x,_=a.reset();y,_=b.reset();np.testing.assert_array_equal(x,y)
        state=module.planning_snapshot(a);motions=copy.deepcopy(a.last_scene['obstacle_motion']);a.set_curriculum(2)
        module.assert_planning_paused(a,state)
        assert a.speed==.005 and a.last_scene['obstacle_motion']==motions and a.episode_level==0
        b.set_curriculum(2)
        x,_=a.reset();y,_=b.reset();np.testing.assert_array_equal(x,y)
        assert a.speed==.02 and a.episode_level==2 and a.last_scene['obstacle_motion']!=motions
        with pytest.raises(ValueError,match='cannot override'):a.reset(options={'scenario':cases[0]['scene']})
    finally:a.close();b.close()


def test_training_bank_refuses_validation_seeds(setup):
    spec,p,cases=setup
    with pytest.raises(ValueError,match='split'):
        module.DynamicFourBankEnv(p['configuration'],p['evaluation_settings'],cases,spec,seed=61,excluded=[cases[0]['scene']['seed']])


def test_advancement_requires_current_target_and_static_nonregression():
    m={'episodes':64,'success':50,'collision':12,'timeout':2,'joint_limit':0}
    assert module.curriculum_gate(m,m,m,m,m,m)['passed']
    bad=dict(m,success=49,collision=13)
    assert not module.curriculum_gate(bad,m,m,m,m,m)['passed']
    assert not module.curriculum_gate(m,m,bad,m,m,m)['passed']
    assert not module.curriculum_gate(m,m,m,m,bad,m)['passed']


def test_actual_dynamic_rrt_teacher_requires_timestamps_and_fresh_physical_replay(setup):
    _,p,_=setup
    case=json.loads((ROOT/'results/diagnostics/four_slow_dynamic_preflight_v1_seed61/cases.json').read_text())[0]
    env=module.HybridFourEnv(p['configuration'],[],p['evaluation_settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        ref=env.cursor.path.copy();row,pairs=module.verified_expert_episode(env,int(case['scene']['seed'])+100000,obs)
        assert row['reason']=='success' and row['accepted_for_imitation']
        assert row['pre_verified'] and row['post_time_verified'] and row['physical_replay_verified']
        assert row['replay_steps']==row['steps']==len(pairs) and row['replay_max_error']<=1e-6
        assert row['dynamic_planning_mode']=='timed' and np.array_equal(ref,env.cursor.path)
        assert all(x.shape==(77,) and y.shape==(7,) for x,y in pairs)
    finally:env.close()


class Fake(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)
    def __init__(self):self.serial=0;self.env=SimpleNamespace(steps=0);self.task='dynamic';self.speed=.005;self.amplitude=.05;self.episode_level=0
    def reset(self,*,seed=None,options=None):
        self.serial+=1;self.last_scene={'seed':self.serial};self.env.steps=0
        return np.full(77,self.serial,np.float32),{}
    def step(self,action):
        self.env.steps+=1;done=self.env.steps==3
        return np.full(77,self.serial,np.float32),1.,done,False,{'termination_reason':'success' if done else 'running',
            'position_error':.01,'orientation_error':.01,'executed_action':np.asarray(action).tolist()}


def test_dynamic_teacher_data_never_enters_ppo_actions_or_likelihoods(setup,monkeypatch,tmp_path):
    spec,_,_=setup;torch.set_num_threads(1)
    def teacher(env,seed,obs):
        pairs=[]
        for _ in range(3):
            old=obs.copy();obs,_,_,_,info=env.step(np.full(7,.9,np.float32));pairs.append((old,np.full(7,.9,np.float32)))
        return dict(module.episode_record(env,info,'rrt',3.),accepted_for_imitation=True,replay_steps=3),pairs
    monkeypatch.setattr(module,'verified_expert_episode',teacher)
    wrapped=module.DynamicFourEpisodeMixture(Fake(),seed=0);env=DummyVecEnv([lambda:wrapped]);audits=[]
    class Audit(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_end(self):
            b=self.model.rollout_buffer;x=torch.as_tensor(b.observations.reshape(-1,77));a=torch.as_tensor(b.actions.reshape(-1,7))
            with torch.no_grad():_,logp,_=self.model.policy.evaluate_actions(x,a)
            np.testing.assert_allclose(logp.numpy(),b.log_probs.ravel(),rtol=1e-5,atol=1e-5)
            teacher_seeds={r['seed'] for r in wrapped.records if r['controller']=='rrt'}
            assert teacher_seeds.isdisjoint(b.observations[:,0,0].tolist());audits.append(True)
    try:
        tiny=dict(spec,n_steps=4,batch_size=4,n_epochs=1)
        model,_=load_continuation(ROOT/spec['source_model'],env,tiny);start=model.num_timesteps
        aux=AuxiliaryImitation(tmp_path,dict(spec['auxiliary_bc'],checkpoint_every=4),61)
        model.learn(8,reset_num_timesteps=False,callback=[Audit(),aux])
        assert model.num_timesteps-start==8 and len(audits)==2 and len(aux.pool)>0
        assert aux.accepted+aux.rejected==1
    finally:env.close()


def test_default_train_entry_only_checks_and_selection_guards_static_retention(setup,monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import train_four_dynamic_ppo as script
    monkeypatch.setattr(script,'train',lambda *a:(_ for _ in ()).throw(AssertionError('Unexpected training')))
    monkeypatch.setattr('sys.argv',['train_four_dynamic_ppo.py']);script.main();assert 'check_only' in capsys.readouterr().out
    m={'success':50,'collision':12,'timeout':2,'joint_limit':0}
    assert script.select_score(m,m,m,m,m,m)[0]==1
    assert script.select_score(dict(m,success=55,collision=7),m,m,m,dict(m,success=49),m)[0]==0
