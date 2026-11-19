"""Verified teacher labels stay outside on-policy buffers and log likelihoods."""
from types import SimpleNamespace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl import four_episode_mix as module
from wall_panel_rl.rl.episode_imitation import AuxiliaryImitation
from wall_panel_rl.rl.four_ppo_continuation import load_continuation
from wall_panel_rl.utils.config import ROOT


class Fake(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)
    def __init__(self):self.serial=0;self.env=SimpleNamespace(steps=0,collision=None);self.task='static'
    def info(self):return {'termination_reason':'running','position_error':.1,'orientation_error':.1}
    def reset(self,*,seed=None,options=None):
        self.serial+=1;self.last_scene={'seed':self.serial};self.env.steps=0;self.cursor=SimpleNamespace(path=np.zeros((2,7)))
        return np.full(77,self.serial,np.float32),self.info()
    def step(self,action):
        self.env.steps+=1;done=self.env.steps==3;info=self.info()
        info.update(termination_reason='success' if done else 'running',executed_action=np.asarray(action).tolist())
        return np.full(77,self.serial,np.float32),1.,done,False,info


class Expert:
    planning_success=True;static_path_verified=True;planning_time=.01
    path=np.zeros((2,7));static_validation_resolution=.005
    def action(self,obs):return np.full(7,.9,np.float32)


def test_real_ppo_likelihoods_after_auxiliary_bc_remain_on_policy(monkeypatch,tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setattr(module,'make_rrt',lambda *a:Expert());monkeypatch.setattr(module,'dense_route_valid',lambda *a:True)
    wrapped=module.FourEpisodeMixture(Fake(),seed=0,reward_scale=.01)
    env=DummyVecEnv([lambda:wrapped]);audits=[]
    class Audit(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_end(self):
            b=self.model.rollout_buffer;x=torch.as_tensor(b.observations.reshape(-1,77));a=torch.as_tensor(b.actions.reshape(-1,7))
            with torch.no_grad():_,logp,_=self.model.policy.evaluate_actions(x,a)
            np.testing.assert_allclose(logp.numpy(),b.log_probs.ravel(),atol=1e-5,rtol=1e-5)
            rrt_seeds={r['seed'] for r in wrapped.records if r['controller']=='rrt'}
            assert rrt_seeds.isdisjoint(b.observations[:,0,0].tolist());audits.append(True)
    try:
        spec={'seed':61,'n_steps':4,'batch_size':4,'n_epochs':1,'learning_rate':1e-5,'clip_range':.05,'target_kl':.005,'vf_coef':.1}
        model,_=load_continuation(ROOT/'results/diagnostics/four_expanded_ppo_v5_seed61/selected_model.zip',env,spec)
        start=model.num_timesteps
        aux=AuxiliaryImitation(tmp_path,{'capacity':100,'batch_size':4,'checkpoint_every':4,'updates_per_rollout':1,
            'weight':.05,'learning_rate':1e-6,'maximum_kl':.003},61)
        model.learn(8,reset_num_timesteps=False,callback=[Audit(),aux])
        assert model.num_timesteps-start==8 and len(audits)==2
        assert aux.accepted+aux.rejected==1 and len(aux.pool)>0
        assert all(r['pre_verified'] and r['post_verified'] and r['reference_preserved'] for r in wrapped.records if r['controller']=='rrt')
        assert sum(r['steps'] for r in wrapped.records if r['controller']=='rrt')>0
    finally:env.close()


@pytest.mark.parametrize('failure',['plan','precheck','postcheck'])
def test_unverified_or_failed_expert_provides_no_bc_labels(monkeypatch,failure):
    expert=Expert()
    if failure=='plan':expert.planning_success=False
    if failure=='precheck':expert.static_path_verified=False
    monkeypatch.setattr(module,'make_rrt',lambda *a:expert)
    monkeypatch.setattr(module,'dense_route_valid',lambda *a:failure!='postcheck')
    env=module.FourEpisodeMixture(Fake(),seed=0)
    try:
        obs,_=env.reset()
        for _ in range(3):obs,_,done,_,_=env.step(np.zeros(7))
        env.reset()
        rows=[r for r in env.records if r['controller']=='rrt']
        assert rows and not env.pending and all(not r['accepted_for_imitation'] for r in rows)
        if failure!='postcheck':assert all(r['reason']=='planning_failure' and r['steps']==0 for r in rows)
    finally:env.close()


def test_one_owner_and_scaled_ppo_reward(monkeypatch):
    monkeypatch.setattr(module,'make_rrt',lambda *a:Expert());monkeypatch.setattr(module,'dense_route_valid',lambda *a:True)
    env=module.FourEpisodeMixture(Fake(),seed=8,reward_scale=.01)
    try:
        _,info=env.reset();assert info['episode_controller']=='ppo'
        assert len(env.records)==1 and env.records[0]['controller']=='rrt'
        for _ in range(3):_,reward,done,_,info=env.step(np.zeros(7));assert reward==.01
        assert done and info['episode']['r']==.03 and env.records[-1]['controller']=='ppo'
        with pytest.raises(RuntimeError):env.step(np.zeros(7))
    finally:env.close()


def test_default_cli_never_starts_training_or_evaluation(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import train_four_episode_mix as runner
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected long run')
    monkeypatch.setattr(runner,'train',forbidden);monkeypatch.setattr(runner,'evaluate',forbidden)
    monkeypatch.setattr('sys.argv',['train_four_episode_mix.py']);runner.main()
    assert 'check_only' in capsys.readouterr().out
