"""RRT episode accounting, genuine on-policy buffers, and isolated BC gradients."""
import copy
from pathlib import Path
from types import SimpleNamespace
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl import episode_imitation as module
from wall_panel_rl.utils.config import ROOT


class FakeEnv(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)
    def __init__(self,timeout=False):self.serial=0;self.env=SimpleNamespace(steps=0);self.task='static';self.timeout=timeout
    def info(self):return {'termination_reason':'running','position_error':.1,'orientation_error':.1}
    def reset(self,*,seed=None,options=None):
        self.serial+=1;self.last_scene={'seed':self.serial};self.env.steps=0
        return np.full(77,self.serial,np.float32),self.info()
    def step(self,action):
        self.env.steps+=1;done=self.env.steps==3;info=self.info()
        info.update(termination_reason=('timeout' if self.timeout else 'success') if done else 'running',
                    executed_action=np.asarray(action).tolist())
        return np.full(77,self.serial,np.float32),1.,done and not self.timeout,done and self.timeout,info


class Expert:
    planning_success=True;planning_time=.01
    def action(self,obs):return np.full(7,.9,np.float32)


class Callback(BaseCallback):
    def _on_step(self):return True


def test_rrt_never_enters_ppo_action_or_log_probability_buffer(monkeypatch,tmp_path):
    torch.set_num_threads(1)
    monkeypatch.setattr(module,'make_rrt',lambda *a:Expert())
    wrapper=module.ExpertEpisodeEnv(FakeEnv(),seed=0,journal=tmp_path/'episodes.jsonl')
    env=DummyVecEnv([lambda:wrapper])
    try:
        model=PPO('MlpPolicy',env,n_steps=4,batch_size=4,n_epochs=1,seed=0,device='cpu')
        _,callback=model._setup_learn(4,Callback())
        sampled=[]
        def record(policy,args,output):sampled.append(tuple(x.detach().numpy().copy() for x in output))
        hook=model.policy.register_forward_hook(record)
        try:assert model.collect_rollouts(env,callback,model.rollout_buffer,4)
        finally:hook.remove()
        assert model.num_timesteps==4 and len(sampled)==4
        buffer=model.rollout_buffer
        np.testing.assert_array_equal(buffer.actions[:,0,:],np.concatenate([x[0] for x in sampled]))
        np.testing.assert_array_equal(buffer.log_probs[:,0],np.concatenate([x[2] for x in sampled]))
        rrt_rows=[r for r in wrapper.records if r['controller']=='rrt']
        assert len(rrt_rows)==3 and sum(r['steps'] for r in rrt_rows)==9
        rrt_seeds={r['seed'] for r in rrt_rows}
        assert rrt_seeds.isdisjoint(buffer.observations[:,0,0].tolist())
        np.testing.assert_array_equal(buffer.episode_starts[:,0],[1,0,0,1])
        assert len(wrapper.drain_demonstrations())==9 and not wrapper.drain_demonstrations()
    finally:env.close()


def test_failed_plans_are_rrt_failures_without_demonstrations_or_fallback(monkeypatch):
    failed=Expert();failed.planning_success=False
    monkeypatch.setattr(module,'make_rrt',lambda *a:failed)
    env=module.ExpertEpisodeEnv(FakeEnv(),seed=0)
    try:
        env.reset()
        for _ in range(3):env.step(np.zeros(7))
        env.reset()
        failed_rows=[r for r in env.records if r['controller']=='rrt']
        assert len(failed_rows)==3 and all(r['reason']=='planning_failure' and r['steps']==0 for r in failed_rows)
        assert not env.pending
        assert env.started['rrt']==3 and env.started['ppo']==2
    finally:env.close()


def test_timeout_bootstrapping_and_episode_boundary_are_preserved(monkeypatch):
    env=DummyVecEnv([lambda:module.ExpertEpisodeEnv(FakeEnv(timeout=True),probability=0)])
    try:
        model=PPO('MlpPolicy',env,n_steps=4,batch_size=4,seed=0,device='cpu')
        monkeypatch.setattr(model.policy,'predict_values',lambda obs:torch.full((len(obs),1),3.))
        _,callback=model._setup_learn(4,Callback())
        model.collect_rollouts(env,callback,model.rollout_buffer,4)
        assert np.isclose(model.rollout_buffer.rewards[2,0],1+model.gamma*3)
        np.testing.assert_array_equal(model.rollout_buffer.episode_starts[:,0],[1,0,0,1])
    finally:env.close()


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.residual=torch.nn.Linear(77,7);self.prior_gate=torch.nn.Linear(77,1)
        self.value_net=torch.nn.Linear(77,1)
        torch.nn.init.zeros_(self.residual.weight);torch.nn.init.zeros_(self.residual.bias)
    def tracking_mean(self,x):return self.residual(x)+.1*torch.sigmoid(self.prior_gate(x))
    def get_distribution(self,x):
        mean=self.tracking_mean(x)
        return SimpleNamespace(distribution=torch.distributions.Normal(mean,torch.full_like(mean,.1)))
    def set_training_mode(self,mode):self.train(mode)


@pytest.mark.parametrize('reject',[False,True])
def test_auxiliary_bc_updates_only_actor_and_rolls_back_excessive_kl(tmp_path,reject):
    policy=Policy();before=copy.deepcopy(policy.state_dict())
    obs=np.zeros(77,np.float32);target=np.full(7,.2,np.float32)
    vector=SimpleNamespace(env_method=lambda method:[[(obs,target)]])
    model=SimpleNamespace(policy=policy,num_timesteps=0,device='cpu',_last_obs=obs[None],get_env=lambda:vector)
    spec={'capacity':10,'batch_size':2,'checkpoint_every':5000,'updates_per_rollout':1,'weight':.1,
          'learning_rate':1. if reject else .001,'maximum_kl':1e-12 if reject else .005}
    callback=module.AuxiliaryImitation(tmp_path,spec);callback.init_callback(model)
    callback._on_training_start();callback._on_rollout_start()
    after=policy.state_dict()
    assert all(torch.equal(before[n],after[n]) for n in before if n.startswith('value_net.'))
    if reject:
        assert callback.rejected==1 and callback.accepted==0
        assert all(torch.equal(v,after[n]) for n,v in before.items())
        assert not callback.optimizer.state_dict()['state']
    else:
        assert callback.accepted==1 and not torch.equal(before['residual.bias'],after['residual.bias'])


def test_rrt_uses_original_reference_endpoint_and_real_scene_is_not_relabelled(monkeypatch):
    import json
    torch.set_num_threads(1)
    manifest=json.loads((ROOT/'results/diagnostics/safety_first_ppo_20261102/manifest.json').read_text())
    cases=json.loads((ROOT/'data/staged_avoidance_v3_seed61/one_static/validation_staged.json').read_text())
    case=next(c for c in cases if c['suite']=='stage_target' and c['scene']['seed']==900000)
    original=copy.deepcopy(case);calls=[];real=module.ClassicalController
    def expert(env,seed,**kwargs):calls.append(kwargs['goal_q'].copy());return real(env,seed,**kwargs)
    monkeypatch.setattr(module,'ClassicalController',expert)
    result=module.evaluate_controller(None,manifest['configuration'],manifest['evaluation_settings'],[case],mode='rrt')
    assert len(calls)==1 and np.array_equal(calls[0],case['scene']['reference_joint_path'][-1])
    assert case==original and result['by_controller']['rrt']['episodes']==1
    assert result['rows'][0]['online_avoidance_planner']
    assert result['rows'][0]['reason']=='success'


def test_zero_auxiliary_updates_drain_labels_without_changing_weights_or_rng(tmp_path):
    """Ablation keeps expert execution but disables all imitation gradients."""
    policy=Policy();before=copy.deepcopy(policy.state_dict())
    obs=np.zeros(77,np.float32);target=np.full(7,.2,np.float32)
    pending=[(obs,target)]
    def drain(method):
        result=pending.copy();pending.clear();return [result]
    vector=SimpleNamespace(env_method=drain)
    model=SimpleNamespace(policy=policy,num_timesteps=0,device='cpu',_last_obs=obs[None],get_env=lambda:vector)
    spec={'capacity':10,'batch_size':2,'checkpoint_every':5000,'updates_per_rollout':0,'weight':.05,
          'learning_rate':1e-6,'maximum_kl':.003}
    callback=module.AuxiliaryImitation(tmp_path,spec);callback.init_callback(model)
    callback._on_training_start();rng=torch.get_rng_state().clone()
    callback._on_rollout_start()
    assert not pending and len(callback.pool)==1
    assert callback.accepted==callback.rejected==0
    assert all(torch.equal(v,policy.state_dict()[n]) for n,v in before.items())
    assert torch.equal(rng,torch.get_rng_state())
    assert not callback.optimizer.state_dict()['state']


def test_default_cli_only_checks_and_does_not_launch_training(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import sys
    import train_episode_rrt_ppo as script
    monkeypatch.setattr(script,'train',lambda *a:(_ for _ in ()).throw(AssertionError('Training launched')))
    monkeypatch.setattr(sys,'argv',['train_episode_rrt_ppo.py'])
    script.main();assert 'check_only' in capsys.readouterr().out


def test_two_tiny_rollouts_keep_likelihoods_on_policy_after_auxiliary_update(monkeypatch,tmp_path):
    # Eight artificial transitions exercise callbacks, not a formal experiment.
    torch.set_num_threads(1);monkeypatch.setattr(module,'make_rrt',lambda *a:Expert())
    wrapped=module.ExpertEpisodeEnv(FakeEnv(),seed=0)
    env=DummyVecEnv([lambda:wrapped]);audits=[]
    class Audit(Callback):
        def _on_rollout_end(self):
            buffer=self.model.rollout_buffer
            obs=torch.as_tensor(buffer.observations.reshape(-1,77))
            action=torch.as_tensor(buffer.actions.reshape(-1,7))
            with torch.no_grad():_,logp,_=self.model.policy.evaluate_actions(obs,action)
            np.testing.assert_allclose(logp.numpy(),buffer.log_probs.ravel(),rtol=1e-5,atol=1e-5)
            audits.append(True)
    spec={'capacity':100,'batch_size':2,'checkpoint_every':4,'updates_per_rollout':1,
          'weight':.1,'learning_rate':1e-5,'maximum_kl':.005}
    try:
        model=PPO.load(ROOT/'results/diagnostics/safety_first_ppo_20261102/selected_model.zip',
            env=env,device='cpu',n_steps=4,batch_size=4,n_epochs=1)
        start=model.num_timesteps;aux=module.AuxiliaryImitation(tmp_path,spec)
        model.learn(8,reset_num_timesteps=False,callback=[Audit(),aux])
        assert model.num_timesteps-start==8 and len(audits)==2
        assert aux.accepted+aux.rejected==1 and len(aux.pool)>0
        assert (tmp_path/'final_aux_state.pt').is_file()
    finally:env.close()
