"""Preserve learned values/Adam, isolate hard layouts, reproducible noise tests."""
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl.four_ppo_continuation import load_continuation, state_digest, hard_probabilities, stochastic_evaluation
from wall_panel_rl.rl.four_priority_ppo import FourPolicyTrainingEnv
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES
from wall_panel_rl.utils.config import ROOT


class Toy(gym.Env):
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)
    def reset(self,*,seed=None,options=None):
        self.steps=0;x=np.zeros(77,np.float32);x[:7]=-.1;x[70:]=-.05
        for i in range(4):x[20+i*10:29+i*10]=.2;x[29+i*10]=1
        self.x=x;return x.copy(),{}
    def step(self,action):
        self.steps+=1
        return self.x.copy(),float(-np.square(action).mean()),self.steps==2,False,{'termination_reason':'success'}


def test_continue_preserves_trained_critic_and_adam_then_really_updates():
    torch.set_num_threads(1);env=DummyVecEnv([Toy])
    path=ROOT/'results/diagnostics/four_priority_ppo_v4_seed61/selected_model.zip'
    reference=PPO.load(path,device='cpu');spec={'seed':61,'n_steps':4,'batch_size':4,'n_epochs':1,
        'learning_rate':1e-5,'clip_range':.05,'target_kl':.005,'vf_coef':.1}
    try:
        model,meta=load_continuation(path,env,spec)
        assert meta['optimizer_state_preserved'] and meta['value_network_preserved']
        assert torch.equal(model.policy.value_net.weight,reference.policy.value_net.weight)
        assert state_digest(model.policy.optimizer.state_dict())==state_digest(reference.policy.optimizer.state_dict())
        before=copy.deepcopy(model.policy.state_dict());counter=model.num_timesteps;digest=meta['optimizer_sha256']
        model.learn(8,reset_num_timesteps=False)
        assert model.num_timesteps-counter==8 and state_digest(model.policy.optimizer.state_dict())!=digest
        assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items() if not k.startswith(('multi_adapter.',*CRITIC_PREFIXES)))
    finally:env.close()


def test_hard_sampling_only_train_pool_and_uniform_coverage():
    cases=[{'scene':{'seed':i,'obstacles':[{}]*4,'obstacle_motion':[]},'strength':[.05,.02]} for i in [1,2,3,4]]
    probabilities=hard_probabilities(cases,[2],.25)
    np.testing.assert_allclose(probabilities,[.1875,.4375,.1875,.1875])
    assert np.isclose(probabilities.sum(),1) and np.all(probabilities>0)
    with pytest.raises(ValueError):hard_probabilities(cases,[99],.25)
    env=FourPolicyTrainingEnv(Toy(),cases,excluded=[99],sampling_probabilities=[1,0,0,0])
    for _ in range(10):env.reset();assert env.case['scene']['seed']==1
    with pytest.raises(ValueError):FourPolicyTrainingEnv(Toy(),cases,sampling_probabilities=[.2,.2])
    with pytest.raises(ValueError):FourPolicyTrainingEnv(Toy(),cases,sampling_probabilities=[np.nan,0,0,1])


def test_stochastic_audit_is_per_scene_reproducible_and_restores_rng(monkeypatch):
    from wall_panel_rl.rl import four_ppo_continuation as module
    choices=[]
    class Model:
        def predict(self,observation,deterministic=True):choices.append(deterministic);return np.zeros(7),None
    def evaluate(adapter,config,settings,cases,mode):
        adapter.predict(np.zeros(77));return {'rows':[{'seed':cases[0]['scene']['seed'],
            'reason':'success' if float(torch.randn(()))>0 else 'collision'}]}
    monkeypatch.setattr(module,'evaluate_controller',evaluate)
    cases=[{'scene':{'seed':i}} for i in [11,12,13]];state=torch.get_rng_state()
    a=stochastic_evaluation(Model(),{},{},cases,100)
    b=stochastic_evaluation(Model(),{},{},cases[::-1],100)
    assert torch.equal(state,torch.get_rng_state()) and choices==[False]*6
    assert sorted(a['rows'],key=lambda r:r['seed'])==sorted(b['rows'],key=lambda r:r['seed'])


def test_continue_cli_without_run_does_not_train(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import train_expanded_four_ppo as module
    monkeypatch.setattr(module,'run',lambda *_:(_ for _ in ()).throw(AssertionError('Unexpected training')))
    monkeypatch.setattr('sys.argv',['train_expanded_four_ppo.py']);module.main()
    assert 'check_only' in capsys.readouterr().out
