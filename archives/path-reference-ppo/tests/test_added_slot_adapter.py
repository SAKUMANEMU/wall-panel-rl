"""Guarantee original one-static actor survives initialization and gradient steps."""
import copy
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.added_slot_adapter import configure_adapter,original_actor_preserved,ADAPTER_WEIGHTS


def test_zero_initialized_new_columns_preserve_single_obstacle_action():
    torch.set_num_threads(1)
    model=PPO.load(ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/selected_model.zip',device='cpu')
    source=copy.deepcopy(model.policy);obs=torch.zeros(8,77)
    obs[:,:20]=torch.randn(8,20)*.1;obs[:,70:]=.02;obs[:,20:29]=.1;obs[:,29]=1
    with torch.no_grad():before=source.tracking_mean(obs).clone()
    configure_adapter(model.policy,initialize=True)
    with torch.no_grad():after=model.policy.tracking_mean(obs)
    assert torch.equal(before,after) and original_actor_preserved(source,model.policy)


def test_bc_and_ppo_style_updates_cannot_change_original_actor_connections():
    torch.set_num_threads(1)
    model=PPO.load(ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/selected_model.zip',device='cpu')
    source=copy.deepcopy(model.policy);configure_adapter(model.policy,initialize=True,train_critic=True)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-3)
    before={n:p.detach().clone() for n,p in model.policy.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad()
        # Nonzero gradients everywhere test masking and requires_grad together.
        loss=sum(p.sum() for p in model.policy.parameters() if p.requires_grad)
        loss.backward();optimizer.step()
    assert original_actor_preserved(source,model.policy)
    assert any(not torch.equal(before[n][:,30:70],dict(model.policy.named_parameters())[n][:,30:70]) for n in ADAPTER_WEIGHTS)
    assert any(not torch.equal(before[n],p) for n,p in model.policy.named_parameters() if n.startswith('value_net.'))


def test_real_tiny_ppo_update_and_reload_preserve_original_actor(tmp_path):
    class Toy(gym.Env):
        observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
        action_space=spaces.Box(-1,1,(7,),np.float32)
        def obs(self):
            x=np.zeros(77,np.float32);x[20:29]=.1;x[30:39]=.2;x[29]=x[39]=1;x[70:]=.02
            return x
        def reset(self,*,seed=None,options=None):self.steps=0;return self.obs(),{}
        def step(self,action):
            self.steps+=1
            return self.obs(),float(1-.01*np.square(action).sum()),self.steps==4,False,{}
    torch.set_num_threads(1);env=DummyVecEnv([Toy])
    try:
        model=PPO.load(ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/selected_model.zip',
            env=env,device='cpu',n_steps=4,batch_size=4,n_epochs=1,learning_rate=1e-3)
        source=copy.deepcopy(model.policy)
        configure_adapter(model.policy,initialize=True,train_critic=True)
        model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),lr=1e-3,**model.policy.optimizer_kwargs)
        start=model.num_timesteps;model.learn(8,reset_num_timesteps=False)
        assert model.num_timesteps-start==8 and original_actor_preserved(source,model.policy)
        model.save(tmp_path/'model.zip')
        assert original_actor_preserved(source,PPO.load(tmp_path/'model.zip',device='cpu').policy)
    finally:env.close()
