"""Backward-compatible 77-feature tracking prior plus learned route corrections.

Zero-initialized new output layers preserve the inherited actor and critic at
migration. Both stochastic sampling and deterministic inference use the same
route-aware mean. No planner or simulator is called from the neural policy.
"""
import numpy as np
import torch
from torch import nn
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl.standalone_policy import StandalonePolicy, ACTOR_PREFIXES
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES
from wall_panel_rl.envs.known_route_motion import OBSERVATIONS


class RouteFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space):
        super().__init__(observation_space,features_dim=77)
        self.path_projection = nn.Sequential(nn.Linear(90,128),nn.Tanh(),nn.Linear(128,77))
        nn.init.zeros_(self.path_projection[-1].weight)
        nn.init.zeros_(self.path_projection[-1].bias)

    def forward(self,observations):
        return observations[:,:77]+self.path_projection(observations[:,77:])


class KnownRoutePolicy(StandalonePolicy):
    def __init__(self,*args,route_scale=1.,**kwargs):
        self.route_scale = float(route_scale)
        kwargs['features_extractor_class'] = RouteFeatures
        super().__init__(*args,**kwargs)
        self.route_actor = nn.Sequential(nn.Linear(OBSERVATIONS,128),nn.Tanh(),
            nn.Linear(128,64),nn.Tanh(),nn.Linear(64,7))
        nn.init.zeros_(self.route_actor[-1].weight)
        nn.init.zeros_(self.route_actor[-1].bias)
        self.configure_learning(train_critic=True)
        schedule = kwargs.get('lr_schedule',args[2] if len(args)>2 else None)
        self.optimizer = self.optimizer_class(self.parameters(),lr=schedule(1),**self.optimizer_kwargs)

    def configure_learning(self,*,train_critic=False):
        for name,parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith((*ACTOR_PREFIXES,'route_actor.')) or
                (train_critic and name.startswith((*CRITIC_PREFIXES,'features_extractor.'))))

    def _route_correction(self,observations):
        strength,_,_ = self.correction_components(observations[:,:77])
        return strength*self.route_scale*torch.tanh(self.route_actor(observations))

    def tracking_mean(self,observations):
        base = super().tracking_mean(observations[:,:77])
        return (base+self._route_correction(observations)).clamp(-1,1)

    def imitation_mean(self,observations):
        return super().imitation_mean(observations[:,:77])+self._route_correction(observations)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),route_scale=self.route_scale)


class RouteSpaces(gym.Env):
    observation_space = spaces.Box(-np.inf,np.inf,(OBSERVATIONS,),np.float32)
    action_space = spaces.Box(-1,1,(7,),np.float32)
    def reset(self,*,seed=None,options=None):
        return np.zeros(OBSERVATIONS,np.float32),{}
    def step(self,action):
        raise RuntimeError('Attach the physical environment before training')


def initialize_known_routes(source,seed=20261116):
    old = PPO.load(source,device='cpu')
    model = PPO(KnownRoutePolicy,DummyVecEnv([RouteSpaces]),device='cpu',seed=seed,
        n_steps=256,batch_size=128,n_epochs=3,learning_rate=3e-6,clip_range=.05,
        target_kl=.005,ent_coef=0.,policy_kwargs=dict(old.policy_kwargs))
    missing,unexpected = model.policy.load_state_dict(old.policy.state_dict(),strict=False)
    allowed = ('route_actor.','features_extractor.','pi_features_extractor.','vf_features_extractor.')
    if unexpected or any(not name.startswith(allowed) for name in missing):
        raise ValueError(f'Unexpected model migration: {missing}, {unexpected}')
    model.num_timesteps = old.num_timesteps
    return old,model
