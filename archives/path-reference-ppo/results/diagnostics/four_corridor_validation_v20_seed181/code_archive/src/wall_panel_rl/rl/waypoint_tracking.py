"""RRT-reference tracking with a small learned joint servo, not global planning.

Observation: original 70 features followed by normalized next joint waypoint (7).
The actor learns seven gains, velocity terms and biases. The route cursor is part
of the planner interface; it never selects actions or replaces failed rollouts.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


class RouteCursor:
    def __init__(self,path,tolerance):
        self.path=np.asarray(path,dtype=float).copy()
        if self.path.ndim!=2 or self.path.shape[1]!=7 or len(self.path)<1 or not np.isfinite(self.path).all():
            raise ValueError('Expected a finite nonempty joint route (N,7)')
        self.index=min(1,len(self.path)-1);self.tolerance=float(tolerance)

    def reference(self,q):
        while self.index<len(self.path)-1 and np.linalg.norm(self.path[self.index]-q)<self.tolerance:
            self.index+=1
        return self.path[self.index].copy()


def reference_observation(observation,qref,lower,upper):
    waypoint=2*(np.asarray(qref)-np.asarray(lower))/(np.asarray(upper)-np.asarray(lower))-1
    return np.r_[observation,waypoint].astype(np.float32)


def bounded_action(q,action,lower,upper,increment,margin=.01):
    lo=np.clip((lower+margin-q)/increment,-1,1);hi=np.clip((upper-margin-q)/increment,-1,1)
    return np.maximum(np.minimum(action,hi),lo).astype(np.float32)


class WaypointTrackingPolicy(ActorCriticPolicy):
    def __init__(self,*args,joint_lower,joint_upper,joint_increment,joint_margin=.01,initial_gain=.7,**kwargs):
        self.joint_lower=list(joint_lower);self.joint_upper=list(joint_upper)
        self.joint_increment=float(joint_increment);self.joint_margin=float(joint_margin);self.initial_gain=float(initial_gain)
        super().__init__(*args,**kwargs)
        if self.features_dim!=77:raise ValueError('Expected original 70 features plus 7 waypoint coordinates')
        self.tracking_gain_raw=nn.Parameter(torch.full((7,),float(np.log(np.expm1(initial_gain)))))
        self.tracking_bias=nn.Parameter(torch.zeros(7));self.tracking_velocity=nn.Parameter(torch.zeros(7))
        # The generic MLP actor is unused; only the structured servo is learned.
        for module in (self.mlp_extractor.policy_net,self.action_net):
            for parameter in module.parameters():parameter.requires_grad_(False)
        self.log_std.requires_grad_(False)
        schedule=kwargs.get('lr_schedule',args[2] if len(args)>2 else None)
        self.optimizer=self.optimizer_class(self.parameters(),lr=schedule(1),**self.optimizer_kwargs)

    def tracking_mean(self,obs):
        span=torch.as_tensor(self.joint_upper,dtype=obs.dtype,device=obs.device)-torch.as_tensor(self.joint_lower,dtype=obs.dtype,device=obs.device)
        error=(obs[:,70:77]-obs[:,:7])*span*.5/self.joint_increment
        return (F.softplus(self.tracking_gain_raw)*error+self.tracking_bias-self.tracking_velocity*obs[:,7:14]).clamp(-1,1)

    def get_distribution(self,obs):
        return self.action_dist.proba_distribution(self.tracking_mean(obs),self.log_std)

    def forward(self,obs,deterministic=False):
        features=self.extract_features(obs);dist=self.get_distribution(obs)
        action=dist.get_actions(deterministic=deterministic)
        value=self.value_net(self.mlp_extractor.forward_critic(features))
        return action,value,dist.log_prob(action)

    def evaluate_actions(self,obs,actions):
        dist=self.get_distribution(obs);features=self.extract_features(obs)
        return self.value_net(self.mlp_extractor.forward_critic(features)),dist.log_prob(actions),dist.entropy()

    def _predict(self,observation,deterministic=False):
        action=self.get_distribution(observation).get_actions(deterministic=deterministic)
        lower=torch.as_tensor(self.joint_lower,dtype=observation.dtype,device=observation.device)
        upper=torch.as_tensor(self.joint_upper,dtype=observation.dtype,device=observation.device)
        q=lower+(observation[:,:7]+1)*.5*(upper-lower)
        lo=((lower+self.joint_margin-q)/self.joint_increment).clamp(-1,1)
        hi=((upper-self.joint_margin-q)/self.joint_increment).clamp(-1,1)
        return torch.maximum(torch.minimum(action,hi),lo)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),joint_lower=self.joint_lower,joint_upper=self.joint_upper,
            joint_increment=self.joint_increment,joint_margin=self.joint_margin,initial_gain=self.initial_gain)
