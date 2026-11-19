"""Run a tiny physical PPO update before the long V32 adaptation pipeline."""
import _bootstrap
import argparse
import copy
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_json,load_runtime
from wall_panel_rl.rl.variable_weaving import VariableCaseBankEnv
from wall_panel_rl.rl.known_route_policy import KnownRoutePolicy
from run_adaptation import save,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    runtime=load_runtime()
    spec=yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())['training']
    cases=[]
    for count in [2,3,4,5]:
        case=copy.deepcopy(load_json(ROOT/f'checks_site/count{count}.json'))
        for motion in case['scene']['obstacle_motion']:
            motion['speed_m_s']=min(motion['speed_m_s'],.18)
        case['strength'][1]=.18
        cases.append(case)
    source=ROOT/'models/current_ppo.zip';before=digest(source)
    env=DummyVecEnv([lambda:VariableCaseBankEnv(runtime,cases,spec,20261119)])
    try:
        model=PPO.load(source,env=env,device='cpu',custom_objects={'policy_class':KnownRoutePolicy},
            n_steps=32,batch_size=16,n_epochs=1,learning_rate=3e-6,seed=20261119,
            tensorboard_log=None)
        model.policy.configure_learning(train_critic=True)
        with torch.no_grad():model.policy.log_std.fill_(np.log(.03))
        model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),lr=3e-6,
            **model.policy.optimizer_kwargs)
        frozen={n:p.detach().clone() for n,p in model.policy.named_parameters() if not p.requires_grad}
        trained={n:p.detach().clone() for n,p in model.policy.named_parameters() if p.requires_grad}
        old_steps=model.num_timesteps;old_updates=model._n_updates
        model.learn(64,reset_num_timesteps=False)
        parameters=dict(model.policy.named_parameters())
        assert model.num_timesteps-old_steps==64 and model._n_updates>old_updates
        assert all(torch.isfinite(p).all() for p in parameters.values())
        assert all(torch.equal(parameters[n],p) for n,p in frozen.items())
        assert any(not torch.equal(parameters[n],p) for n,p in trained.items())
        assert model.observation_space.shape==(167,) and model.action_space.shape==(7,)
        assert digest(source)==before
        save(args.output/'results.json',{'passed':True,'physical_on_policy_steps':64,
            'optimizer_updates':model._n_updates-old_updates,'frozen_prior_unchanged':True,
            'source_unchanged':True,'speed_range_m_s':[.04,.18],
            'smoke_only':True,'included_in_formal_training_or_evaluation':False})
    finally:env.close()


if __name__=='__main__':main()
