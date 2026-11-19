"""One real teacher episode followed by BC and a 256-step on-policy update."""
import _bootstrap
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from run_adaptation import initialize_worker, collect_one, train_bc, Status, digest, save
from wall_panel_rl.submission import load_json, load_agent, CaseBankEnv
from wall_panel_rl.utils.config import ROOT

def main():
    torch.set_num_threads(1); initialize_worker()
    out=ROOT/'checks';out.mkdir(exist_ok=True);(out/'models').mkdir(exist_ok=True)
    source=ROOT/'models/current_ppo.zip';sha=digest(source)
    case=load_json(ROOT/'data/train_cases.json')[1]
    logs,x,y=collect_one(case)
    save(out/'teacher_smoke.json',logs)
    assert len(x)>0,'The real teacher did not succeed on the development smoke case'
    assert x.shape[1:]==(77,) and y.shape[1:]==(7,)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.abs(y).max()<=1.000001
    labels={'observations':x,'actions':y,'scene_ids':np.full(len(x),case['scene']['seed'])}
    path=train_bc(source,labels,out,20261112,Status(out),updates=4)
    env=DummyVecEnv([lambda:CaseBankEnv([case],reward_scale=.01)])
    try:
        env.seed(20261112)
        model=load_agent(path,env=env,n_steps=256,batch_size=64,n_epochs=3,
                         learning_rate=3e-6,tensorboard_log=None)
        model.policy.configure_learning(train_critic=True)
        model.ep_info_buffer=None;model.ep_success_buffer=None
        before=model.num_timesteps
        model.learn(256,reset_num_timesteps=False)
        model.save(out/'models/ppo_smoke.zip')
        assert model.num_timesteps-before==256
        assert digest(source)==sha
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        save(out/'smoke_summary.json',{'passed':True,'teacher_steps':len(x),'teacher_seed':case['scene']['seed'],
             'bc_updates':4,'ppo_additional_steps':256,'source_unchanged':True,
             'formal_validation_or_test_used_for_training':False})
        print('ADAPTATION SMOKE PASSED',flush=True)
    finally:env.close()
if __name__=='__main__':main()
