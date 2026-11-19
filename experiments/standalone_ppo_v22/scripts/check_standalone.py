"""Check corrective labels, student gradients and anchored PPO end to end."""
import _bootstrap
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from run_adaptation import initialize_worker,Status,save,digest
from run_standalone import collect,Supervised,converted,AnchoredPPO
from wall_panel_rl.submission import load_json,CaseBankEnv
from wall_panel_rl.utils.config import ROOT

def main():
    torch.set_num_threads(1);initialize_worker()
    out=ROOT/'checks_standalone';out.mkdir(exist_ok=True)
    source=ROOT/'models/current_ppo.zip';original=digest(source)
    case=load_json(ROOT/'data/train_cases.json')[0]
    rows,x,y,w=collect((str(source),case,True))
    save(out/'teacher_check.json',rows)
    assert len(x) and np.isfinite(x).all() and np.isfinite(y).all()
    data={'observations':x,'actions':y,'weights':w,'scene_ids':np.full(len(x),case['scene']['seed'])}
    student=Supervised(source,data,1e-5,20261112)
    losses=[student.update() for _ in range(4)];student.verify()
    student.model.save(out/'bc_smoke.zip')
    env=DummyVecEnv([lambda:CaseBankEnv([case],reward_scale=.01)])
    try:
        model=converted(out/'bc_smoke.zip',env=env,n_steps=256,batch_size=64,n_epochs=3,
                        learning_rate=3e-6,clip_range=.05,target_kl=.005,tensorboard_log=None)
        model.policy.configure_learning(train_critic=True)
        with torch.no_grad():model.policy.log_std.fill_(np.log(.05))
        model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),lr=3e-6,**model.policy.optimizer_kwargs)
        model.ep_info_buffer=None;model.ep_success_buffer=None
        callback=AnchoredPPO(data,Status(out),out/'anchor_updates.jsonl',20261112)
        before=model.num_timesteps
        model.learn(256,reset_num_timesteps=False,callback=callback)
        model.save(out/'ppo_smoke.zip')
        reload=converted(out/'ppo_smoke.zip')
        assert model.num_timesteps-before==256
        assert all(torch.isfinite(p).all() for p in reload.policy.parameters())
        assert digest(source)==original
        save(out/'summary.json',{'passed':True,'training_seed':case['scene']['seed'],
            'labels':len(x),'bc_updates':4,'ppo_steps':256,
            'auxiliary_accepted':callback.accepted,'auxiliary_rejected':callback.rejected,
            'source_unchanged':True,'validation_or_test_used_for_training':False})
        print('STANDALONE INTEGRATION CHECK PASSED',flush=True)
    finally:env.close()
if __name__=='__main__':main()
