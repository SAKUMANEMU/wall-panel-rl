"""Read-only interface audit: forward passes only, no learn/backward/optimizer step."""
import _bootstrap
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from wall_panel_rl.rl.waypoint_tracking import bounded_action


def main():
    torch.set_num_threads(1)
    release=Path('releases/path_reference_bc_20261025')
    checkpoint=release/'tracker_bc.zip'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model=PPO.load(checkpoint,device='cpu')
    policy=model.policy
    before={n:t.detach().clone() for n,t in policy.state_dict().items()}
    with np.load(release/'training/teacher_arrays.npz',allow_pickle=False) as data:
        observations=torch.as_tensor(data['obs'].copy())
        labels=torch.as_tensor(data['actions'].copy())
        offsets=data['offsets'].copy()
    with torch.no_grad():
        means=policy.tracking_mean(observations)
        mse=float(torch.mean((means-labels)**2))
        altered=observations[:100].clone()
        altered[:,14:70]+=10
        insensitive=bool(torch.equal(policy.tracking_mean(altered),means[:100]))
        # Valid joint state inside limits, but closer than the 0.01-rad safety margin.
        probe=observations[:1].clone()
        lower=np.array(policy.joint_lower);upper=np.array(policy.joint_upper)
        q=upper-.001
        probe[0,:7]=torch.as_tensor(2*(q-lower)/(upper-lower)-1,dtype=probe.dtype)
        probe[0,7:14]=0
        probe[0,70:77]=1
        raw,values,logprob=policy.forward(probe,deterministic=True)
        deployed=policy._predict(probe,deterministic=True)
        guarded=bounded_action(q,raw.numpy()[0],lower,upper,policy.joint_increment,policy.joint_margin)
        evaluated=policy.evaluate_actions(probe,raw)
    assert model.observation_space.shape==(77,) and model.action_space.shape==(7,)
    assert np.allclose(deployed.numpy()[0],guarded,atol=2e-5)
    assert torch.isfinite(values).all() and torch.isfinite(logprob).all()
    assert all(torch.isfinite(x).all() for x in evaluated)
    assert all(torch.equal(before[n],t) for n,t in policy.state_dict().items())
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    report={
        'model_sha256':digest,'model_unchanged':True,'training_updates_executed':0,
        'observation_dim':77,'action_dim':7,'teacher_episodes':len(offsets)-1,
        'teacher_transitions':len(observations),'teacher_mse':mse,
        'loaded_trainable_parameters':{n:p.numel() for n,p in policy.named_parameters() if p.requires_grad},
        'loaded_frozen_parameters':{n:p.numel() for n,p in policy.named_parameters() if not p.requires_grad},
        'actor_ignores_target_and_obstacles_features_14_to_69':insensitive,
        'std':policy.log_std.detach().exp().tolist(),
        'near_limit_forward_action':raw.numpy()[0].tolist(),
        'near_limit_predict_action':deployed.numpy()[0].tolist(),
        'forward_predict_max_difference':float(torch.max(torch.abs(raw-deployed))),
        'predict_matches_external_guard':True,
        'loaded_ppo_hyperparameters':{k:getattr(model,k) for k in ['n_steps','batch_size','learning_rate','n_epochs','target_kl','ent_coef']},
    }
    output=Path('results/diagnostics/path_reference_local_20261026/interface_audit.json')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
