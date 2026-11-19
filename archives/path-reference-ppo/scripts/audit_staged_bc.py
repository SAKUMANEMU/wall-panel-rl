"""Read-only model/data audit and deterministic rollouts; never optimize weights."""
import _bootstrap
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from staged_avoidance import resolve
from wall_panel_rl.rl.avoidance_curriculum import (
    initialize_gated, StagedAvoidanceEnv)
from wall_panel_rl.rl.avoidance_imitation import imitation_split
from wall_panel_rl.rl.waypoint_tracking import WaypointTrackingPolicy, bounded_action


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def compare_actions(model, observations, labels):
    with torch.no_grad():
        predicted=model.policy.tracking_mean(torch.as_tensor(observations)).numpy()
    error=predicted-labels
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))), max_abs=float(np.max(np.abs(error))),
        p95_abs=float(np.percentile(np.abs(error),95)),
        action_mean=predicted.mean(0).tolist(), label_mean=labels.mean(0).tolist())


def run(s, output):
    output.mkdir(parents=True,exist_ok=False)
    data=Path(s['data']);stage=Path(s['stage_output'])
    config=json.loads((data/'stage_metadata.json').read_text())['environment_config']
    paths={'source':Path(s['source_model']), 'bc':stage/'bc/model.zip',
           'ppo_final':stage/'ppo/final_model.zip'}
    models={k:PPO.load(v,device='cpu') for k,v in paths.items()}
    models['gated_initial']=initialize_gated(paths['source'],s)
    report={'training_performed':False,'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*paths.values(),data/'teacher/teacher.npz']}}
    with np.load(data/'teacher/teacher.npz',allow_pickle=False) as z:
        obs=z['obs'].copy();actions=z['actions'].copy();raw=z['raw_actions'].copy()
        offsets=z['offsets'].copy();seeds=z['seeds'].copy();tasks=z['tasks'].copy()
    train,held=imitation_split(offsets,seeds,s['seed'])
    groups={'all':np.arange(len(actions))}
    for name,episodes in [('train',train),('heldout',held),
            ('alignment',np.flatnonzero(tasks=='alignment')),
            ('avoidance',np.flatnonzero(tasks=='avoidance'))]:
        groups[name]=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in episodes])
    near=np.flatnonzero(np.linalg.norm(obs[:,14:17],axis=1)<.1)
    if len(near):groups['position_under_10cm']=near
    report['dataset']={'episodes':len(seeds),'transitions':len(obs),
        'raw_vs_executed_max_abs':float(np.max(np.abs(raw-actions))),
        'projection_changed_fraction':float(np.mean(np.any(np.abs(raw-actions)>1e-6,axis=1))),
        'label_action_range':[float(actions.min()),float(actions.max())],
        'group_rows':{k:len(v) for k,v in groups.items()}}
    report['teacher_forced_action_errors']={name:{group:compare_actions(model,obs[idx],actions[idx])
        for group,idx in groups.items()} for name,model in models.items()}
    checks={}
    x=torch.as_tensor(obs)
    for name,model in models.items():
        with torch.no_grad():
            direct=model.policy.tracking_mean(x)
            forward,_,logp=model.policy(x,deterministic=True)
            _,evaluated_logp,_=model.policy.evaluate_actions(x,forward)
        predicted=model.predict(obs,deterministic=True)[0]
        with tempfile.TemporaryDirectory(prefix='bc-interface-') as tmp:
            p=Path(tmp)/'reload.zip';model.save(p);reloaded=PPO.load(p,device='cpu')
            roundtrip=reloaded.predict(obs,deterministic=True)[0]
        checks[name]={'direct_vs_predict_max_abs':float(np.max(np.abs(direct.numpy()-predicted))),
            'forward_vs_predict_max_abs':float(np.max(np.abs(forward.numpy()-predicted))),
            'forward_vs_evaluate_logprob_max_abs':float((logp-evaluated_logp).abs().max()),
            'save_reload_action_max_abs':float(np.max(np.abs(roundtrip-predicted))),
            'action_std':model.policy.log_std.exp().detach().numpy().tolist()}
    report['interfaces']=checks
    bc=models['bc']
    with torch.no_grad():
        prior=WaypointTrackingPolicy.tracking_mean(bc.policy,x)
        gate=torch.sigmoid(bc.policy.prior_gate(x))
        residual=bc.policy.residual_scale*torch.tanh(bc.policy.residual(x))
    report['bc_components']={'gate_min_mean_max':[float(gate.min()),float(gate.mean()),float(gate.max())],
        'prior_mean_abs':float(prior.abs().mean()),'residual_mean_abs':float(residual.abs().mean()),
        'residual_max_abs':float(residual.abs().max())}
    dump(output/'audit.json',report)
    print('Offline action interfaces checked',flush=True)

    # Replaying stored labels needs no expert/planner and tests observation/action pairing.
    attempts=json.loads((data/'teacher/attempts.json').read_text())
    successful=[a for a in attempts if a['reason']=='success']
    if len(successful)!=len(seeds):raise ValueError('Teacher episode/attempt mismatch')
    env=StagedAvoidanceEnv(config,[],s)
    replays=[];projection_error=0.;normalization_error=0.
    try:
        for i,attempt in enumerate(successful):
            observation,_=env.reset(options={'scenario':attempt['scene'],
                'task':'alignment' if tasks[i]=='alignment' else s['active_stage']['mode'],
                'strength':[s['active_stage']['amplitude'],s['active_stage']['speed']]})
            max_error=0.;q_error=0.;count=0;reason='running'
            for j in range(offsets[i],offsets[i+1]):
                max_error=max(max_error,float(np.max(np.abs(observation-obs[j]))))
                q,_=env.env.joint_state()
                q_from_obs=env.env.lower+(observation[:7]+1)*.5*(env.env.upper-env.env.lower)
                q_error=max(q_error,float(np.max(np.abs(q_from_obs-q))))
                expected=bounded_action(q,actions[j],env.env.lower,env.env.upper,env.env.max_increment,s['joint_margin'])
                observation,_,done,truncated,info=env.step(actions[j]);count+=1
                projection_error=max(projection_error,float(np.max(np.abs(expected-info['executed_action']))))
                reason=info['termination_reason']
                if done or truncated:break
            normalization_error=max(normalization_error,q_error)
            replays.append({'episode':i,'seed':int(seeds[i]),'task':str(tasks[i]),
                'reason':reason,'steps':count,'saved_steps':int(offsets[i+1]-offsets[i]),
                'max_observation_error':max_error,'q_normalization_error_rad':q_error})
        dump(output/'teacher_replays.json',replays)
        report['replay']={'episodes':len(replays),'success':sum(r['reason']=='success' for r in replays),
            'max_observation_error':max(r['max_observation_error'] for r in replays),
            'q_normalization_error_rad':normalization_error,'execution_projection_max_abs':projection_error}
        dump(output/'audit.json',report)
        print('Teacher replay:',report['replay'],flush=True)

        cases=json.loads((data/'validation_staged.json').read_text())
        cases=[c for c in cases if c['suite']!='five_full_diagnostic']
        # Add successful teacher scenarios to distinguish validation shift from closed-loop drift.
        for task in ('alignment','avoidance'):
            selected=[(i,a) for i,a in enumerate(successful) if tasks[i]==task][:10]
            for i,a in selected:
                cases.append({'suite':'teacher_'+task,'scene':a['scene'],
                    'task':'alignment' if task=='alignment' else s['active_stage']['mode'],
                    'strength':[s['active_stage']['amplitude'],s['active_stage']['speed']]})
        rows=[]
        for name,model in models.items():
            for case in cases:
                observation,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
                total=0.;first=None;max_action=0.;last_obs=None
                while True:
                    action=model.predict(observation,deterministic=True)[0]
                    if first is None:first=action.tolist()
                    max_action=max(max_action,float(np.max(np.abs(action))));last_obs=observation.copy()
                    observation,reward,done,truncated,info=env.step(action);total+=reward
                    if done or truncated:break
                rows.append({'model':name,'suite':case['suite'],'seed':case['scene']['seed'],
                    'reason':info['termination_reason'],'steps':env.env.steps,
                    'position_m':float(info['position_error']),
                    'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                    'first_action':first,'last_action':action.tolist(),'max_abs_action':max_action,
                    'last_waypoint_error':(last_obs[70:77]-last_obs[:7]).tolist()})
            print('Closed-loop evaluated',name,flush=True)
        dump(output/'rollouts.json',rows)
        summary=[]
        for name in models:
            for suite in sorted({c['suite'] for c in cases}):
                group=[r for r in rows if r['model']==name and r['suite']==suite]
                summary.append({'model':name,'suite':suite,'episodes':len(group),
                    **{key:sum(r['reason']==key for r in group) for key in ('success','collision','joint_limit','timeout')}})
        with (output/'rollout_summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
        report['closed_loop']=summary;dump(output/'audit.json',report)
        print(json.dumps(summary,indent=2),flush=True)
    finally:env.close()
    if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in report['hashes'].items()):
        raise AssertionError('Input artifact modified')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    # Guard against accidentally introducing any optimization into a diagnostic.
    def forbidden(*args,**kwargs):raise RuntimeError('Training is forbidden in this audit')
    PPO.learn=forbidden;torch.optim.Adam.step=forbidden
    root=yaml.safe_load(Path('configs/staged_avoidance_v2.yaml').read_text())
    settings=resolve(root,'one_static')
    run(settings,Path(args.output))


if __name__=='__main__':main()
