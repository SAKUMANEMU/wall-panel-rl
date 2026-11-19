"""Read-only replay of six lost successes and saved RRT-label disagreements."""
import _bootstrap
import argparse
from collections import Counter,defaultdict
import copy
import csv
from datetime import datetime
from pathlib import Path
import json
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.rrt_validation import contacts_at_failure


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:print({'mode':'check_only','training_steps':0});return
    torch.set_num_threads(1);source=Path('results/diagnostics/four_episode_mix_v6_seed61')
    root=source.parent/'four_mix_regression_v7_seed61'
    if root.exists():raise FileExistsError('Refusing diagnostic overwrite')
    protocol=read(source/'protocol.json');spec=protocol['spec'];config=protocol['configuration'];settings=protocol['evaluation_settings']
    cases=read(Path(spec['data_root'])/'validation_cases.json');by_seed={c['scene']['seed']:c for c in cases}
    saved={n:{r['seed']:r for r in read(source/'validation'/f'ppo_{n}.json')['rows']} for n in (0,5000,10000)}
    lost=[seed for seed,row in saved[0].items() if row['reason']=='success' and saved[10000][seed]['reason']!='success']
    assert len(lost)==6
    paths={n:Path(spec['model_root'])/'checkpoints'/f'ppo_{n}.zip' for n in saved}
    hashes={str(path):sha(path) for path in paths.values()};models={n:PPO.load(path,device='cpu') for n,path in paths.items()}
    root.mkdir();dump(root/'protocol.json',{'source':str(source),'checkpoint_hashes':hashes,'validation_sha256':sha(Path(spec['data_root'])/'validation_cases.json'),
        'lost_scene_seeds':lost,'same_observation_action_comparison':True,'training_steps':0,'code_sha256':sha(__file__)})
    rows=[];env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings)
    try:
        for seed in lost:
            case=by_seed[seed]
            for n,model in models.items():
                obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
                reference=env.cursor.path.copy();frames=[]
                while True:
                    tensor=torch.as_tensor(obs[None]);q=env.env.joint_state()[0]
                    with torch.no_grad():means={k:m.policy.tracking_mean(tensor).cpu().numpy()[0] for k,m in models.items()}
                    command=model.predict(obs,deterministic=True)[0]
                    assert np.max(np.abs(command-means[n]))<=1e-6
                    frame={'step':env.env.steps,'q':q.tolist(),'reference_q':env.ref.tolist(),'reference_index':int(env.cursor.index),
                        'reference_last_index':len(env.cursor.path)-1,'raw_action':command.tolist(),
                        'same_state_action_by_checkpoint':{str(k):v.tolist() for k,v in means.items()},
                        'action_max_change_from_initial':float(np.max(np.abs(means[n]-means[0]))),
                        'reference_distance_rad':float(np.linalg.norm(q-env.ref))}
                    obs,_,done,truncated,info=env.step(command)
                    frame.update(position_m=float(info['position_error']),orientation_deg=float(np.rad2deg(info['orientation_error'])),
                        executed_action=np.asarray(info['executed_action']).tolist(),
                        clearance_capped_m=info.get('obstacle_clearance_capped_m'),
                        recovery_attempts=info.get('conditional_recovery_attempts',0),recoveries=info.get('conditional_recovery_count',0))
                    frames.append(frame)
                    if done or truncated:break
                expected=saved[n][seed];assert info['termination_reason']==expected['reason'] and env.env.steps==expected['steps']
                assert np.array_equal(reference,env.cursor.path)
                contacts=contacts_at_failure(env.env,case['scene']) if info['termination_reason']=='collision' else []
                tail=frames[-min(30,len(frames)):];start=tail[0];end=tail[-1]
                row={'seed':seed,'checkpoint':n,'reason':info['termination_reason'],'steps':env.env.steps,
                    'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                    'max_same_state_action_change':max(f['action_max_change_from_initial'] for f in frames),
                    'first_change_over_0_05_step':next((f['step'] for f in frames if f['action_max_change_from_initial']>.05),None),
                    'last30_position_progress_m':start['position_m']-end['position_m'],
                    'last30_joint_displacement_rad':float(np.linalg.norm(np.asarray(end['q'])-start['q'])),
                    'reference_index_final':int(env.cursor.index),'reference_last_index':len(env.cursor.path)-1,
                    'reference_distance_final_rad':float(np.linalg.norm(env.ref-env.env.joint_state()[0])),
                    'recovery_attempts':info.get('conditional_recovery_attempts',0),'recoveries':info.get('conditional_recovery_count',0),
                    'dominant_contact':contacts[0] if contacts else None}
                rows.append(row);dump(root/'traces'/f'{seed}_{n}.json',{'summary':row,'contacts':contacts,'frames':frames,'recovery_events':env.conditional_events})
            print('MIX_DIAG',seed,saved[10000][seed]['reason'],flush=True)
        # Identical initial observations in the old TRAIN expert pool can expose
        # conflicting route labels; this audit never adds labels to training.
        train_cases=read(Path(spec['data_root'])/'train_cases.json');initial={}
        for case in train_cases:
            x,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            initial[x.round(6).tobytes()]=case['scene']['seed']
    finally:env.close()
    aux=torch.load(Path(spec['model_root'])/'auxiliary_bc/final_aux_state.pt',map_location='cpu',weights_only=False)
    grouped=defaultdict(list)
    for x,y in aux['pool']:
        key=np.asarray(x,dtype=np.float32).round(6).tobytes()
        if key in initial:grouped[key].append(np.asarray(y,dtype=float))
    conflicts=[]
    for key,labels in grouped.items():
        if len(labels)<2:continue
        labels=np.asarray(labels);opposing=0;max_change=0.
        for i,a in enumerate(labels):
            for b in labels[i+1:]:
                opposing=max(opposing,int(np.sum((np.abs(a)>.05)&(np.abs(b)>.05)&(np.sign(a)!=np.sign(b)))))
                max_change=max(max_change,float(np.max(np.abs(a-b))))
        conflicts.append({'train_seed':initial[key],'repeat_count':len(labels),'maximum_action_difference':max_change,
                          'maximum_opposing_active_joints':opposing,'labels':labels.tolist()})
    dump(root/'repeated_initial_expert_labels.json',conflicts)
    flat=[]
    for row in rows:
        item={k:v for k,v in row.items() if k!='dominant_contact'}
        item.update(contact_body=(row['dominant_contact'] or {}).get('moving_body'),contact_slot=(row['dominant_contact'] or {}).get('obstacle_slot'))
        flat.append(item)
    with (root/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    summary={'lost_success_scenes':lost,'final_failure_counts':dict(Counter(saved[10000][s]['reason'] for s in lost)),
        'all_reason_step_replays_match':True,'original_reference_preserved':True,'training_steps':0,
        'repeated_initial_label_groups':len(conflicts),'opposing_initial_label_groups':sum(c['maximum_opposing_active_joints']>0 for c in conflicts),
        'all_checkpoints_unchanged':all(sha(path)==expected for path,expected in hashes.items())}
    dump(root/'summary.json',summary);dump(root/'episodes.json',rows)
    shutil.copy2(__file__,root/'executed_diagnostic.py');print(summary)


if __name__=='__main__':main()
