"""Audit the large-motion pilot and export checkpoint counts/curves."""
import _bootstrap
import argparse,csv,json,shutil
from collections import Counter
from pathlib import Path
import torch,yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from wall_panel_rl.utils.config import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/four_large_motion_training.yaml')
    args=parser.parse_args();spec=yaml.safe_load(args.config.read_text());root=ROOT/spec['report_root'];models=ROOT/spec['model_root']
    torch.set_num_threads(1)
    status=read(root/'status.json');assert status['status']=='completed'
    p=read(root/'protocol.json');s=read(root/'summary.json');history=read(root/'history.json')
    assert [h['ppo_steps'] for h in history]==list(range(0,spec['steps']+1,spec['evaluate_every']))
    for f,digest in p['code_hashes'].items():assert sha(root/'code_archive'/f)==digest
    for f,digest in p['input_hashes'].items():assert sha(ROOT/f)==digest
    assert sha(ROOT/spec['source_model'])==p['source_sha256']
    records=[json.loads(line) for f in models.glob('episodes_worker_*.jsonl') for line in f.read_text().splitlines()]
    assert all(r['seed'] in p['train_seeds'] and r['seed'] not in p['excluded_seeds'] for r in records)
    experts=[r for r in records if r.get('accepted_for_imitation')]
    assert all(r['pre_verified'] and r['post_time_verified'] and r['physical_replay_verified'] and r['reference_preserved'] and r['reason']=='success' and r['replay_steps']==r['steps'] and r['replay_max_error']<=1e-6 for r in experts)
    assert sum(r['steps'] for r in records if r['controller']=='rrt')==s['rrt_transitions']
    assert sum(r.get('replay_steps',0) for r in records)==s['verification_replay_transitions']
    aux=torch.load(models/'auxiliary_bc/final_aux_state.pt',map_location='cpu',weights_only=False)
    initial=s['bc_restoration']['pool_size']
    assert len(aux['pool'])==min(spec['auxiliary_bc']['capacity'],initial+sum(r['steps'] for r in experts))
    source=PPO.load(ROOT/spec['source_model'],device='cpu');selected=PPO.load(root/'selected_model.zip',device='cpu')
    same=all(torch.equal(v,selected.policy.state_dict()[k]) for k,v in source.policy.state_dict().items())
    assert same==s['selected_equals_source']
    if s['selected']['ppo_steps']==0:assert same
    n=s['selected']['ppo_steps']
    matched=models/'auxiliary_bc'/('final_aux_state.pt' if n==spec['steps'] else f'aux_{n}.pt')
    matched_state=torch.load(matched,map_location='cpu',weights_only=False)
    assert matched_state['ppo_num_timesteps']==selected.num_timesteps
    saved=models/'selected_auxiliary_state.pt';shutil.copy2(matched,saved)
    manifest=read(root/'candidate_manifest.json')
    assert sha(root/'selected_model.zip')==sha(models/'selected_model.zip')==manifest['model_sha256']
    manifest.update(auxiliary_state=str(saved.relative_to(ROOT)),auxiliary_state_sha256=sha(saved),ppo_global_steps=selected.num_timesteps)
    dump(root/'candidate_manifest.json',manifest)
    cache=read(root/'validation_expert_cache.json');owners=[(r['seed'],r['controller']) for r in cache['owners']]
    rows=[]
    target_key='dynamic_'+format(spec['validation_target_speed'],'.3f')
    for h in history:
        row={k:h[k] for k in ['ppo_steps','curriculum_amplitude','curriculum_speed','next_curriculum_level']}
        row['selected']=h['ppo_steps']==s['selected']['ppo_steps']
        for mode in ['target','static_frozen','legacy_static','mixture']:
            for outcome in ['success','collision','timeout','joint_limit','planning_failure']:row[mode+'_'+outcome]=h[mode][outcome]
        rows.append(row)
        mixed=read(root/'validation'/f"mixture_{h['ppo_steps']}.json")
        assert [(r['seed'],r['controller']) for r in mixed['rows']]==owners
        assert [r for r in mixed['rows'] if r['controller']=='rrt']==cache['rrt_rows']
        result=read(root/'validation'/f"{target_key}_{h['ppo_steps']}.json")
        assert all(r['speed_m_s']==spec['validation_target_speed'] and r['amplitude_m']==spec['validation_target_amplitude'] for r in result['rows'])
    with (root/'checkpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    before=read(root/'validation'/f'{target_key}_0.json')['rows']
    after=read(root/'validation'/f'{target_key}_{n}.json')['rows']
    paired=[]
    for a,b in zip(before,after):
        assert a['seed']==b['seed']
        paired.append({'seed':a['seed'],'source_reason':a['reason'],'selected_reason':b['reason'],
            'source_steps':a['steps'],'selected_steps':b['steps'],
            'source_position_m':a['position_m'],'selected_position_m':b['position_m'],
            'source_reference_index':a['reference_index'],'selected_reference_index':b['reference_index']})
    with (root/'paired_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    learning=list(csv.DictReader((models/'training/progress.csv').open()))
    assert all(float(row['train/learning_rate'])==spec['learning_rate'] for row in learning if row.get('train/learning_rate'))
    test=read(root/'independent_test.json')
    assert test['performed']==(s['selected']['ppo_steps']>0)
    if test['performed']:
        ids=set(read(ROOT/spec['data_root']/'split.json')['test'])
        assert ids.isdisjoint(p['train_seeds'])
        for name in ['source','selected']:
            result=read(root/f'independent_{name}.json');assert result['independent_test']
            assert {r['seed'] for r in result['rows']}==ids
    complete=sum(r['steps'] for r in records if r['controller']=='ppo')
    audit={'source_unchanged':True,'archived_code_and_input_hashes_verified':True,'train_evaluation_disjoint':True,
        'verified_teacher_episodes':len(experts),'all_accepted_teachers_physically_replayed':True,
        'initial_expert_pairs':initial,'final_expert_pairs':len(aux['pool']),
        'ppo_completed_episode_steps':complete,'ppo_unfinished_episode_steps':spec['steps']-complete,
        'paired_mixture_cache_verified':True,'fixed_target_motion_verified':True,
        'selected_equals_source':same,'independent_test_performed':test['performed'],
        'selected_auxiliary_state_counter_matches':True,'selected_auxiliary_state':str(saved),
        'validation_paired_outcomes':dict(Counter(a['reason']+' -> '+b['reason'] for a,b in zip(before,after))),
        'rescued_validation_seeds':[a['seed'] for a,b in zip(before,after) if a['reason']!='success' and b['reason']=='success'],
        'lost_validation_success_seeds':[a['seed'] for a,b in zip(before,after) if a['reason']=='success' and b['reason']!='success']}
    dump(root/'verification.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7));x=[h['ppo_steps'] for h in history]
    for ax,outcome in zip(axes.flat[:3],['success','collision','timeout']):
        for key,label in [('target','Dynamic PPO (15 cm, 0.10 m/s)'),('mixture','50/50 RRT/PPO'),('static_frozen','Paired frozen'),('legacy_static','Original static')]:
            ax.plot(x,[100*h[key][outcome]/h[key]['episodes'] for h in history],marker='o',label=label)
        ax.set(title=outcome.capitalize()+' rate',xlabel='Additional PPO steps',ylabel='%');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    ax=axes[1,1];ax.step(x,[spec['curriculum_speeds'][h['next_curriculum_level']] for h in history],where='post',label='Speed (m/s)')
    ax.step(x,[spec['curriculum_amplitudes'][h['next_curriculum_level']] for h in history],where='post',label='Amplitude (m)')
    ax.set(title='Requested curriculum for new episodes',xlabel='Additional PPO steps');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Large-motion adaptation: 90% dynamic, episode-level 50/50 RRT/PPO');fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(root/'curves.png',dpi=160);plt.close(fig)
    def counts(m):return f"{m['success']}/{m['collision']}/{m['timeout']}/{m['planning_failure']}"
    text='# Large-motion PPO adaptation\n\nAll counts are success/collision/timeout/planning-failure. Each validation suite has 64 episodes. The target is fixed at 15 cm Euclidean anchor offset and 0.10 m/s Euclidean speed bounds; realized per-axis values are smaller. Curriculum changes only on new episodes.\n\n'
    text+='| Added PPO steps | Training amplitude/speed | Target PPO | Frozen | Original static | RRT/PPO mixture |\n|---:|---|---|---|---|---|\n'
    for h in history:text+=f"| {h['ppo_steps']} | {h['curriculum_amplitude']:.3f}m / {h['curriculum_speed']:.3f}m/s | {counts(h['target'])} | {counts(h['static_frozen'])} | {counts(h['legacy_static'])} | {counts(h['mixture'])} |\n"
    text+=f"\nSelected: +{s['selected']['ppo_steps']} PPO steps. Selected weights equal the source: {same}. Source and previous demo defaults are preserved. This is one training seed.\n\n"
    text+=f"Validation paired outcomes: {audit['validation_paired_outcomes']}. Rescued scene seeds: {audit['rescued_validation_seeds']}; lost prior successes: {audit['lost_validation_success_seeds']}.\n\n"
    text+='Independent paired test: '+(f"source {counts(test['source'])}; selected {counts(test['selected'])}; outcomes {test['paired']}. Selection was locked before this test; no subsequent tuning.\n" if test['performed'] else 'not performed because no new checkpoint passed selection; reserved100 cases remain unused.\n')
    if test['performed'] and test['selected']['success']<=test['source']['success']:
        text+='\nThis experiment does not demonstrate an independent-test success-rate improvement. In this run one timeout becomes a success, while one previous success becomes a collision. Keep the source as the default; the validation-selected checkpoint remains an experimental result.\n'
    text+=f"\nTraining PPO transitions: {spec['steps']}; additional RRT execution: {s['rrt_transitions']}; verification replay: {s['verification_replay_transitions']}. Completed branch outcomes: {s['completed_by_controller']}. Scene modes: {s['completed_by_task']}. New accepted BC updates: {s['bc_auxiliary_updates']}.\n"
    text+='\nBanks are fixed before training, with geometric/nominal-reference construction rejection logs. On-path anchors may be resampled within the previous path intervals to satisfy the full motion envelope. No outcome with obstacles is used to choose cases. This changes the scene distribution relative to earlier demo preflights, so those small preflight rates are not the baseline here. Evaluation labels never enter BC or PPO. RRT teachers enter auxiliary BC only after time checks and independent physical replay; their actions never enter the PPO rollout buffer.\n\nSee `protocol.json`, `verification.json`, `checkpoints.csv`, `curves.png` and `validation/`.\n'
    (root/'README.md').write_text(text)
    shutil.copy2(__file__,root/'executed_summary.py')
    status.update(process_exited=True,verification=audit,selected_target=s['selected']['target']);dump(root/'status.json',status)
    print(json.dumps({'report':str(root),'selected':s['selected'],'test':test,'verification':audit},indent=2))

if __name__=='__main__':main()
