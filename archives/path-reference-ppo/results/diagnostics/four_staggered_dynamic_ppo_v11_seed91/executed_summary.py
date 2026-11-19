"""Audit the large-motion pilot and export checkpoint counts/curves."""
import _bootstrap
import argparse,csv,json,shutil
from collections import Counter
from pathlib import Path
import torch,yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/four_staggered_dynamic_training.yaml')
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
    final=PPO.load(root/'final_candidate.zip',device='cpu')
    assert final.num_timesteps==source.num_timesteps+spec['steps'] and final._n_updates>source._n_updates
    assert all(torch.equal(value,final.policy.state_dict()[key]) for key,value in source.policy.state_dict().items()
               if not key.startswith(('multi_adapter.',*CRITIC_PREFIXES)))
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
    manifest.update(dynamic_only_labels=True,auxiliary_state=str(saved.relative_to(ROOT)),auxiliary_state_sha256=sha(saved),ppo_global_steps=selected.num_timesteps)
    dump(root/'candidate_manifest.json',manifest)
    cache=read(root/'validation_expert_cache.json');owners=[(r['seed'],r['controller']) for r in cache['owners']]
    rows=[]
    target_key='dynamic_'+format(spec['validation_target_speed'],'.3f')
    for h in history:
        row={k:h[k] for k in ['ppo_steps','curriculum_amplitude','curriculum_speed','next_curriculum_level']}
        row['selected']=h['ppo_steps']==s['selected']['ppo_steps']
        for mode in ['target','mixture']:
            for outcome in ['success','collision','timeout','joint_limit','planning_failure']:row[mode+'_'+outcome]=h[mode][outcome]
        rows.append(row)
        mixed=read(root/'validation'/f"mixture_{h['ppo_steps']}.json")
        assert [(r['seed'],r['controller']) for r in mixed['rows']]==owners
        assert [r for r in mixed['rows'] if r['controller']=='rrt']==cache['rrt_rows']
        result=read(root/'validation'/f"{target_key}_{h['ppo_steps']}.json")
        assert all(r['speed_m_s']==spec['validation_target_speed'] and r['amplitude_m']==spec['validation_target_amplitude'] for r in result['rows'])
    with (root/'checkpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    warm=read(ROOT/spec['warmup_report']/'summary.json')
    original_validation=read(ROOT/spec['warmup_report']/'baseline_validation.json')
    before=original_validation['rows']
    assert all(r['task']=='dynamic' for r in records)
    assert s['completed_by_task'].get('static',0)==0 and warm['static_labels']==0
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
    assert test['performed']==(s['selected']['ppo_steps']>0 or warm['selected_bc_updates']>0)
    if test['performed']:
        ids=set(read(ROOT/spec['data_root']/'split.json')['test'])
        assert ids.isdisjoint(p['train_seeds'])
        for name in ['source','selected']:
            result=read(root/f'independent_{name}.json');assert result['independent_test']
            assert {r['seed'] for r in result['rows']}==ids
    complete=sum(r['steps'] for r in records if r['controller']=='ppo')
    audit={'source_unchanged':True,'archived_code_and_input_hashes_verified':True,'train_evaluation_disjoint':True,
        'actual_final_ppo_counter':final.num_timesteps,'ppo_epoch_counter_delta':final._n_updates-source._n_updates,
        'final_base_actor_and_std_preserved':True,'static_training_episodes':0,
        'observed_online_motion_pairs':sorted({(r['amplitude_m'],r['speed_m_s']) for r in records}),
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
    bc_history=read(ROOT/spec['warmup_report']/'history.json')
    bc_rows=[{'bc_updates':h['bc_updates'],**h['metrics'],'accepted_updates':h['accepted_updates'],
              'rejected_updates':h['rejected_updates']} for h in bc_history]
    with (ROOT/spec['warmup_report']/'checkpoints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(bc_rows[0]));writer.writeheader();writer.writerows(bc_rows)
    bc_fig,bc_axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,key in zip(bc_axes,['success','collision','timeout']):
        ax.plot([h['bc_updates'] for h in bc_history],[100*h['metrics'][key]/h['metrics']['episodes'] for h in bc_history],marker='o')
        ax.set(title=key.capitalize()+' rate',xlabel='Attempted BC updates',ylabel='%');ax.grid(alpha=.2)
    bc_fig.suptitle('Dynamic-only BC warmup: validation selects the retained source')
    bc_fig.tight_layout();bc_fig.savefig(ROOT/spec['warmup_report']/'curves.png',dpi=160);plt.close(bc_fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7));x=[h['ppo_steps'] for h in history]
    for ax,outcome in zip(axes.flat[:3],['success','collision','timeout']):
        for key,label in [('target','Dynamic PPO (18 cm, 0.12 m/s)'),('mixture','50/50 RRT/PPO')]:
            ax.plot(x,[100*h[key][outcome]/h[key]['episodes'] for h in history],marker='o',label=label)
        ax.axhline(100*original_validation['metrics'][outcome]/len(before),linestyle='--',color='gray',label='Original source PPO')
        ax.set(title=outcome.capitalize()+' rate',xlabel='Additional PPO steps',ylabel='%');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=8)
    ax=axes[1,1];ax.step(x,[spec['curriculum_speeds'][h['next_curriculum_level']] for h in history],where='post',label='Speed (m/s)')
    ax.step(x,[spec['curriculum_amplitudes'][h['next_curriculum_level']] for h in history],where='post',label='Amplitude (m)')
    ax.set(title='Requested curriculum for new episodes',xlabel='Additional PPO steps');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Staggered obstacles: 100% dynamic, episode-level 50/50 RRT/PPO');fig.tight_layout(rect=[0,.06,1,.95]);fig.savefig(root/'curves.png',dpi=160);plt.close(fig)
    def counts(m):return f"{m['success']}/{m['collision']}/{m['timeout']}/{m['planning_failure']}"
    text='# Staggered dynamic-only BC/PPO adaptation\n\nCounts are success/collision/timeout/planning-failure. All training and evaluation episodes have four moving unequal boxes. Motion bounds are18cm Euclidean anchor offset and0.12m/s Euclidean speed; along-route vertical offsets alternate by3-7cm. No static labels or static regression suites are used.\n\n'
    text+=f"Original source validation: {counts(original_validation['metrics'])}. BC warmup selected {warm['selected_bc_updates']} updates from {warm['bc_updates']}; verified offline teachers: {warm['teacher_episodes']} episodes, {warm['teacher_transitions']} transitions.\n\n"
    text+=f"Actual online motion pairs (amplitude m, speed m/s): {audit['observed_online_motion_pairs']}. The curriculum did not advance because its dynamic non-regression gates were not met. Offline teachers, fixed-target validation and the enhanced demo use0.18m/0.12m/s.\n\n"
    diagnosis=read(ROOT/spec['warmup_report']/'diagnosis_summary.json')
    contact_analysis=read(ROOT/spec['warmup_report']/'collision_analysis.json')
    text+=f"Training-only source diagnosis: {diagnosis['outcomes']}; collision stages: {diagnosis['collision_stages']}. Contact categories by episode: {contact_analysis['episodes_by_contact_category']}. Terminal-query gaps were resolved by22 physical-substep replays, reproducing source outcomes and step counts. No optimizer updates were made during diagnosis.\n\n"
    text+='| Additional PPO steps | Preceding training amplitude/speed | Pure PPO | 50/50 mixture |\n|---:|---|---|---|\n'
    for h in history:text+=f"| {h['ppo_steps']} | {h['curriculum_amplitude']:.3f}m / {h['curriculum_speed']:.3f}m/s | {counts(h['target'])} | {counts(h['mixture'])} |\n"
    text+=f"\nSelected: BC{warm['selected_bc_updates']} plus PPO{n}. PPO step0 is the validation-selected BC checkpoint. Original source models are preserved. This is one training seed.\n\n"
    text+=f"Original-source versus final-selected validation pairs: {audit['validation_paired_outcomes']}. Rescued seeds: {audit['rescued_validation_seeds']}; lost successes: {audit['lost_validation_success_seeds']}.\n\n"
    text+='Independent paired test: '+(f"original source {counts(test['source'])}; final selected {counts(test['selected'])}; changes {test['paired']}. Selection was locked beforehand; no tuning follows this test.\n" if test['performed'] else 'not evaluated because no updated checkpoint passed selection; reserved test remains unused.\n')
    if test['performed'] and (test['selected']['success']<=test['source']['success'] or test['selected']['collision']>test['source']['collision']):
        text+='\nThe independent test does not establish a safer success-rate improvement. Keep the original source as the default; retain this checkpoint as an experimental result.\n'
    text+=f"\nPPO transitions: {spec['steps']}; online RRT execution: {s['rrt_transitions']}; online verification replay: {s['verification_replay_transitions']}. Training outcomes: {s['completed_by_controller']}; modes: {s['completed_by_task']}. Online auxiliary BC accepted updates: {s['bc_auxiliary_updates']}. Offline teacher/diagnostic physics are additional and logged separately.\n"
    text+='\nTraining failures determine hard-case sampling and teacher phase variants. Evaluation labels never enter either optimizer. All offline/online teacher labels require time checks and independent physical replay. RRT actions never enter the PPO rollout buffer. Scene construction rejects only geometry or unavailable complete nominal references; it never filters by a policy outcome with obstacles. The stronger, staggered scene distribution differs from earlier experiments; compare only the paired source in this report.\n\nSee `protocol.json`, `verification.json`, `checkpoints.csv`, `paired_validation.csv`, `curves.png`, and the separate warmup report.\n'
    (root/'README.md').write_text(text)
    shutil.copy2(__file__,root/'executed_summary.py')
    status.update(process_exited=True,verification=audit,selected_target=s['selected']['target']);dump(root/'status.json',status)
    print(json.dumps({'report':str(root),'selected':s['selected'],'test':test,'verification':audit},indent=2))

if __name__=='__main__':main()
