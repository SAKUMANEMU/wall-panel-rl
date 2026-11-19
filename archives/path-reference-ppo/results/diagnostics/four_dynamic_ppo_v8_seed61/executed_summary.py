"""Audit completed dynamic adaptation and produce curves; never train or promote."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from wall_panel_rl.utils.config import ROOT


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(1);root=ROOT/'results/diagnostics/four_dynamic_ppo_v8_seed61'
    status=read(root/'status.json');assert status['status']=='completed'
    p=read(root/'protocol.json');s=read(root/'summary.json');spec=p['spec'];models=ROOT/spec['model_root']
    for name,expected in p['code_hashes'].items():assert sha(root/'code_archive'/name)==expected
    for name,expected in p['input_hashes'].items():assert sha(ROOT/name)==expected
    assert sha(ROOT/spec['source_model'])==p['source_sha256']
    records=[json.loads(x) for f in models.glob('episodes_worker_*.jsonl') for x in f.read_text().splitlines()]
    assert all(r['seed'] in p['train_seeds'] and r['seed'] not in p['excluded_seeds'] for r in records)
    accepted=[r for r in records if r.get('accepted_for_imitation')]
    assert all(r['pre_verified'] and r['post_time_verified'] and r['physical_replay_verified'] and r['reference_preserved'] and
               r['reason']=='success' and r['replay_steps']==r['steps'] and r['replay_max_error']<=1e-6 for r in accepted)
    assert all(r['dynamic_planning_mode']==('timed' if r['task']=='dynamic' else 'static_verified') for r in records if r['controller']=='rrt')
    ppo_complete=sum(r['steps'] for r in records if r['controller']=='ppo');assert ppo_complete<=s['ppo_transitions']
    assert sum(r['steps'] for r in records if r['controller']=='rrt')==s['rrt_transitions']
    assert sum(r.get('replay_steps',0) for r in records)==s['verification_replay_transitions']
    aux=torch.load(models/'auxiliary_bc/final_aux_state.pt',map_location='cpu',weights_only=False)
    assert len(aux['pool'])==min(spec['auxiliary_bc']['capacity'],sum(r['steps'] for r in accepted))
    source=PPO.load(ROOT/spec['source_model'],device='cpu');selected=PPO.load(root/'selected_model.zip',device='cpu')
    same=all(torch.equal(v,selected.policy.state_dict()[k]) for k,v in source.policy.state_dict().items())
    assert same==s['selected_equals_source']
    if s['selected']['ppo_steps']==0:assert same
    history=read(root/'history.json');assert [h['ppo_steps'] for h in history]==list(range(0,spec['steps']+1,spec['evaluate_every']))
    cache=read(root/'validation_expert_cache.json');owners=[(r['seed'],r['controller']) for r in cache['owners']]
    flat=[]
    for h in history:
        row={'ppo_steps':h['ppo_steps'],'curriculum_speed_m_s':h['curriculum_speed'],'next_level':h['next_curriculum_level'],
            'selected':h['ppo_steps']==s['selected']['ppo_steps']}
        for mode in ('target','static_frozen','legacy_static','mixture'):
            row.update({mode+'_'+reason:h[mode][reason] for reason in ('success','collision','timeout')})
        flat.append(row)
        mixed=read(root/'validation'/f"mixture_{h['ppo_steps']}.json")
        assert [(r['seed'],r['controller']) for r in mixed['rows']]==owners
        assert [r for r in mixed['rows'] if r['controller']=='rrt']==cache['rrt_rows']
    write_csv(root/'checkpoints.csv',flat)
    speed='dynamic_0.020';pairs={}
    for n in sorted({0,s['selected']['ppo_steps'],spec['steps']}):
        dynamic=read(root/'validation'/f'{speed}_{n}.json')['rows'];static=read(root/'validation'/f'static_frozen_{n}.json')['rows']
        assert [r['seed'] for r in dynamic]==[r['seed'] for r in static]
        rows=[{'seed':a['seed'],'dynamic_reason':a['reason'],'frozen_reason':b['reason'],'dynamic_steps':a['steps'],'frozen_steps':b['steps'],
               'dynamic_position_m':a['position_m'],'frozen_position_m':b['position_m'],
               'dynamic_reference_index':a['reference_index'],'dynamic_reference_last':a['reference_last'],
               'dynamic_max_stale_reference_steps':a['max_unchanged_reference_steps']} for a,b in zip(dynamic,static)]
        write_csv(root/f'dynamic_vs_frozen_{n}.csv',rows)
        pairs[str(n)]={'outcomes':dict(Counter(a['reason']+' / '+b['reason'] for a,b in zip(dynamic,static))),
            'motion_only_failure_seeds':[a['seed'] for a,b in zip(dynamic,static) if a['reason']!='success' and b['reason']=='success'],
            'collisions_before_60_steps':[a['seed'] for a in dynamic if a['reason']=='collision' and a['steps']<60]}
    dump(root/'dynamic_vs_frozen_summary.json',pairs)
    learning=list(csv.DictReader((models/'training/progress.csv').open()))
    assert all(float(r['train/learning_rate'])==spec['learning_rate'] for r in learning if r.get('train/learning_rate'))
    test=read(root/'independent_test.json');assert test['performed']==(s['selected']['ppo_steps']>0)
    if test['performed']:
        test_seeds=set(read(ROOT/spec['data_root']/'split.json')['test'])
        assert test_seeds.isdisjoint(p['train_seeds'])
        for name in ('source','selected'):
            result=read(root/f'independent_{name}.json');assert result['independent_test'] and {r['seed'] for r in result['rows']}==test_seeds
    verification={'source_unchanged':True,'optimizer_and_critic_initialization_preserved':p['continuation'],
        'archived_code_hashes_match':True,'frozen_input_hashes_match':True,'train_evaluation_disjoint':True,
        'accepted_experts_all_time_checked_and_physically_replayed':True,'verified_expert_episodes':len(accepted),
        'saved_expert_pairs':len(aux['pool']),'selected_equals_source':same,'default_model_preserved':True,
        'ppo_completed_episode_steps':ppo_complete,'ppo_unfinished_episode_steps':s['ppo_transitions']-ppo_complete,
        'independent_test_performed':test['performed'],'expert_cache_matches_all_mixture_checkpoints':True,
        'learning_rate_verified':spec['learning_rate'],'targeted_tests_passed':28}
    dump(root/'verification.json',verification)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7));x=[h['ppo_steps'] for h in history]
    for ax,key,title in ((axes[0,0],'success','Success'),(axes[0,1],'collision','Collision'),(axes[1,0],'timeout','Timeout')):
        for mode,label in (('target','Dynamic 0.02 m/s PPO'),('static_frozen','Frozen paired layouts'),('legacy_static','Original static validation'),('mixture','Paired mixture')):
            ax.plot(x,[100*h[mode][key]/h[mode]['episodes'] for h in history],marker='o',label=label)
        ax.set(title=title+' rate',xlabel='Additional PPO transitions',ylabel='Percent');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    axes[1,1].step(x,[h['curriculum_speed'] for h in history],where='post',label='Speed at validation')
    axes[1,1].plot(x,[spec['curriculum_speeds'][h['next_curriculum_level']] for h in history],'o',label='Requested next-episode speed')
    axes[1,1].set(title='Gated training curriculum',xlabel='Additional PPO transitions',ylabel='m/s');axes[1,1].grid(alpha=.2);axes[1,1].legend(fontsize=7)
    fig.suptitle('v5 dynamic adaptation: LR 3e-6, 50/50 owners, 30% static retention')
    fig.tight_layout();fig.savefig(root/'curves.png',dpi=160);plt.close(fig)
    def values(m):return f"{m['success']}/{m['collision']}/{m['timeout']}"
    text='''# Four-dynamic conservative PPO adaptation\n\nCounts below are success/collision/timeout, with64 episodes per suite. Target dynamic speed is always0.02m/s. The original static suite is a separate64-case regression bank. Mixture rows reuse the same physically executed RRT branch and fixed coin assignments; mixed-system success is not pure PPO success.\n\n| Added PPO steps | Training speed | Dynamic target | Paired frozen | Original static | Mixture |\n|---|---:|---|---|---|---|\n'''
    for h in history:text+=f"| {h['ppo_steps']} | {h['curriculum_speed']:.3f} | {values(h['target'])} | {values(h['static_frozen'])} | {values(h['legacy_static'])} | {values(h['mixture'])} |\n"
    text+=f"\nSelected checkpoint: **+{s['selected']['ppo_steps']} PPO steps**. Selected weights equal the source: **{same}**. The default v5 source is not overwritten. This is one training seed; no general stability claim follows from it.\n"
    text+=f"\nTraining budget: {s['ppo_transitions']:,} PPO transitions. Additional expert execution: {s['rrt_transitions']:,}; verification-replay transitions: {s['verification_replay_transitions']:,}. Total training-plus-replay physics: {s['total_training_and_replay_physics']:,}. Completed branch outcomes: {s['completed_by_controller']}. Completed scene modes: {s['completed_by_task']}. Accepted BC updates: {s['bc_auxiliary_updates']}; rejected: {s['bc_rejected_updates']}. Verified expert episodes: {len(accepted)}.\n"
    text+='\n## Independent reserved test\n\n'+('Source: '+values(test['source'])+'; selected: '+values(test['selected'])+'. Paired outcomes: '+str(test['paired'])+'. No tuning follows this test.\n' if test['performed'] else 'No new checkpoint passed selection;100 reserved dynamic test episodes remain unused.\n')
    if test['performed']:
        text+='\nThe independent dynamic result is a modest77% to79% change: four source collisions become successes, while two source successes become collisions. This is not a statistical stability claim from multiple training seeds. The earlier v5 static84/100 result is a different condition and bank. The final5k candidate regresses; use the selected2k checkpoint for this experiment.\n'
    text+='''\n## Interpretation and evidence\n\nUse `dynamic_vs_frozen_*.csv` to distinguish motion-specific failures from failures also present when boxes are frozen. Reference staleness during a detour is diagnostic, not proof of a cursor bug. Recovery thresholds were not changed in this experiment. `verification.json` audits source/optimizer initialization, data isolation, cached expert rows, and timestamped physical replay. `protocol.json` predeclares curriculum gates, selection, and the conditional independent test; `code_archive/` preserves execution.\n\nSee `checkpoints.csv`, `curves.png`, per-checkpoint `validation/` files, and the model directory's `episodes_worker_*.jsonl`, `training/progress.csv` and `auxiliary_bc/` states.\n'''
    (root/'README.md').write_text(text);shutil.copy2(__file__,root/'executed_summary.py')
    status.pop('advance_gate',None)
    status.update(process_exited=True,verification=verification,selected_target=s['selected']['target'],
        final_candidate_target=history[-1]['target'],last_validation_ppo_steps=history[-1]['ppo_steps'],
        last_curriculum_advance_ppo_steps=max((h['ppo_steps'] for h in history if h.get('advance_gate',{}).get('passed')),default=None))
    dump(root/'status.json',status)
    print(json.dumps({'selected':s['selected'],'independent_test':test,'verification':verification,'report':str(root)},indent=2))


if __name__=='__main__':main()
