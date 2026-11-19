"""Verify and compare matched 5k mixed-training runs; never train or promote."""
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


def main():
    torch.set_num_threads(1)
    old=Path('results/diagnostics/four_episode_mix_v6_seed61')
    new=Path('results/diagnostics/four_episode_mix_nobc_v7_seed61')
    out=Path('results/diagnostics/four_mix_ablation_v7_seed61')
    assert read(new/'status.json')['status']=='completed', 'Training has not finished'
    if out.exists():raise FileExistsError('Refusing comparison overwrite')
    a,b=read(old/'protocol.json'),read(new/'protocol.json')
    allowed={'steps','model_root','report_root','evaluation_root','configuration_file','auxiliary_bc'}
    assert all(b['spec'][k]==v for k,v in a['spec'].items() if k not in allowed)
    aa,bb=a['spec']['auxiliary_bc'],b['spec']['auxiliary_bc']
    assert aa['updates_per_rollout']==1 and bb['updates_per_rollout']==0
    assert all(bb[k]==v for k,v in aa.items() if k!='updates_per_rollout')
    for key in ('configuration','evaluation_settings','source_sha256','continuation',
                'train_seeds','excluded_seeds','validation_case_sha256'):
        assert a[key]==b[key], f'Changed comparison input: {key}'
    assert sha(b['spec']['source_model'])==b['source_sha256']
    assert sha(Path(b['spec']['data_root'])/'validation_cases.json')==b['validation_case_sha256']
    for name,expected in b['code_hashes'].items():assert sha(new/'code_archive'/name)==expected
    for name,expected in a['code_hashes'].items():
        assert sha(old/'code_archive'/name)==expected
        if name.startswith('src/'):assert b['code_hashes'][name]==expected
    summ=read(new/'summary.json');assert summ['ppo_transitions']==5000
    assert summ['bc_auxiliary_updates']==summ['bc_rejected_updates']==0
    models=Path(b['spec']['model_root'])
    aux=torch.load(models/'auxiliary_bc/final_aux_state.pt',map_location='cpu',weights_only=False)
    old_aux=torch.load(Path(a['spec']['model_root'])/'auxiliary_bc/aux_5000.pt',map_location='cpu',weights_only=False)
    assert not aux['optimizer']['state'] and len(aux['pool'])>0
    assert aux['accepted_updates']==aux['rejected_updates']==0
    same_expert_pool=(len(old_aux['pool'])==len(aux['pool']) and all(
        np.array_equal(x[0],y[0]) and np.array_equal(x[1],y[1]) for x,y in zip(old_aux['pool'],aux['pool'])))
    assert same_expert_pool, 'Expert observations or labels differ in this matched experiment'
    records=[json.loads(line) for p in models.glob('episodes_worker_*.jsonl') for line in p.read_text().splitlines()]
    assert all(r['seed'] in b['train_seeds'] and r['seed'] not in b['excluded_seeds'] for r in records)
    assert all(r['pre_verified'] and r['post_verified'] and r['reference_preserved'] for r in records if r.get('accepted_for_imitation'))
    assert sum(r['steps'] for r in records if r['controller']=='rrt')==summ['rrt_transitions']
    source=PPO.load(b['spec']['source_model'],device='cpu')
    zero=PPO.load(models/'checkpoints/ppo_0.zip',device='cpu')
    selected=PPO.load(new/'selected_model.zip',device='cpu')
    assert all(torch.equal(v,zero.policy.state_dict()[k]) for k,v in source.policy.state_dict().items())
    selected_source_equal=all(torch.equal(v,selected.policy.state_dict()[k]) for k,v in source.policy.state_dict().items())
    rows=[];paired={};paired_rows=[];initial_match=True
    for label,root in (('BC_on',old),('BC_off',new)):
        hist=read(root/'history.json')
        for h in hist:
            if h['ppo_steps']>5000:continue
            for mode in ('ppo','mixture'):
                rows.append({'experiment':label,'ppo_steps':h['ppo_steps'],'mode':mode,**h[mode]})
    for mode in ('ppo','mixture'):
        data={}
        for label,root in (('BC_on',old),('BC_off',new)):
            for n in (0,5000):
                data[label,n]=read(root/'validation'/f'{mode}_{n}.json')['rows']
        keys=[(r['seed'],r['controller']) for r in data['BC_on',0]]
        assert all([(r['seed'],r['controller']) for r in values]==keys for values in data.values())
        assert [r['reason'] for r in data['BC_on',0]]==[r['reason'] for r in data['BC_off',0]]
        initial_match &= all(x['steps']==y['steps'] and np.isclose(x['position_m'],y['position_m'])
                             for x,y in zip(data['BC_on',0],data['BC_off',0]))
        paired[mode]=dict(Counter(x['reason']+' -> '+y['reason'] for x,y in zip(data['BC_on',5000],data['BC_off',5000])))
        for x,y in zip(data['BC_on',5000],data['BC_off',5000]):
            paired_rows.append({'mode':mode,'seed':x['seed'],'controller':x['controller'],
                'bc_on_reason':x['reason'],'bc_off_reason':y['reason'],'bc_on_steps':x['steps'],'bc_off_steps':y['steps'],
                'bc_on_position_m':x['position_m'],'bc_off_position_m':y['position_m']})
    assert initial_match
    out.mkdir(parents=True)
    with (out/'checkpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'paired_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(paired_rows[0]));w.writeheader();w.writerows(paired_rows)
    dump(out/'paired_5k.json',paired)
    verification={'sole_algorithm_change':'auxiliary_bc.updates_per_rollout: 1 -> 0',
        'matched_budget':5000,'source_weights_at_zero_match':True,'baseline_episodes_match':True,
        'same_ppo_optimizer_and_value_initialization':True,'same_validation_assignments':True,
        'same_archived_runtime_source_hashes':True,'bc_accepted_updates':0,'bc_rejected_updates':0,
        'same_expert_observations_and_action_labels_at_5k':same_expert_pool,'bc_on_updates_by_5k':old_aux['accepted_updates'],
        'saved_expert_pairs':len(aux['pool']),'bc_optimizer_empty':True,
        'selected_ppo_steps':summ['selected']['ppo_steps'],'selected_equals_source':selected_source_equal,
        'source_unchanged':True,'independent_test_used':False,'default_model_preserved':True}
    dump(out/'verification.json',verification)
    dump(out/'summary.json',{'checkpoints':rows,'paired_5k':paired,'training':summ,'verification':verification})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,mode in zip(axes,('ppo','mixture')):
        for label in ('BC_on','BC_off'):
            group=[r for r in rows if r['experiment']==label and r['mode']==mode]
            ax.plot([r['ppo_steps'] for r in group],[100*r['success']/r['episodes'] for r in group],marker='o',label=label)
        ax.set(title='Pure PPO validation' if mode=='ppo' else 'Fixed episode-mixture validation',
               xlabel='Additional PPO transitions',ylabel='Success (%)',xticks=[0,5000],ylim=(0,100))
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle('Same source, seed and 64 scenes; online BC updates 1 vs 0')
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=160);plt.close(fig)
    def counts(metrics):return f"{metrics['success']} / {metrics['collision']} / {metrics['timeout']}"
    text='''# Online-BC ablation at a matched 5,000 PPO transitions\n\nOnly the number of online auxiliary BC updates per rollout changes (1 to 0). Both experiments start from the original PPOv5 weights, optimizer and value network, with the same seed 61, training bank, hard-case sampling, PPO hyperparameters, reward scale, and 50% RRT / 50% PPO episode routing. The no-BC run still executes and verifies RRT episodes; it does not learn from their labels.\n\n| Experiment | Added PPO steps | Evaluation | Success / collision / timeout |\n|---|---:|---|---|\n'''
    for r in rows:text+=f"| {r['experiment']} | {r['ppo_steps']} | {r['mode']} | {counts(r)} |\n"
    on=next(r for r in rows if r['experiment']=='BC_on' and r['mode']=='ppo' and r['ppo_steps']==5000)
    off=next(r for r in rows if r['experiment']=='BC_off' and r['mode']=='ppo' and r['ppo_steps']==5000)
    text+=f"\nAt 5k, disabling online BC changes pure-PPO success by {off['success']-on['success']:+d}/64, collisions by {off['collision']-on['collision']:+d}/64, and timeouts by {off['timeout']-on['timeout']:+d}/64 versus BC-on. This is a single-seed validation ablation. It does not establish a general causal conclusion; the original 10k endpoint is not a budget-matched comparison.\n"
    text+=f"\nBoth runs have exactly the same {len(aux['pool']):,} verified expert observation/action pairs at 5k (array equality), with {old_aux['accepted_updates']} versus 0 accepted BC updates by that checkpoint. This run does not support online BC as the sole cause of regression: removing it did not improve the endpoint. Further work should examine PPO drift and the sensitivity of reference-cursor advancement while preserving the original source as a control.\n"
    text+=f"\nThe no-BC run selects +{summ['selected']['ppo_steps']} steps using full pure-PPO validation. Selected policy equals the source: **{selected_source_equal}**. This diagnostic run does not promote a new default model. The current independent-test result remains the earlier PPOv5 84/100; no fresh independent test is run here.\n"
    text+=f"\nTraining: {summ['ppo_transitions']:,} PPO and {summ['rrt_transitions']:,} RRT transitions, {summ['total_physics_transitions']:,} total. Episode starts: {summ['episode_starts']}. Completed outcomes: {summ['completed_by_controller']}. BC accepted/rejected updates: 0/0; saved verified expert pairs: {len(aux['pool'])}.\n"
    text+='''\n`verification.json` records protocol, source, archive and data-isolation checks. `checkpoints.csv`, `curves.png`, `paired_5k.json` and `paired_cases.csv` give the comparison. The detailed prior-regression diagnosis is in `../four_mix_regression_v7_seed61/README.md`; experiment-specific checkpoints and episode journals remain in their original model/report directories.\n'''
    (out/'README.md').write_text(text)
    # Final status counts must include episodes completed in the last rollout.
    status=read(new/'status.json');status['completed']={name:sum(summ['completed_by_controller'][name].values()) for name in ('rrt','ppo')}
    status['episode_starts']=summ['episode_starts'];dump(new/'status.json',status)
    (new/'README.md').write_text(text)
    shutil.copy2(out/'curves.png',new/'curves.png');shutil.copy2(out/'checkpoints.csv',new/'checkpoints.csv')
    shutil.copy2(__file__,out/'executed_summary.py')
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/out.name
    if visible.is_symlink():assert visible.resolve()==out.resolve()
    else:assert not visible.exists();visible.symlink_to(out.resolve(),target_is_directory=True)
    print(json.dumps({'comparison':str(visible),'on_5k':on,'off_5k':off,'verification':verification},indent=2))


if __name__=='__main__':main()
