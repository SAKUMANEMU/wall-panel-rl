"""Paired fixed-speed continuation; validation-only, no reuse of the used test bank."""
import _bootstrap
import argparse
from collections import Counter
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.four_ppo_continuation import state_digest
from run_targeted_bc import read,dump,sha


def profiles(comparison):
    paths=[ROOT/p for p in comparison['arms']]
    specs=[yaml.safe_load(p.read_text()) for p in paths]
    if len(specs)!=2:raise ValueError('Two arms required')
    a,b=specs;allowed={'curriculum_speeds','comparison_arm','model_root','report_root'}
    differences={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    if differences!=allowed:raise ValueError('Unexpected inter-arm changes: '+str(differences))
    if [s['curriculum_speeds'] for s in specs]!=[[.01],[.02]]:raise ValueError('Expected 0.01 vs 0.02 m/s')
    for s in specs:
        if s['static_probability']!=comparison['static_probability'] or s['static_probability']!=.1:
            raise ValueError('User-confirmed proportion is 90% dynamic / 10% static')
        if s['expert_probability']!=.5 or s['steps']!=2000 or s['evaluate_every']!=1000 or s['validation_target_speed']!=.02:
            raise ValueError('Paired 2k budgets and fixed 0.02m/s validation required')
        if s.get('run_independent_test',True):raise ValueError('Speed screening must not use an independent test')
        if s['initial_source_added_ppo_steps']!=comparison['expected_source_added_ppo_steps']:
            raise ValueError('Wrong source stage')
    return paths,specs


def source_facts(spec):
    torch.set_num_threads(1);model=PPO.load(ROOT/spec['source_model'],device='cpu')
    source=read(ROOT/spec['source_report']/'summary.json')
    if source['selected']['ppo_steps']!=2000:raise ValueError('Expected selected 2k source')
    manifest=read(ROOT/spec['source_report']/'candidate_manifest.json')
    if sha(ROOT/spec['source_model'])!=manifest['model_sha256']:raise ValueError('Source hash differs from manifest')
    aux=torch.load(ROOT/spec['resume_auxiliary_state'],map_location='cpu',weights_only=False)
    if aux['ppo_num_timesteps']!=model.num_timesteps:raise ValueError('Mismatched PPO and BC checkpoint counters')
    facts={'source_model_sha256':sha(ROOT/spec['source_model']),'source_aux_sha256':sha(ROOT/spec['resume_auxiliary_state']),
        'ppo_counter':model.num_timesteps,'ppo_updates':model._n_updates,
        'ppo_optimizer_sha256':state_digest(model.policy.optimizer.state_dict()),
        'bc_optimizer_sha256':state_digest(aux['optimizer']),'bc_rng_sha256':state_digest(aux['rng']),
        'bc_pool_size':len(aux['pool']),'bc_inherited_updates':aux['accepted_updates'],
        'expected_target':source['selected']['target'],'expected_static_frozen':source['selected']['static_frozen'],
        'expected_legacy_static':source['selected']['legacy_static']}
    del model,aux;gc.collect();return facts


def summarize(comparison):
    _,specs=profiles(comparison);root=ROOT/comparison['report_root'];p=read(root/'protocol.json');facts=p['source']
    for name,expected in p['configuration_hashes'].items():assert sha(ROOT/name)==expected
    if 'runner_sha256' in p:assert sha(root/'code_archive/scripts/run_dynamic_speed_comparison.py')==p['runner_sha256']
    rows=[];summaries={};histories={};restorations=[]
    for s in specs:
        arm=s['comparison_arm'];report=ROOT/s['report_root'];assert read(report/'status.json')['status']=='completed'
        protocol=read(report/'protocol.json');summary=read(report/'summary.json');history=read(report/'history.json')
        assert not summary['independent_test']['performed'] and summary['ppo_transitions']==2000
        assert protocol['source_sha256']==facts['source_model_sha256'] and protocol['continuation']['optimizer_sha256']==facts['ppo_optimizer_sha256']
        assert protocol['continuation']['source_counter']==facts['ppo_counter']
        for name,expected in protocol['code_hashes'].items():assert sha(report/'code_archive'/name)==expected
        for name,expected in protocol['input_hashes'].items():assert sha(ROOT/name)==expected
        restore=summary['bc_restoration'];restorations.append(restore)
        assert restore['restored'] and restore['source_state_sha256']==facts['source_aux_sha256']
        assert restore['optimizer_sha256']==facts['bc_optimizer_sha256'] and restore['rng_sha256']==facts['bc_rng_sha256']
        assert restore['pool_size']==facts['bc_pool_size'] and restore['accepted_updates']==facts['bc_inherited_updates']
        assert history[0]['target']==facts['expected_target'] and history[0]['static_frozen']==facts['expected_static_frozen']
        assert history[0]['legacy_static']==facts['expected_legacy_static']
        assert [h['ppo_steps'] for h in history]==[0,1000,2000]
        records=[json.loads(x) for f in (ROOT/s['model_root']).glob('episodes_worker_*.jsonl') for x in f.read_text().splitlines()]
        assert all(r['seed'] in protocol['train_seeds'] and r['seed'] not in protocol['excluded_seeds'] for r in records)
        accepted=[r for r in records if r.get('accepted_for_imitation')]
        assert all(r['pre_verified'] and r['post_time_verified'] and r['physical_replay_verified'] for r in accepted)
        state=torch.load(ROOT/s['model_root']/'auxiliary_bc/final_aux_state.pt',map_location='cpu',weights_only=False)
        assert len(state['pool'])==min(s['auxiliary_bc']['capacity'],facts['bc_pool_size']+sum(r['steps'] for r in accepted))
        assert state['accepted_updates']==facts['bc_inherited_updates']+summary['bc_auxiliary_updates']
        summaries[arm]=summary;histories[arm]=history
        for h in history:
            row={'arm':arm,'train_speed_m_s':s['curriculum_speeds'][0],'additional_ppo_steps':h['ppo_steps'],
                'total_added_since_v5':2000+h['ppo_steps'],'selected':h['ppo_steps']==summary['selected']['ppo_steps']}
            for mode in ('target','static_frozen','legacy_static','mixture'):
                row.update({mode+'_'+k:h[mode][k] for k in ('success','collision','timeout')})
            rows.append(row)
        import csv
        arm_rows=[r for r in rows if r['arm']==arm]
        with (report/'checkpoints.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(arm_rows[0]));writer.writeheader();writer.writerows(arm_rows)
        status=read(report/'status.json');status.update(process_exited=True,selected_target=summary['selected']['target'],final_candidate_target=history[-1]['target'])
        dump(report/'status.json',status)
    assert restorations[0]==restorations[1]
    import csv
    with (root/'checkpoints.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source_score=histories[specs[0]['comparison_arm']][0]['score']
    best=max(tuple(s['selected']['score']) for s in summaries.values())
    winners=[name for name,s in summaries.items() if tuple(s['selected']['score'])==best] if best>tuple(source_score) else []
    summary={'arms':summaries,'winner_arms_by_validation':winners,'retained_source_if_no_improvement':not winners,
        'new_ppo_transitions':4000,'static_probability':.1,'expert_probability':.5,
        'validation_target_speed':.02,'independent_test_performed':False,'source':facts,
        'sole_inter_arm_algorithm_change':'fixed training speed 0.01 vs 0.02m/s',
        'shared_change_from_prior_run':'static episode probability 0.30 -> 0.10; no 70/30 control in this experiment',
        'fresh_physics_episodes_after_resume':True,'default_model_preserved':True}
    dump(root/'summary.json',summary)
    verification={'matched_ppo_and_bc_source_states':True,'bc_pool_and_rng_restored':True,'initial_validation_matches_source':True,
        'train_heldout_separated':True,'archived_code_and_data_hashes_match':True,'used_test_bank_not_reevaluated':True,
        'new_bc_updates_reported_separately_from_inherited':True,'source_and_default_preserved':True}
    dump(root/'verification.json',verification)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key in zip(axes,('success','collision','timeout')):
        for s in specs:
            h=histories[s['comparison_arm']];ax.plot([x['ppo_steps'] for x in h],[100*x['target'][key]/64 for x in h],marker='o',label=f"train {s['curriculum_speeds'][0]:.2f} m/s")
        ax.set(title='Target validation: '+key,xlabel='Additional PPO transitions',ylabel='Percent');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Same +2k source; 90% dynamic / 10% static; both validate at 0.02 m/s')
    fig.tight_layout();fig.savefig(root/'curves.png',dpi=160);plt.close(fig)
    def vals(m):return f"{m['success']}/{m['collision']}/{m['timeout']}"
    text='''# Paired fixed-speed continuation (90% dynamic, 10% static)\n\nBoth arms start from exactly the selected v8+2k policy/PPO Adam/critic and matched BC Adam/pool/RNG. Each uses fresh physical episodes with the same training seed71 and validation seed61,50/50 RRT/PPO ownership, and2,000 new PPO transitions. Only the fixed training speed differs. All target validation uses0.02m/s. This is a single-seed validation screening, not an independent-test result.\n\nCounts are success/collision/timeout out of64.\n\n| Arm | Additional PPO steps | Dynamic target | Paired frozen | Original static | Mixture |\n|---|---:|---|---|---|---|\n'''
    for s in specs:
        arm=s['comparison_arm']
        for h in histories[arm]:text+=f"| {arm} | {h['ppo_steps']} | {vals(h['target'])} | {vals(h['static_frozen'])} | {vals(h['legacy_static'])} | {vals(h['mixture'])} |\n"
    text+='\n## Selection\n\n'
    for s in specs:
        arm=s['comparison_arm'];selected=summaries[arm]['selected']
        text+=f"- {arm}: select +{selected['ppo_steps']} new PPO steps (v5+{2000+selected['ppo_steps']} total for this adaptation chain); target {vals(selected['target'])}.\n"
    text+=f"\nValidation winners beyond the common starting model: {winners or 'none; retain source'}. The final2k continuation and selected checkpoint are stored separately for each arm. No default file is overwritten.\n"
    text+='''\n## Limits and accounting\n\nThe90/10 probability applies to new training scene episodes; finite observed proportions can differ, and the inherited BC pool is preserved intact. Both arms change that proportion together, so this experiment cannot isolate the effect of90/10 versus70/30. It isolates training speed within the new shared proportion. RRT trajectories and episode counts can diverge as a consequence of the speed change; input rules/seeds are paired, not the resulting physics. Fresh environments replace unfinished old episodes; this is learned-state continuation, not bitwise physics/RNG continuation of the previous run.\n\nThe previously used100-case test remains excluded and is not reevaluated. A later final model needs a fresh reserved test after configuration selection and multi-seed checks. `verification.json` audits source/optimizer/pool/RNG matching and data isolation. `summary.json` contains each arm's PPO, RRT and verification-replay transition totals; audit replay steps are not training samples.\n\nSee `curves.png`, `checkpoints.csv` and each arm report's validation JSON/checkpoints. Existing v5/v8 source models are preserved.\n'''
    (root/'README.md').write_text(text)
    for s in specs:
        report=ROOT/s['report_root'];arm=s['comparison_arm'];selected=summaries[arm]['selected']
        (report/'README.md').write_text(f"# {arm} speed continuation\n\nFixed speed {s['curriculum_speeds'][0]}m/s,90% dynamic/10% static,50/50 RRT/PPO. Complete2k new PPO steps; select+{selected['ppo_steps']}.\n\nTarget counts: {vals(selected['target'])}. No independent test used.\n\n[Full paired comparison](../{root.name}/README.md)\n")
    shutil.copy2(__file__,root/'executed_summary.py')
    return summary


def run(comparison,config_file):
    paths,specs=profiles(comparison);root=ROOT/comparison['report_root']
    if root.exists():raise FileExistsError('Refusing comparison overwrite')
    for s in specs:
        if (ROOT/s['model_root']).exists() or (ROOT/s['report_root']).exists():raise FileExistsError('Arm output exists')
    facts=source_facts(specs[0]);root.mkdir(parents=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    if visible.resolve()!=root.resolve():
        if visible.exists() or visible.is_symlink():raise FileExistsError('Visible alias exists')
        visible.symlink_to(root,target_is_directory=True)
    dump(root/'protocol.json',{'comparison':comparison,'arm_specs':specs,'source':facts,'runner_sha256':sha(__file__),
        'configuration_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(config_file),*paths]},
        'source_state_rule':'Matched 62000 PPO counter and BC aux_2000 snapshot; fresh physical episodes for both arms',
        'test_rule':'Do not reevaluate the used100-case test during speed screening'})
    for path in [Path(config_file),*paths,ROOT/'scripts/run_dynamic_speed_comparison.py']:
        target=root/'code_archive'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    completed=[]
    def status(phase,**extra):
        state={'status':'completed' if phase=='complete' else 'failed' if phase=='failed' else 'running',
            'phase':phase,'updated_at':datetime.now().astimezone().isoformat(),'completed_arms':completed,
            'new_ppo_budget_total':4000,'dynamic_probability':.9,'static_probability':.1,**extra}
        dump(root/'status.json',state)
    try:
        for path,s in zip(paths,specs):
            protocol=read(root/'protocol.json')
            for name,expected in protocol['configuration_hashes'].items():assert sha(ROOT/name)==expected
            arm=s['comparison_arm'];log=Path('/home/johnsonchen/wall-panel-rl/logs')/(Path(s['report_root']).name+'.log')
            command=[sys.executable,'-u',str(ROOT/'scripts/train_four_dynamic_ppo.py'),'--config',str(path),'--train']
            print('SPEED_COMPARISON_START',arm,str(log),flush=True)
            with log.open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                status('running_arm',active_arm=arm,child_pid=child.pid,log=str(log),active_ppo_steps=0)
                while child.poll() is None:
                    current=ROOT/s['report_root']/'status.json'
                    if current.exists():
                        try:
                            snapshot=read(current)
                            status('running_arm',active_arm=arm,child_pid=child.pid,log=str(log),active_phase=snapshot['phase'],
                                   active_ppo_steps=snapshot.get('ppo_steps',0),active_status_path=str(current.resolve()))
                        except json.JSONDecodeError:pass
                    time.sleep(5)
            if child.returncode:raise RuntimeError(f'{arm} failed with code {child.returncode}; see {log}')
            completed.append(arm);print('SPEED_COMPARISON_FINISHED',arm,flush=True)
        summary=summarize(comparison);status('complete',summary=summary)
        print('SPEED_COMPARISON_COMPLETE',str(visible),flush=True)
    except Exception as exc:status('failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_dynamic_speed_comparison.yaml')
    group=p.add_mutually_exclusive_group();group.add_argument('--run',action='store_true');group.add_argument('--summarize',action='store_true')
    args=p.parse_args();comparison=yaml.safe_load(args.config.read_text());_,specs=profiles(comparison)
    print({'mode':'run' if args.run else 'summarize' if args.summarize else 'check_only','arms':[(s['comparison_arm'],s['curriculum_speeds'][0]) for s in specs],
        'ppo_steps_per_arm':2000,'dynamic_probability':.9,'static_probability':.1,'expert_probability':.5,'independent_test':False},flush=True)
    if args.run:run(comparison,args.config.resolve())
    if args.summarize:print(json.dumps(summarize(comparison),indent=2))


if __name__=='__main__':main()
