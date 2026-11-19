"""Audit predictive-controller comparison and export paired evidence/figures."""
import _bootstrap
import csv,json,shutil
from pathlib import Path
import numpy as np,yaml
from wall_panel_rl.utils.config import ROOT
from run_targeted_bc import read,dump,sha
from wall_panel_rl.rl.dynamic_four_training import counts,compose_mixture
from evaluate_predictive_safety import preserved_successes


def main():
    spec=yaml.safe_load((ROOT/'configs/four_predictive_safety.yaml').read_text());root=ROOT/spec['report_root']
    status=read(root/'status.json');assert status['status']=='completed';p=read(root/'protocol.json');s=read(root/'summary.json')
    for f,h in p['code_hashes'].items():assert sha(root/'code_archive'/f)==h
    for f,h in p['input_hashes'].items():assert sha(ROOT/f)==h
    assert sha(ROOT/spec['source_model'])==p['source_sha256']
    split=read(ROOT/spec['data_root']/'split.json');assert set(p['tuning_train_seeds']).issubset(split['train'])
    source=read(root/'validation_source.json');safety=read(root/'validation_safety.json');preserved_successes(source['rows'],safety['rows'])
    assert counts(source['rows'])==s['validation_source'] and counts(safety['rows'])==s['validation_safety']
    pairs=[]
    for a,b in zip(source['rows'],safety['rows']):
        pairs.append({'seed':a['seed'],'source_reason':a['reason'],'predictive_reason':b['reason'],'source_steps':a['steps'],'predictive_steps':b['steps'],
            'interventions':b['interventions'],'no_safe_action':b['no_safe_action'],'discarded_prediction_steps':b['discarded_prediction_steps']})
    with (root/'paired_validation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    cache=read(ROOT/'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation_expert_cache.json')
    mixed=read(root/'validation_mixture_safety.json');assert compose_mixture(safety,cache)==mixed
    assert [r for r in mixed['rows'] if r['controller']=='rrt']==cache['rrt_rows']
    test=read(root/'independent_test.json')
    if test['performed']:
        a=read(root/'independent_source.json');b=read(root/'independent_safety.json');preserved_successes(a['rows'],b['rows'])
        assert {r['seed'] for r in a['rows']}==set(split['test']) and not set(split['test'])&set(split['train'])
    audit={'archived_code_and_frozen_inputs_verified':True,'model_source_unchanged':True,'optimizer_updates':0,
        'controller_is_pure_ppo':False,'parameter_selection_train_only':True,'original_validation_successes_unchanged':True,
        'independent_test_performed':test['performed'],'all_original_test_successes_unchanged':bool(test.get('original_successes_preserved',False)),
        'common_rrt_rows_identical':True,'validation_paired_outcomes':dict(__import__('collections').Counter(a['reason']+' -> '+b['reason'] for a,b in zip(source['rows'],safety['rows']))),
        'selected_horizon':s['selected_horizon'],'discarded_prediction_steps':s['validation_prediction_steps']}
    dump(root/'verification.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    groups=[('Validation (64 scenes)',[('PPO',s['validation_source']),('PPO + prediction',s['validation_safety']),('50/50 original',s['validation_mixture_source']),('50/50 predictive',s['validation_mixture_safety'])])]
    if test['performed']:groups.append(('Independent test (100 scenes)',[('PPO',test['source']),('PPO + prediction',test['safety'])]))
    for row in range(2):
        for col,key in enumerate(['success','collision','timeout']):
            ax=axes[row,col]
            if row>=len(groups):ax.axis('off');ax.text(.5,.5,'Independent test not run',ha='center');continue
            title,items=groups[row];values=[100*m[key]/m['episodes'] for name,m in items]
            bars=ax.bar(range(len(items)),values,color=['#4676a9','#369e6c','#d89745','#8d69b6'][:len(items)])
            ax.set_xticks(range(len(items)),[name for name,m in items],rotation=18,ha='right',fontsize=8);ax.set(title=title+'\n'+key.capitalize()+' rate',ylabel='%');ax.set_ylim(0,max(12,max(values)*1.2));ax.grid(axis='y',alpha=.2)
            for bar,(_,m) in zip(bars,items):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f"{m[key]}/{m['episodes']}",ha='center',fontsize=8)
    fig.suptitle('Execution-controller comparison: PPO weights unchanged; all scenes dynamic');fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    def values(m):return f"{m['success']}/{m['collision']}/{m['timeout']}/{m['planning_failure']}"
    text='# PPO with predictive execution safety\n\nThis is a model-based execution-controller improvement, not newly learned pure PPO. Original PPO weights and optimizers are unchanged. The extra layer runs short physical rollouts at candidate speeds, then restores physics, simulation time, velocities, observations, reference cursor and recovery history before actual execution. It uses known obstacle motion; planning time is paused. It has no global RRT fallback. Actual collisions/timeouts retain the original termination rules.\n\n'
    text+=f"Fixed scene: four moving unequal staggered obstacles;18cm offset,0.12m/s speed. Selected horizon: {s['selected_horizon']} steps, chosen using only24 training scenes. No validation/test labels train a model.\n\n"
    text+='Counts below are success/collision/timeout/planning-failure.\n\n| Suite | Source PPO/system | Predictive PPO/system |\n|---|---|---|\n'
    text+=f"| Validation PPO branch | {values(s['validation_source'])} | {values(s['validation_safety'])} |\n| Validation50/50 mixture | {values(s['validation_mixture_source'])} | {values(s['validation_mixture_safety'])} |\n"
    if test['performed']:text+=f"| Independent PPO-branch test | {values(test['source'])} | {values(test['safety'])} |\n"
    text+=f"\nValidation transitions: {audit['validation_paired_outcomes']}. All original successes reproduce their outcome, episode length and final errors without an intervention. Test transitions: {test.get('paired',{})}. Parameters are locked before the independent test, with no subsequent tuning.\n\n"
    text+=f"Virtual validation prediction steps discarded: {s['validation_prediction_steps']}; interventions: {s['validation_interventions']}. These are computation, not training samples or real episode transitions. Finite-horizon checking does not guarantee safety beyond the horizon. Extra model information and computation distinguish this controller from pure PPO. Source/default PPO remains available.\n"
    (root/'README.md').write_text(text);shutil.copy2(__file__,root/'executed_summary.py')
    status.update(process_exited=True,verification=audit);dump(root/'status.json',status);print(json.dumps({'summary':s,'verification':audit},indent=2))

if __name__=='__main__':main()
