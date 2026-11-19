"""Audit paired progress recovery results and expose timeout/compute tradeoffs."""
import _bootstrap
import csv,json,shutil
from pathlib import Path
from collections import Counter
import numpy as np,yaml
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import counts,compose_mixture
from evaluate_progress_recovery import preserve
from run_targeted_bc import read,dump,sha

def main():
    spec=yaml.safe_load((ROOT/'configs/four_progress_recovery.yaml').read_text());root=ROOT/spec['report_root'];status=read(root/'status.json');assert status['status']=='completed'
    p=read(root/'protocol.json');s=read(root/'summary.json')
    for f,h in p['code_hashes'].items():assert sha(root/'code_archive'/f)==h
    for f,h in p['input_hashes'].items():assert sha(ROOT/f)==h
    before=read(root/'validation_baseline.json');after=read(root/'validation_recovery.json');preserve(before['rows'],after['rows'])
    def paired(a,b,name):
        rows=[{'seed':x['seed'],'baseline_reason':x['reason'],'recovery_reason':y['reason'],'baseline_steps':x['steps'],'recovery_steps':y['steps'],
            'baseline_zero_waits':x['gain_counts'].get('0.0',0),'recovery_zero_waits':y['gain_counts'].get('0.0',0),
            'recovery_actions':y['recovery_actions'],'rescues':y['rescues'],'baseline_wall_s':x['total_wall_s'],'recovery_wall_s':y['total_wall_s'],
            'forecast_cache_hits':y['forecast_cache_hits']} for x,y in zip(a['rows'],b['rows'])]
        with (root/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return dict(Counter(x['baseline_reason']+' -> '+x['recovery_reason'] for x in rows))
    val_pairs=paired(before,after,'paired_validation')
    cache=read(ROOT/'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation_expert_cache.json');assert compose_mixture(after,cache)==read(root/'validation_mixture_recovery.json')
    test=read(root/'independent_test.json');old=new=None
    if test['performed']:
        old=read(root/'independent_baseline.json');new=read(root/'independent_recovery.json');preserve(old['rows'],new['rows'])
        split=read(ROOT/spec['fresh_test_root']/'split.json');assert {r['seed'] for r in old['rows']}==set(split['test'])
        assert not set(split['test'])&set(split['excluded_prior_ids'])
        assert paired(old,new,'paired_independent_test')==test['paired']
    def compute(a,b):
        return {'baseline_mean_wall_s':float(np.mean([r['total_wall_s'] for r in a['rows']])),
            'recovery_mean_wall_s':float(np.mean([r['total_wall_s'] for r in b['rows']])),
            'baseline_zero_waits':sum(r['gain_counts'].get('0.0',0) for r in a['rows']),
            'recovery_zero_waits':sum(r['gain_counts'].get('0.0',0) for r in b['rows']),
            'local_recovery_actions':sum(r['recovery_actions'] for r in b['rows']),
            'forecast_cache_hits':sum(r['forecast_cache_hits'] for r in b['rows'])}
    audit={'archived_code_and_frozen_inputs_verified':True,'model_unchanged':True,'optimizer_updates':0,'static_episodes':0,
        'unchanged_real_step_limit':240,'old_validation_successes_unchanged':True,'validation_paired':val_pairs,
        'new_test_disjoint_from_all_previous_ids':bool(test['performed']),'old_test_successes_unchanged':bool(test.get('old_successes_preserved',False)),
        'no_tuning_after_test':bool(test.get('no_tuning_after_test',False)),'validation_compute':compute(before,after)}
    if old:audit['independent_compute']=compute(old,new)
    dump(root/'verification.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(11,6.5));groups=[('Validation (64)',s['validation_baseline'],s['validation_recovery'])]
    if test['performed']:groups.append(('New independent test (100)',test['baseline'],test['recovery']))
    for i in range(2):
        for j,key in enumerate(['success','collision','timeout']):
            ax=axes[i,j]
            if i>=len(groups):ax.axis('off');continue
            title,a,b=groups[i];ms=[a,b];vals=[100*m[key]/m['episodes'] for m in ms];bars=ax.bar([0,1],vals,color=['#567fa7','#2d9c78'])
            ax.set_xticks([0,1],['Previous safety','Progress recovery'],rotation=12,ha='right',fontsize=8);ax.set(title=title+'\n'+key.capitalize()+' rate',ylabel='%');ax.set_ylim(0,max(8,max(vals)*1.2));ax.grid(axis='y',alpha=.2)
            for bar,m in zip(bars,ms):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.4,f"{m[key]}/{m['episodes']}",ha='center',fontsize=9)
    fig.suptitle('Progress recovery: same neural model, dynamics, collision rules and240-step limit');fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    def show(m):return f"{m['success']}/{m['collision']}/{m['timeout']}/{m['planning_failure']}"
    text='# Reducing waits with progress-aware local recovery\n\nThis remains a model-based execution-controller improvement, not a new PPO model. The previous12-step predictive safety controller is kept. After step100, low progress or predicted budget exhaustion can trigger recovery. A model simulation first protects continuations which would succeed with the old controller. For other cases, collision-free goal-directed joint actions or brief single-joint biases are compared locally. Actual reference paths are never overwritten and no global RRT is used in this branch.\n\n'
    text+='Scene: four unequal moving obstacles,18cm offset,0.12m/s speed, alternating3-7cm anchor offsets. Robot/panel geometry, success tolerances and240 real steps remain unchanged. All trial states and virtual time are restored before real execution.\n\nCounts are success/collision/timeout/planning-failure.\n\n| Suite | Previous safety | Progress recovery |\n|---|---|---|\n'
    text+=f"| Validation | {show(s['validation_baseline'])} | {show(s['validation_recovery'])} |\n| Validation50/50 RRT mixture | {show(s['validation_mixture_baseline'])} | {show(s['validation_mixture_recovery'])} |\n"
    if test['performed']:text+=f"| New independent test | {show(test['baseline'])} | {show(test['recovery'])} |\n"
    text+=f"\nValidation paired changes: {val_pairs}. New-test changes: {test.get('paired',{})}. Old successes preserve outcome, length and final errors with no recovery action. The previous100-case v13 bank is used only for historical development diagnosis; this test bank starts at1144000 and has no overlapping scene ID. Controller parameters are frozen before the new test.\n\n"
    text+=f"Computational tradeoff: {audit.get('independent_compute',audit['validation_compute'])}. These wall times include concurrent CPU work; predictive computation pauses simulation time. Recovery reduces simulated waiting/timeout but adds model-prediction work. There is no real-time guarantee.\n\n"
    text+='See `paired_validation.csv`, `paired_independent_test.csv`, `verification.json`, `comparison.png`, `validation_gate.json` and `independent_test.json`. Earlier pure-PPO and predictive-only demos remain available. New explicit demo: `scripts/demo_progress_dynamic.py`, default50/50 RRT and predictive-PPO-progress branches; use `--controller ppo_progress` for this branch alone.\n'
    (root/'README.md').write_text(text);shutil.copy2(__file__,root/'executed_summary.py');status.update(process_exited=True,verification=audit);dump(root/'status.json',status);print(json.dumps({'summary':s,'verification':audit},indent=2))
if __name__=='__main__':main()
