"""Summarize completed waypoint pilot results without rerunning evaluation."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(1)
    root=Path('models/waypoint_ppo_pilot_seed61')
    training=json.loads((root/'training.json').read_text())
    history=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
    assert history[-1]['kind']=='final_saved_reload'
    assert training['additional_steps']==10000
    rows=[]
    for event in history:
        for method,suites in event['metrics'].items():
            for suite,metrics in suites.items():
                rows.append({'additional_steps':event['additional_steps'],'kind':event['kind'],
                    'method':method,'suite':suite,**metrics})
    frame=pd.DataFrame(rows)
    output=Path('results/diagnostics/waypoint_ppo_pilot_seed61');output.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output/'validation_summary.csv',index=False)
    monitors=pd.concat([pd.read_csv(p,skiprows=1) for p in root.glob('worker_*.monitor.csv')],ignore_index=True)
    source=PPO.load(root/'checkpoints/checkpoint_0.zip',device='cpu')
    final=PPO.load(root/'final_model.zip',device='cpu')
    changes={name:float((value-source.policy.state_dict()[name]).abs().max())
             for name,value in final.policy.state_dict().items()}
    assert changes['tracking_velocity']==0 and changes['log_std']==0
    summary={'training':training,'completed_training_episodes':len(monitors),
        'training_termination_counts':monitors.termination_reason.value_counts().to_dict(),
        'selection':json.loads((root/'selection.json').read_text()),
        'parameter_max_absolute_changes':changes}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for method in ('ordinary','bc','ppo'):
        group=frame[(frame.method==method)&(frame.suite=='static')]
        axes[0].plot(group.additional_steps,100*group.success_rate,'o-',label=method)
        axes[1].plot(group.additional_steps,group.successful_mean_steps,'o-',label=method)
    for suite in ('alignment_far','alignment_near'):
        group=frame[(frame.method=='ppo')&(frame.suite==suite)]
        axes[2].plot(group.additional_steps,100*group.success_rate,'o-',label=suite)
    for ax in axes:ax.set_xlabel('Additional PPO transitions');ax.legend(fontsize=8)
    axes[0].set(ylabel='Static success (%)',ylim=(0,105))
    axes[1].set(ylabel='Mean control steps, successes only',ylim=(0,70))
    axes[2].set(ylabel='Alignment success (%)',ylim=(0,105))
    fig.tight_layout();fig.savefig(output/'overview.png',dpi=180);plt.close(fig)
    lines=['# Static waypoint PPO pilot results','',
        f"Completed {training['additional_steps']:,} additional transitions; {len(monitors)} completed training episodes; elapsed {training['elapsed_seconds']/60:.2f} minutes.",'',
        '| Additional steps | Method | Static successes | Collisions | Planning failures | Mean successful control steps | Far alignment | Near alignment |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for event in history:
        for method,m in event['metrics'].items():
            s=m['static'];n=s['episodes']
            lines.append(f"| {event['additional_steps']} | {method} | {round(s['success_rate']*n)}/{n} | {round(s['collision_rate']*n)} | {round(s['planning_failure_rate']*n)} | {s['successful_mean_steps']} | {m['alignment_far']['success_rate']:.0%} | {m['alignment_near']['success_rate']:.0%} |")
    lines += ['',f"Selected checkpoint: {summary['selection']['additional_steps']} additional steps.",
        'Selection prioritizes static success, collision rate, alignment retention, then successful control steps. A checkpoint at zero means retaining the BC initialization.',
        'All comparisons share a freshly planned path per scene within a validation event. Finite planning timeouts can change between events. These reused validation scenes are not an independent generalization test.',
        'The actor learns servo gain/bias coefficients; route planning remains RRT. No independent test, dynamic extension or automatic longer training is included.',
        '', '![Overview](overview.png)','']
    (output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
