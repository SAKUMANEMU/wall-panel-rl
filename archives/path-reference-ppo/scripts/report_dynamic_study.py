"""Summarize completed dynamic70 study from recorded outcomes only."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    folder=Path('results/raw/dynamic70_main')
    rows=pd.read_csv(folder/'episodes.csv')
    root=Path('models/dynamic70_study')
    expected={(condition,method) for condition in ['clean','noise']
              for method in ['rrt','ppo_clean','ppo_noise']}
    counts=rows.groupby(['evaluation_noise','method']).size()
    if set(counts.index)!=expected or not (counts==100).all():
        raise RuntimeError('Report requires all six completed 100-episode test cells')
    if rows.duplicated(['evaluation_noise','method','seed']).any():
        raise RuntimeError('Duplicate benchmark episodes')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    colors={'rrt':'#777777','ppo_clean':'#2474b5','ppo_noise':'#cc7a22'}
    for i,condition in enumerate(['clean','noise']):
        for j,method in enumerate(colors):
            r=rows[(rows.evaluation_noise==condition)&(rows.method==method)]
            axes[0].bar(i*4+j,100*r.success.mean(),color=colors[method],label=method if i==0 else None)
    axes[0].set(xticks=[1,5],xticklabels=['Clean observations','Noisy observations'],ylabel='Success (%)',ylim=(0,105))
    axes[0].legend(fontsize=8)
    for condition in ['clean','noise']:
        model=root/f'ppo_{condition}'
        v=pd.read_json(model/'validation.jsonl',lines=True)
        axes[1].plot(v.steps,v.success_rate*100,label=condition)
        pieces=[]
        for phase,start in [('alignment',0),('static',200000),('dynamic',400000)]:
            frames=[pd.read_csv(p,skiprows=1) for p in model.glob(f'{phase}_*.monitor.csv')]
            for f in frames:f['step']=start+4*f.l.cumsum()
            combined=pd.concat(frames).sort_values('step')
            pieces.append(combined)
        curve=pd.concat(pieces).sort_values('step')
        axes[2].plot(curve.step,curve.r.rolling(100,min_periods=20).mean(),label=condition)
    axes[1].set(xlabel='Training steps',ylabel='Validation success (%)',ylim=(0,105));axes[1].legend()
    axes[2].set(xlabel='Approximate aggregate training steps',ylabel='Episode reward (100-episode mean)');axes[2].legend()
    fig.tight_layout();fig.savefig(folder/'overview.png',dpi=180);plt.close(fig)
    lines=['# Dynamic PPO comparison with 70 observations','',
           'All figures below are computed from recorded episodes. Test scenes were held out from training and checkpoint selection.',
           '', '| Method | Observation noise | N | Success | Collision | Position error (mm) | Orientation error (deg) |',
           '|---|---|---:|---:|---:|---:|---:|']
    for (condition,method),group in rows.groupby(['evaluation_noise','method']):
        lines.append(f'| {method} | {condition} | {len(group)} | {100*group.success.mean():.1f}% | {100*group.collision.mean():.1f}% | {1000*group.position_error_m.mean():.2f} | {group.orientation_error_deg.mean():.2f} |')
    lines+=['','## Failure accounting','',
            '| Method | Observation noise | Initial planning failures | Collisions | Joint-limit events | Timeouts |',
            '|---|---|---:|---:|---:|---:|']
    for (condition,method),group in rows.groupby(['evaluation_noise','method']):
        planning=(group.planning_success==False)
        timeout=(~group.success & ~group.collision & ~group.joint_limit_violation & ~planning)
        lines.append(f'| {method} | {condition} | {int(planning.sum())} | {int(group.collision.sum())} | {int(group.joint_limit_violation.sum())} | {int(timeout.sum())} |')
    lines+=['','Collision and joint-limit counts can overlap; planning failures and timeouts are separate categories.']
    lines+=['','## Training provenance','']
    for condition in ['clean','noise']:
        path=root/f'ppo_{condition}'
        training=json.loads((path/'training.json').read_text());selection=json.loads((path/'selection.json').read_text())
        lines.append(f'- PPO-{condition}: {training["steps"]:,} environment steps, seed {training["seed"]}, selected checkpoint at {selection["steps"]:,} steps. Training elapsed {training["elapsed_seconds"]/3600:.2f} hours.')
    lines+=['','Reward, tolerance, and final scene difficulty were not relaxed. Errors include failed episodes.',
            'RRT knows prescribed obstacle motion; PPO observes position and velocity but not phases/frequencies. Planning pauses simulation time.',
            'One training seed per condition does not establish general superiority or convergence. The learning-curve step locations combine four worker monitors approximately.',
            '', '![Overview](overview.png)']
    (folder/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
