"""Publish completed unchanged-parameter static PPO stability results."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(Path(path).read_text())


def main():
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/static_stability_v3_seed61')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json');study=manifest['study']
    output=Path(study['output_root'])/'ppo';history=summary['checkpoint_history'];rows=[]
    for event in history:
        target=event['metrics']['stage_target']
        rows.append({'additional_steps':event['additional_steps'],**{k:target[k] for k in
            ('episodes','success','collision','timeout','joint_limit')},
            'alignment_success':event['metrics']['alignment_far']['success']+event['metrics']['alignment_near']['success']})
    with (report/'checkpoint_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    monitors=pd.concat([pd.read_csv(p,comment='#') for p in sorted(output.glob('worker_*.monitor.csv'))],ignore_index=True)
    training={task:{'episodes':len(group),'reasons':dict(Counter(group['termination_reason']))}
              for task,group in monitors.groupby('curriculum_task')}
    (report/'training_episodes.json').write_text(json.dumps(training,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323')]:
        axes[0].plot([r['additional_steps'] for r in rows],[100*r[key]/r['episodes'] for r in rows],
                     'o-',label=key,color=color)
    axes[0].set(title='Unchanged-parameter static PPO',xlabel='Additional PPO environment steps',ylabel='Original validation (%)',ylim=(0,105))
    axes[0].grid(alpha=.25);axes[0].legend()
    bottom=np.zeros(2)
    for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323'),('joint_limit','#777777')]:
        vals=np.array([summary['regression'][n]['stage_target'][key] for n in ('baseline','selected')])
        axes[1].bar(['Starting model','Selected model'],vals,bottom=bottom,label=key,color=color);bottom+=vals
    axes[1].set(title='Known-bank regression (100 scenes)',ylabel='Episodes',ylim=(0,105));axes[1].legend()
    fig.savefig(report/'overview.png',dpi=150);plt.close(fig)
    assessment=summary['promotion_assessment']
    lines=['# 静态 PPO 稳定性续训结果','',
        '从最新墙面反馈 PPO 检查点追加 10,000 个环境步；物理环境、学习率、墙面反馈、停滞反馈等全部沿用上一轮设置。',
        '原验证集检查 0/5000/10000 步；基线也参与选模，避免训练退化覆盖原模型。','',
        '| 追加步数 | 成功 | 碰撞 | 超时 | 关节越限 | 无障碍对准 |','|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'| {r["additional_steps"]} | {r["success"]}/20 | {r["collision"]}/20 | {r["timeout"]}/20 | {r["joint_limit"]} | {r["alignment_success"]}/20 |')
    lines.extend(['',f'完成追加环境步：{summary["completed_additional_steps"]}；选中追加步数：{summary["selected_additional_steps"]}。',
        f'连续两次后更新检查及所选模型共同满足阶段门槛：**{assessment["passed"]}**。',
        '门槛保持成功率至少 80%、碰撞率不超过 10%、无关节越限及对准保留。两次必须对应不同更新步数，初始模型不算后更新检查。',
        '本轮只验证静态阶段；无自动动态训练或额外预算。','',
        '## 已见场景回归检查','',
        '下表使用上一轮已查看过的同一批 100 个场景，不是新的独立测试，也不参与检查点选择。','',
        '| 模型 | 成功 | 碰撞 | 超时 |','|---|---:|---:|---:|'])
    for name,label in [('baseline','起始模型'),('selected','选中模型')]:
        m=summary['regression'][name]['stage_target']
        lines.append(f'| {label} | {m["success"]}/100 | {m["collision"]}/100 | {m["timeout"]}/100 |')
    lines.extend(['','## 文件',
        '- `selected_model.zip` / `selected_model.json`：模型及说明；`promotion_assessment.json`：连续检查结果。',
        '- `final_candidate_unselected.zip`：完成追加 10,000 步后的最终候选，原验证 16/20 成功、0 碰撞、4 超时；因超时增加而未选中。',
        '- `summary.json`、`checkpoint_summary.csv`、`overview.png`：指标和曲线。',
        '- `regression_*.json`：已见场景逐回合回归记录；`training_episodes.json`：随机训练回合统计，与确定性评估分开。',
        '- `manifest.json`：不变设置、输入与代码哈希；训练前与训练后模型全部保留。',
        '- 追加步数为 0 的所选模型表示本轮完成训练，但后续模型没有超过基线。',
        f'- 原始模型目录：`{output.resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/static_stability_v3_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'selected_step':summary['selected_additional_steps'],'promotion_passed':assessment['passed'],
                      'validation':rows,'regression':summary['regression'],'training':training},indent=2))


if __name__=='__main__':main()
