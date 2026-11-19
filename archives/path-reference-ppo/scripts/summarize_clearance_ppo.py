"""Report completed matched PPO wall-feedback pilots; no training or selection."""
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
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/clearance_ppo_v3_seed61')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json');study=manifest['study'];root=Path(study['output_root'])
    rows=[];training={}
    for name in study['variants']:
        for line in (root/name/'ppo/validation.jsonl').read_text().splitlines():
            event=json.loads(line);m=event['metrics']['dynamic_ppo']['stage_target']
            rows.append({'variant':name,'additional_steps':event['additional_steps'],
                **{k:m[k] for k in ('episodes','success','collision','timeout','joint_limit')},
                'eligible_no_regression':bool(event['selection_score'][0]),'bc_gate':event['gate']['passed']})
        monitor=pd.concat([pd.read_csv(p,comment='#') for p in sorted((root/name/'ppo').glob('worker_*.monitor.csv'))],ignore_index=True)
        training[name]={task:{'episodes':len(group),'reasons':dict(Counter(group['termination_reason']))}
                        for task,group in monitor.groupby('curriculum_task')}
    with (report/'checkpoint_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (report/'training_episodes.json').write_text(json.dumps(training,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for ax,name,label in [(axes[0],'control','Continued PPO: control'),(axes[1],'wall_feedback','Continued PPO: wall feedback')]:
        group=[r for r in rows if r['variant']==name];x=[r['additional_steps'] for r in group]
        for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323')]:
            ax.plot(x,[100*r[key]/r['episodes'] for r in group],'o-',label=key,color=color)
        ax.set(title=label,xlabel='Additional PPO environment steps',ylabel='Original validation (%)',ylim=(0,105))
        ax.grid(alpha=.25);ax.legend()
    bottom=np.zeros(3);names=['baseline','control','wall_feedback']
    for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323'),('joint_limit','#777777')]:
        vals=np.array([summary['independent'][n]['stage_target'][key] for n in names])
        axes[2].bar(['Baseline','Control','Wall feedback'],vals,bottom=bottom,label=key,color=color);bottom+=vals
    axes[2].set(title='New independent test (100 scenes)',ylabel='Episodes',ylim=(0,105));axes[2].legend()
    fig.savefig(report/'overview.png',dpi=150);plt.close(fig)
    lines=['# 墙面碰撞反馈：PPO 对照实验','',
        '两组从同一最新 PPO 权重和优化器出发，各追加 10,000 个环境步。唯一训练设置差异为墙面反馈权重 0 与 1。',
        '回合上限 240 步；保留停滞反馈与原障碍物惩罚；选模和评估使用原奖励。',
        '', '| 模型 | 原验证成功 | 碰撞 | 超时 | 新测试成功 | 碰撞 | 超时 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('baseline','起始 PPO'),('control','继续 PPO，对照'),('wall_feedback','继续 PPO＋墙面反馈')]:
        val=summary['validation'][name]['metrics']['dynamic_ppo']['stage_target'];test=summary['independent'][name]['stage_target']
        lines.append(f'| {label} | {val["success"]}/20 | {val["collision"]}/20 | {val["timeout"]}/20 | {test["success"]}/100 | {test["collision"]}/100 | {test["timeout"]}/100 |')
    lines.extend(['','## 诊断与预定规则',
        '- 起始模型训练集 5 次碰撞全部为墙板撞墙；原验证集 2 次撞墙、1 次撞障碍物；旧测试 13 次碰撞中 12 次撞墙、1 次自碰撞。',
        '- 因此本轮增加训练专用墙面反馈，原障碍物距离惩罚仍保持 0.5。2cm 内按平方渐增，正确安装的名义间距 2.5cm 不受惩罚。',
        '- 两组相同的种子和起始检查点；只保留对准能力、成功数不下降、碰撞和超时不增加的新检查点，再优先选碰撞较少者。',
        '- 初始模型也参与选模；选中追加步数为 0 表示训练完成但保留基线。',
        f'- 选中追加步数：{summary["selected_additional_steps"]}。按原验证集锁定的总体模型：`{summary["winner_by_validation"]}`。',
        '- 新测试从种子 1000000 起，和训练、验证及旧 980000/990000 测试分开；独立成绩不参与选模。',
        '', '## 检查点','', '| 组别 | 追加环境步 | 成功 | 碰撞 | 超时 | 无退化资格 |','|---|---:|---:|---:|---:|---|'])
    for r in rows:lines.append(f'| {r["variant"]} | {r["additional_steps"]} | {r["success"]}/20 | {r["collision"]}/20 | {r["timeout"]}/20 | {r["eligible_no_regression"]} |')
    lines.extend(['','## 阶段门槛与文件',
        '- 静态推进门槛仍为至少 80% 成功、不超过 10% 碰撞、对准保留，以及两次不同更新步数的连续通过；不自动启动下一阶段。',
        f'- 本轮连续检查和选中模型共同通过情况：{ {k:v["passed"] for k,v in summary["promotion_assessments"].items()} }。',
        '- `selected_model.zip` / `selected_model.json`：选定模型及说明；`summary.json`：原验证和新测试指标。',
        '- `collision_diagnosis.json`：碰撞对象、间距和动作幅度；`checkpoint_summary.csv`、`overview.png`：曲线。',
        '- `independent_*.json`：逐回合新测试记录；`training_episodes.json`：随机训练回合，与确定性评估分开。',
        '- 一个训练种子和重复使用的 20 个验证场景属于工程试验，不能据此宣称统计显著或五个动态障碍成功。',
        '- 新测试集一旦查看结果，今后已不再是未见测试集。',
        f'- 模型目录：`{root.resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/clearance_ppo_v3_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'winner':summary['winner_by_validation'],'selected_steps':summary['selected_additional_steps'],
                      'independent':summary['independent'],'training':training},indent=2))


if __name__=='__main__':main()
