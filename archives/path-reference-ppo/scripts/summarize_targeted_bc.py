"""Publish completed targeted BC outcomes, curves and provenance (no training)."""
import _bootstrap
import csv
from collections import Counter
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(path.read_text())


def main():
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/targeted_bc_v3_seed61')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json')
    study=manifest['study'];root=Path(study['output_root']);data=Path(study['data_root'])
    diagnosis=read(report/'collision_diagnosis.json');replay=read(report/'correction_replay.json')
    traces=read(report/'validation_traces.json')
    contact_episodes=Counter(kind for t in traces if t['reason']=='collision'
        for kind in {c['moving']+'_'+c['kind'] for c in t['contacts']})
    corrections=read(data/'corrections/metadata.json');attempts=read(data/'corrections/attempts.json')
    rows=[];histories={}
    for name in ('original_only','targeted_mix'):
        history=[json.loads(line) for line in (root/name/'bc/closed_loop_history.jsonl').read_text().splitlines()]
        histories[name]=history
        for event in history:
            target=event['validation']['metrics']['dynamic_ppo']['stage_target']
            align=event['validation']['metrics']['dynamic_ppo']
            rows.append({'variant':name,'additional_updates':event['update'],
                'cumulative_updates':event['cumulative_bc_updates'],'episodes':target['episodes'],
                **{k:target[k] for k in ('success','collision','timeout','joint_limit')},
                'alignment_success':align['alignment_far']['success']+align['alignment_near']['success'],
                'gate_passed':event['validation']['passed'],
                'heldout_avoidance_mse':event['losses']['heldout_task_mse']['avoidance']})
    with (report/'checkpoint_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for name,label,color in [('original_only','Original data','#2166ac'),('targeted_mix','Targeted corrections','#b2182b')]:
        group=[r for r in rows if r['variant']==name];x=[r['additional_updates'] for r in group]
        axes[0].plot(x,[100*r['success']/r['episodes'] for r in group],'o-',color=color,label=label)
        axes[1].plot(x,[100*r['collision']/r['episodes'] for r in group],'o-',color=color,label=label)
        axes[2].plot(x,[r['heldout_avoidance_mse'] for r in group],'o-',color=color,label=label)
    for ax,title in zip(axes,['Validation success (20 scenes)','Validation collision (20 scenes)','Same held-out expert states']):
        ax.set_title(title);ax.set_xlabel('Additional BC updates');ax.grid(alpha=.25)
    axes[0].set_ylabel('Success (%)');axes[1].set_ylabel('Collision (%)');axes[2].set_ylabel('Action MSE')
    axes[0].set_ylim(0,105);axes[1].set_ylim(0,105);axes[0].legend()
    fig.savefig(report/'overview.png',dpi=150);plt.close(fig)
    lines=['# 针对性纠偏与 BC 续训对照结果','',
        '两组均从原 50% BC 检查点恢复权重、BC Adam 与采样随机状态，各追加 10,000 次更新；PPO 环境步数为 0。',
        '原 20 个验证场景用于选模，独立 100 个场景仅在两组选模完成后评估。',
        '','| 模型 | 选中追加更新 | 原验证成功 | 原验证碰撞 | 独立成功 | 独立碰撞 |','|---|---:|---:|---:|---:|---:|']
    independent=summary['independent'];base=independent['baseline']['stage_target']
    lines.append(f'| 原 BC 基线 | 0 | 10/20 | 9/20 | {base["success"]}/100 | {base["collision"]}/100 |')
    for name,label in [('original_only','原数据续训'),('targeted_mix','原数据＋针对性纠偏')]:
        selected=summary['selected'][name];target=selected['validation']['stage_target'];test=independent[name]['stage_target']
        lines.append(f'| {label} | {selected["selected_additional_updates"]} | {target["success"]}/20 | {target["collision"]}/20 | {test["success"]}/100 | {test["collision"]}/100 |')
    leads=corrections['warmup_steps'];success=[a for a in attempts if a['reason']=='success']
    late=sum(k>12 for k in leads)
    lines.extend(['','## 诊断和数据',f'- 原验证集碰撞步数：{diagnosis["validation"]["collision_steps"]}。',
        f'- 原验证集碰撞对象（每回合、每类型计一次）：{dict(contact_episodes)}；未能分类：{diagnosis["validation"]["unclassified_collisions"]}。',
        f'- 纠偏：{len(attempts)} 次专家尝试，{corrections["episodes"]} 条成功恢复轨迹，{corrections["transitions"]} 条动作标签。',
        f'- 接管发生在策略执行 {min(leads)}–{max(leads)} 步后，其中 {late} 条超过 12 步；失败前缀不作为示范标签。',
        f'- 抽样回放：{replay["success"]}/{replay["sampled_corrections"]} 成功，最大观测差 {replay["max_observation_error"]}。',
        '- 纠偏仅来自训练布局；原监督 held-out 布局保持隔离。两组使用相同 held-out 专家状态，因此 MSE 可直接对照。',
        '', '## 检查点','', '| 数据组 | 追加更新 | 成功 | 碰撞 | 超时 | 对准成功 | BC 放行 |',
        '|---|---:|---:|---:|---:|---:|---|'])
    for row in rows:
        lines.append(f'| {row["variant"]} | {row["additional_updates"]} | {row["success"]}/20 | {row["collision"]}/20 | {row["timeout"]} | {row["alignment_success"]}/20 | {row["gate_passed"]} |')
    lines.extend(['','## 文件与解释',
        f'- 按原验证集预定规则选出的模型：`selected_bc_model.zip`，来源 `{summary["winner_by_original_validation"]}`；独立成绩不参与选模。',
        '- `summary.json`：选模和独立评估；`collision_diagnosis.json`、`validation_traces.json`、`train_traces.json`：碰撞与完整执行轨迹。',
        '- `checkpoint_summary.csv`、`overview.png`：检查点曲线；`correction_replay.json`：物理回放核对。',
        '- `original_only.json`、`targeted_mix.json`：完整 BC 元数据与保留原先验/critic 的检查。',
        '- `independent_*.json`：冻结独立场景上的逐回合结果；这些场景今后已不再是未见测试集。',
        '- 这仍是单个静态障碍物阶段，不代表五个动态障碍物的成绩。只有一个训练随机种子，不能据此宣称统计显著。',
        '- 若选中追加更新为 0，表示训练后检查点未超过初始 BC；完整预算已执行，训练前模型被保留。',
        f'- 模型目录：`{root.resolve()}`；数据目录：`{data.resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/targeted_bc_v3_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(report),'summary':summary},indent=2))


if __name__=='__main__':main()
