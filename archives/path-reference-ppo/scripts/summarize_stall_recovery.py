"""Create the completed stall study report/curves without fitting or selecting."""
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
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/stall_recovery_v3_seed61')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json');study=manifest['study']
    root=Path(study['output_root']);data=Path(study['data_root'])
    bc=read(report/'bc_low_lr.json');diagnosis=read(report/'train_diagnosis.json')
    recoveries=read(data/'corrections/metadata.json');attempts=read(data/'corrections/attempts.json')
    replay=read(report/'correction_replay.json');rows=[]
    for line in (root/'bc_low_lr/closed_loop_history.jsonl').read_text().splitlines():
        event=json.loads(line);g=event['validation'];m=g['metrics']['dynamic_ppo']['stage_target']
        rows.append({'variant':'bc_low_lr','additional':event['update'],**{k:m[k] for k in
            ('episodes','success','collision','timeout','joint_limit')},'gate_passed':g['passed']})
    for line in (root/'ppo/validation.jsonl').read_text().splitlines():
        event=json.loads(line);g=event['gate'];m=g['metrics']['dynamic_ppo']['stage_target']
        rows.append({'variant':'ppo_stall','additional':event['additional_steps'],**{k:m[k] for k in
            ('episodes','success','collision','timeout','joint_limit')},'gate_passed':g['passed']})
    with (report/'checkpoint_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    monitor=pd.concat([pd.read_csv(p,comment='#') for p in sorted((root/'ppo').glob('worker_*.monitor.csv'))],ignore_index=True)
    training={task:{'completed_episodes':len(group),'reasons':dict(Counter(group['termination_reason']))}
              for task,group in monitor.groupby('curriculum_task')}
    (report/'ppo_completed_episodes.json').write_text(json.dumps(training,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for axis,name,title in [(axes[0],'bc_low_lr','Low-LR BC continuation'),(axes[1],'ppo_stall','PPO with stagnation feedback')]:
        group=[r for r in rows if r['variant']==name];x=[r['additional'] for r in group]
        for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323')]:
            axis.plot(x,[100*r[key]/r['episodes'] for r in group],'o-',label=key,color=color)
        axis.set(title=title,xlabel='Additional BC updates' if name=='bc_low_lr' else 'Additional PPO environment steps',ylabel='Original validation (%)',ylim=(0,105))
        axis.grid(alpha=.25);axis.legend()
    names=['baseline','bc_low_lr','ppo_stall'];bottom=np.zeros(3)
    for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323'),('joint_limit','#777777')]:
        vals=np.array([summary['independent'][n]['stage_target'][key] for n in names])
        axes[2].bar(['Baseline','BC','PPO'],vals,bottom=bottom,label=key,color=color);bottom+=vals
    axes[2].set(title='New independent test (100 scenes)',ylabel='Episodes',ylim=(0,105));axes[2].legend()
    fig.savefig(report/'overview.png',dpi=150);plt.close(fig)
    lines=['# 停滞恢复：低学习率 BC 与短预算 PPO 对照','',
        '两组从同一个原验证 65% 的 BC 模型独立出发。回合上限 240 步，成功标准和碰撞判定保持原定义。',
        'PPO 工作者使用额外停滞奖励；选模与报告中的验证使用原奖励。新独立测试在两组选模完成后执行。','',
        '| 模型 | 原验证成功 | 碰撞 | 超时 | 新独立成功 | 碰撞 | 超时 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('baseline','当前 BC 基线'),('bc_low_lr','低学习率 BC＋停滞示范'),('ppo_stall','PPO＋停滞反馈')]:
        val=summary['validation'][name]['metrics']['dynamic_ppo']['stage_target'];test=summary['independent'][name]['stage_target']
        lines.append(f'| {label} | {val["success"]}/20 | {val["collision"]}/20 | {val["timeout"]}/20 | {test["success"]}/100 | {test["collision"]}/100 | {test["timeout"]}/100 |')
    lines.extend(['','## 诊断、数据与预算',
        f'- 54 个训练布局的初始结果：{diagnosis["outcomes"]}。停滞分解见 `train_diagnosis.json`，多个回合基础动作与修正动作近乎抵消。',
        f'- {len(attempts)} 次恢复尝试，{recoveries["episodes"]} 条成功示范、{recoveries["transitions"]} 条专家动作标签。',
        f'- 回放检查：{replay["success"]}/{replay["sampled_corrections"]} 成功，最大观测差 {replay["max_observation_error"]}。',
        '- 示范只来自训练场景；原监督 held-out 布局不参与恢复采集。前缀动作不作为专家标签。',
        f'- BC：完成 {summary["bc_updates_completed"]} 次追加更新，学习率 1e-4；选中追加更新 {summary["selected_bc_additional_updates"]}。',
        f'- PPO：完成 {summary["ppo_steps_completed"]} 个追加环境步；选中追加步数 {summary["selected_ppo_additional_steps"]}。完整回合统计见 `ppo_completed_episodes.json`，不与评估成功率混淆。',
        '- 选中追加步数为 0 表示该组完成训练但保留初始模型，避免以退化的最终检查点覆盖它。',
        f'- 按原验证规则锁定的总体模型：`{summary["winner_by_validation"]}`，导出为 `selected_model.zip`。独立测试不参与选模。',
        '', '## 检查点','', '| 组别 | 追加更新或环境步 | 成功 | 碰撞 | 超时 | BC 门槛 |',
        '|---|---:|---:|---:|---:|---|'])
    for r in rows:lines.append(f'| {r["variant"]} | {r["additional"]} | {r["success"]}/20 | {r["collision"]}/20 | {r["timeout"]}/20 | {r["gate_passed"]} |')
    lines.extend(['','## 限制和文件',
        '- 当前仅为单个静态障碍物；没有自动进入动态阶段，也没有增加预算。',
        '- 这是两种成套改进的实践比较，不能单独归因于算法、学习率、奖励或数据。只有一个训练随机种子。',
        '- 新测试从种子 990000 起，和此前已见的 980000 测试分开；本轮之后这批场景也已不再是未见测试。',
        '- `summary.json`：全部指标；`independent_*.json`：独立测试逐回合；`checkpoint_summary.csv`、`overview.png`：曲线。',
        '- `manifest.json`：初始化、代码和配置；`bc_low_lr.json`、`ppo_selection.json`：选模细节；`correction_replay.json`：示范回放。',
        f'- 模型目录：`{root.resolve()}`；数据目录：`{data.resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/stall_recovery_v3_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(report),'winner':summary['winner_by_validation'],'independent':summary['independent'],'training_episodes':training},indent=2))


if __name__=='__main__':main()
