"""Publish conditional-reference inference comparison, no training or adoption."""
import _bootstrap
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(path):return json.loads(Path(path).read_text())


def main():
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_recovery_v3_seed61')
    summary=read(report/'summary.json');variants=summary['variants'];rows=[]
    groups=[('validation','Original validation (20)'),('combined_200','Known regression (200)')]
    if summary['fresh_test'] is not None:groups.append(('fresh_test','Fresh matched test (100)'))
    for group,_ in groups:
        for name in ('baseline','conditional'):
            metrics=(summary['fresh_test'][name]['stage_target'] if group=='fresh_test' else
                     variants[name]['validation']['stage_target'] if group=='validation' else variants[name]['combined_200'])
            rows.append({'group':group,'variant':name,**metrics})
    with (report/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,len(groups),figsize=(6*len(groups),4.5),layout='constrained')
    for ax,(group,title) in zip(np.atleast_1d(axes),groups):
        group_rows=[r for r in rows if r['group']==group];bottom=np.zeros(2)
        for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323'),('joint_limit','#777777')]:
            vals=np.array([r[key] for r in group_rows]);ax.bar(['Original','Conditional recovery'],vals,bottom=bottom,label=key,color=color);bottom+=vals
        ax.set(title=title,ylabel='Episodes',ylim=(0,group_rows[0]['episodes']*1.05));ax.legend()
    fig.savefig(report/'overview.png',dpi=150);plt.close(fig)
    lines=['# 停滞触发的受限路点恢复：固定权重对照','',
        '两组使用同一个当前保留 PPO 的全部网络权重。普通恢复半径仍为 0.30rad，终点伺服仍为 0.20rad，回合上限 240 步。',
        '仅条件组会在确认停滞且卡于非终点参考时，临时尝试 0.45rad 前向恢复；每回合最多成功一次、最多尝试三次，尝试间隔至少 20 步。',
        '触发前至少运行 60 步，并观察连续 30 步的位置、姿态、关节净位移与路点进展；不对无障碍对准任务、终止状态或动态场景启用该机制。',
        '改变的是局部参考观测规则，不是 PPO 重新训练或神经网络权重提升；无动作替换或在线专家避障。','',
        '| 场景组 | 设置 | 成功 | 碰撞 | 超时 | 关节越限 |','|---|---|---:|---:|---:|---:|']
    labels={'validation':'原验证','combined_200':'已见 200 回归','fresh_test':'新 100 场景'}
    for row in rows:lines.append(f'| {labels[row["group"]]} | {row["variant"]} | {row["success"]}/{row["episodes"]} | {row["collision"]} | {row["timeout"]} | {row["joint_limit"]} |')
    rescued=sum(c['paired_outcomes'].get('timeout -> success',0) for c in summary['comparisons'].values())
    identical=sum(c['untriggered_check']['identical_untriggered_episodes'] for c in summary['comparisons'].values())
    lines.extend(['',f'预定回归无退化检查：**{summary["non_regression"]["passed"]}**；细项：`{summary["non_regression"]["checks"]}`。',
        '', '## 结论',
        f'- 各批配对检查合计有 {rescued} 个超时转为成功，没有新碰撞或原成功转失败；{identical} 个未触发恢复的回合保持与原设置一致。',
        '- 已见 200 场景成功 182→188、碰撞仍为 10、超时 8→2；原验证成功 16→18、碰撞仍为 2、超时 2→0。',
        '- 新 100 场景成功 93→94、碰撞仍为 4、超时 3→2。新增场景上的改善只有一个回合，不能夸大为统计显著或训练后提升。',
        '- 结果支持把“停滞后受限恢复”作为下一步接入/微调的候选；本轮没有自动替换默认控制器或训练权重。',
        '', '## 触发与逐场景检查',''])
    for group,comparison in summary['comparisons'].items():
        lines.append(f'- {group}：`{comparison["untriggered_check"]}`；配对结果变化：`{comparison["paired_outcomes"]}`。')
    lines.extend(['','## 四个失败案例','', '| 种子 | 原设置 | 条件恢复 | 实际恢复次数 | 尝试事件 |','|---|---|---|---:|---|'])
    baseline=read(report/'baseline/validation.json');conditional=read(report/'conditional/validation.json')
    a={r['seed']:r for r in baseline['rows'] if r['suite']=='stage_target'}
    b={r['seed']:r for r in conditional['rows'] if r['suite']=='stage_target'}
    for seed in (900002,900003,900005,900015):
        lines.append(f'| {seed} | {a[seed]["reason"]}（{a[seed]["steps"]}步） | {b[seed]["reason"]}（{b[seed]["steps"]}步） | {b[seed]["conditional_recoveries"]} | {b[seed]["events"]} |')
    lines.extend(['','## 结果范围',
        '- 原验证及 200 回归场景已被查看，属于回归/调参证据，不是新独立测试。',
        '- 仅当回归不退化时才按预定规则运行新 100 场景；没有事后搜索参数。'])
    if summary['fresh_test'] is None:lines.append(f'- 本轮跳过新测试：{summary["fresh_test_skipped_reason"]}。')
    else:lines.append('- 新场景从种子 1004000 起，两组在同一批冻结场景上评估；该批结果不改变本轮已固定的恢复条件。')
    lines.extend(['- 没有修改默认配置、覆盖模型文件、运行训练或自动启动动态阶段。',
        '- 没有触发恢复的回合必须与原设置的终止原因、步数、误差完全匹配，检查结果已记录。','',
        '## 文件',
        '- `summary.json`、`comparison.csv`、`overview.png`：完整结果；`non_regression.json`：预定门槛。',
        '- 每组的 `validation.json` 包含四例完整动作/路点轨迹；两个回归 JSON 包含逐回合尝试与触发事件。',
        '- 若运行新测试，每组 `fresh_test.json` 为逐回合新测试记录。',
        '- `manifest.json` / `verification.json`：模型、代码、场景与终点参数核对；未导出伪装成新训练模型的 zip。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/conditional_recovery_v3_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'comparison':rows,'non_regression':summary['non_regression'],'paired':summary['comparisons']},indent=2))


if __name__=='__main__':main()
