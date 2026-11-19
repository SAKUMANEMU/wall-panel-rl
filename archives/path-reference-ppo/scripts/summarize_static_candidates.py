"""Publish matched inference-only comparison and timeout evidence."""
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
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/static_candidate_comparison_v3_seed61')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json');study=manifest['study']
    rows=[]
    for bank,groups in summary['results'].items():
        for model,metrics in groups.items():rows.append({'bank':bank,'model':model,**metrics['stage_target']})
    for model,metrics in summary['combined_200_case_descriptive_results'].items():rows.append({'bank':'combined_descriptive','model':model,**metrics})
    with (report/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,bank,title in [(axes[0],'known','Known-bank regression: 100 scenes'),(axes[1],'fresh','Fresh matched test: 100 scenes')]:
        bottom=np.zeros(2)
        for key,color in [('success','#2166ac'),('collision','#b2182b'),('timeout','#e6a323'),('joint_limit','#777777')]:
            values=np.array([summary['results'][bank][model]['stage_target'][key] for model in ('retained','candidate')])
            ax.bar(['Retained PPO','Final candidate'],values,bottom=bottom,label=key,color=color);bottom+=values
        ax.set(title=title,ylabel='Episodes',ylim=(0,105));ax.legend()
    fig.savefig(report/'overview.png',dpi=150);plt.close(fig)
    comparison=summary['validation_timeout_comparison'];details=comparison['candidate_timeouts']
    lines=['# 当前保留 PPO 与最终候选：超时诊断和扩大评估','',
        '本轮只做推理评估，没有训练、改变权重或自动切换模型。retained 为当前保留模型，candidate 为本轮最终零碰撞候选。',
        '原验证“零碰撞”仅指该候选在原 20 个验证场景中的 0 次碰撞，不是全场景保证。','',
        '## 原验证集变化','',f'逐场景结果变化：`{comparison["validation_transitions"]}`。','',
        '| 种子 | 原模型结果 | 候选位置误差 m | 姿态误差 ° | 末 40 步类型 | 路点已到最后 | 动作抵消比 | 延长至 480 步 |',
        '|---|---|---:|---:|---|---|---:|---|']
    for d in details:
        lines.append(f'| {d["seed"]} | {d["retained_reason"]} | {d["position_m"]:.4f} | {d["orientation_deg"]:.2f} | {d["kind"]} | {d["cursor_at_final_waypoint"]} | {d["mean_cancellation_ratio_last40"]:.4f} | {d["extended_reason"]} @ {d["extended_steps"]} |')
    lines.extend(['',
        '抵消比 = 动作均值范数 /（门控基础动作范数＋修正动作范数）；接近 0 表示两者相互抵消。',
        '停滞分类按最后 40 步运动与误差变化进行。路点不前进可能与偏离原参考路径有关，不能仅凭索引不变断言游标存在代码错误。',
        '延长回合只针对原超时案例作诊断，原 240 步正式评估标准和结果保持不变。','',
        '## 扩大配对评估','',
        '| 场景组 | 模型 | 成功 | 碰撞 | 超时 | 关节越限 |','|---|---|---:|---:|---:|---:|'])
    labels={'known':'已见 100 场景','fresh':'新 100 场景','combined_descriptive':'合计 200 场景（描述性）'}
    for r in rows:
        lines.append(f'| {labels[r["bank"]]} | {r["model"]} | {r["success"]}/{r["episodes"]} | {r["collision"]} | {r["timeout"]} | {r["joint_limit"]} |')
    retained=summary['combined_200_case_descriptive_results']['retained'];candidate=summary['combined_200_case_descriptive_results']['candidate']
    strong=sum(d['strong_action_cancellation'] for d in details)
    recovery=manifest['evaluation_settings']['reference_recovery']['radius']
    terminal=manifest['evaluation_settings']['terminal_radius']
    guard_overlap=sum(d['strong_action_cancellation'] and not d['cursor_at_final_waypoint'] and
                      d['route_distance_rad']>recovery and d['goal_max_joint_error_rad']>terminal for d in details)
    lines.extend(['','## 结论与下一步',
        f'- 保留模型合计成功 {retained["success"]}/200，候选 {candidate["success"]}/200；候选碰撞少 {retained["collision"]-candidate["collision"]} 次，但超时多 {candidate["timeout"]-retained["timeout"]} 次。本轮证据支持继续保留原模型。',
        f'- 原验证 4 个超时中 {strong} 个出现强动作抵消；{guard_overlap} 个同时未到最后路点、偏离原参考超过恢复半径 {recovery} rad，且未进入终点伺服范围（最大单关节误差 {terminal} rad 以内）。这提示参考恢复与终点伺服之间可能存在衔接问题。',
        '- 4 个案例延长到 480 步仍未成功，单纯延长回合不是本轮证据支持的主要改进方向。',
        '- 建议下一步先做参考恢复与终点伺服衔接的单项对照；本轮没有修改控制逻辑、参数或训练模型。',
        '', '新场景从种子 1002000 起生成，和训练、原验证及此前各批测试分开。模型在生成场景前已固定。',
        '合计 200 场景含 100 个已查看过的场景，不能将整个合计结果称为新的独立测试。','',
        '## 逐场景配对变化',''])
    for bank in ('known','fresh'):
        paired=read(report/f'{bank}_paired.json')
        lines.append(f'- {labels[bank]}：`{paired["transitions"]}`。')
    lines.extend(['','## 播放动画','',
        '已验证的当前保留模型成功案例：','',
        '```bash',
        'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/demo_autonomous_ppo.py --seed 1000000 --playback-speed 0.6 --hold-seconds 30',
        '```','',
        '查看候选时追加：`--model /home/johnsonchen/wall-panel-rl/results/diagnostics/static_stability_v3_seed61/final_candidate_unselected.zip`。',
        '动画使用与评估相同的自主策略环境和已保存的无障碍参考，不会换成专家避障路线。','',
        '## 文件',
        '- `summary.json`、`comparison.csv`、`overview.png`：全部比较结果。',
        '- `timeout_comparison.json`、`*_validation_traces.json`、`extended_timeout_*.json`：路点、动作分解、间距及完整轨迹。',
        '- `known_*.json`、`fresh_*.json`：逐回合结果和配对变化；模型及原状态不被覆盖。',
        '- `manifest.json`：模型、代码和配置；`status.json`：完成状态。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/static_candidate_comparison_v3_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'timeout_comparison':comparison,'evaluation':summary['results'],
                      'combined':summary['combined_200_case_descriptive_results']},indent=2))


if __name__=='__main__':main()
