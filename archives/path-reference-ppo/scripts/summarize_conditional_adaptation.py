"""Publish the bounded PPO continuation, keeping controller gains separate."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from evaluate_reference_handoff import tensor_fingerprint


def main():
    torch.set_num_threads(1)
    report=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_ppo_adaptation_v3_seed61')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json')
    assert read(report/'status.json')['status']=='completed'
    root=Path(manifest['study']['output_root'])/'ppo'
    initial=PPO.load(manifest['study']['initial_model'],device='cpu')
    selected=PPO.load(report/'selected_model.zip',device='cpu')
    final=PPO.load(report/'final_candidate.zip',device='cpu')
    frozen=[name for name,param in initial.policy.named_parameters() if not param.requires_grad]
    a=initial.policy.state_dict();b=final.policy.state_dict()
    paired={}
    for bank in summary['regression']['conditional_before']:
        before=read(report/'conditional_before'/f'{bank}.json')['rows']
        after=read(report/'conditional_selected'/f'{bank}.json')['rows']
        before={(r['suite'],r['seed']):r for r in before}
        after={(r['suite'],r['seed']):r for r in after}
        assert before.keys()==after.keys()
        paired[bank]=all(all(before[k][v]==after[k][v] for v in
            ('reason','steps','position_m','orientation_deg','conditional_attempts','conditional_recoveries')) for k in before)
    verification={
        'source_file_unchanged':sha(manifest['study']['initial_model'])==manifest['initial_model_sha256'],
        'inputs_unchanged':all(sha(p)==v for p,v in manifest['input_hashes'].items()),
        'training_code_unchanged':all(sha(p)==v for p,v in manifest['code_hashes'].items()),
        'selected_file_matches_export':sha(report/'selected_model.zip')==summary['export_sha256'],
        'selected_tensors_equal_source':tensor_fingerprint(initial.policy)==tensor_fingerprint(selected.policy),
        'final_tensors_updated':tensor_fingerprint(initial.policy)!=tensor_fingerprint(final.policy),
        'frozen_parameter_count':len(frozen),
        'frozen_parameters_unchanged':bool(frozen) and all(torch.equal(a[n],b[n]) for n in frozen),
        'stored_ppo_counters':{'initial':initial.num_timesteps,'selected':selected.num_timesteps,'final':final.num_timesteps},
        'regression_per_episode_identical':paired,
        'selected_model_requires_conditional_controller':True,
        'automatic_stage_advance':False,
    }
    checks=('source_file_unchanged','inputs_unchanged','training_code_unchanged','selected_file_matches_export',
            'selected_tensors_equal_source','final_tensors_updated','frozen_parameters_unchanged')
    assert all(verification[k] for k in checks) and all(paired.values())
    dump(report/'verification.json',verification)

    episodes=[]
    for path in sorted(root.glob('worker_*.monitor.csv')):
        with path.open() as stream:
            next(stream);episodes.extend(dict(row,worker=path.stem) for row in csv.DictReader(stream))
    training={'complete_episodes':len(episodes),'by_task':dict(Counter(r['curriculum_task'] for r in episodes)),
        'outcomes':dict(Counter(r['termination_reason'] for r in episodes)),
        'steps_in_complete_episodes':sum(int(r['l']) for r in episodes),
        'all_transitions':summary['sampling_events']['observed_transitions'],
        'sampling_events':summary['sampling_events']}
    training['steps_in_partial_episodes']=training['all_transitions']-training['steps_in_complete_episodes']
    dump(report/'training_episode_summary.json',training)
    with (report/'training_episodes.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episodes[0]));writer.writeheader();writer.writerows(episodes)

    rows=[]
    for entry in summary['history']:
        rows.append({'additional_steps':entry['additional_steps'],**entry['metrics']['stage_target']})
    with (report/'validation_curve.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    colors={'success':'#2166ac','collision':'#b2182b','timeout':'#d68b00'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    for key,color in colors.items():
        axes[0].plot([r['additional_steps'] for r in rows],
            [100*r[key]/r['episodes'] for r in rows],marker='o',color=color,label=key)
    axes[0].axhline(80,color=colors['success'],linestyle=':',alpha=.6)
    axes[0].axhline(10,color=colors['collision'],linestyle=':',alpha=.6)
    axes[0].set(title='Conditional execution: PPO validation (20 cases)',
        xlabel='Additional PPO transitions',ylabel='Rate (%)',ylim=(-3,105),xticks=[0,5000,10000])
    axes[0].legend(loc='center right');axes[0].grid(alpha=.2)
    groups=[summary['combined_300_regression'][k] for k in ('conditional_before','conditional_selected')]
    bottom=np.zeros(2)
    for key,color in colors.items():
        values=np.array([r[key] for r in groups]);axes[1].bar(['Before adaptation','Retained (step 0)'],values,bottom=bottom,color=color,label=key)
        for i,value in enumerate(values):
            if value:axes[1].text(i,bottom[i]+value/2,str(value),ha='center',va='center',color='white',fontsize=9)
        bottom+=values
    axes[1].set(title='300 known regression cases (not new test)',ylabel='Episodes',ylim=(0,315))
    axes[1].legend(loc='center right')
    fig.suptitle('10k PPO completed; no updated checkpoint replaces the conditional baseline',fontsize=12)
    fig.savefig(report/'overview.png',dpi=160);plt.close(fig)
    lines=['# 条件恢复接入训练与 10,000 步 PPO 适配','',
        '已完成训练环境接入、固定预算 PPO、两次更新检查、300 个已见静态场景回归及产物核验。',
        '**最终保留条件恢复基线（追加步数 0）。本轮 PPO 更新没有提升验证成功率。**','',
        '| 控制/权重 | 成功 | 碰撞 | 超时 |','|---|---:|---:|---:|',
        '| 原控制规则、原权重 | 16/20 | 2 | 2 |',
        '| 条件恢复、原权重（训练起点） | 18/20 | 2 | 0 |']
    for r in rows[1:]:lines.append(f'| 条件恢复、追加 PPO {r["additional_steps"]:,} 步 | {r["success"]}/20 | {r["collision"]} | {r["timeout"]} |')
    lines.extend(['',
        '所有检查点的无障碍远/近对准均 20/20，关节越限为 0。',
        '5k/10k 连续达到预定阶段门槛（成功率至少 80%、碰撞率至多 10%、保留对准能力）。',
        '这只代表通过最低门槛：两者均比起点少成功、超时增至 3，未通过更严格的选模无退化条件；所以保留追加步数 0。',
        '没有自动进入动态训练。末次状态文件里的 17/20 是最终更新候选，不是最终保留模型。','',
        '## 回归检查',
        '三个已见的 100 场景库上，条件恢复基线和最终保留模型各自均为 94/100；合计 282/300 成功、14 碰撞、4 超时、0 越限。',
        '每组对准检查 20/20。逐回合终止原因、步数、最终误差和恢复计数一致。',
        '这次重放确认导出和接入未退化；由于保留的是同一权重，它不构成 PPO 更新改善泛化的证据。',
        '300 个场景均已被查看，不是本轮新独立测试；本轮未对被拒绝的最终候选做 300 场景评估。','',
        '## 实际采样与局限',
        f'- 采样 {training["all_transitions"]:,} 个 transition；完成 {training["complete_episodes"]} 个训练回合，任务构成 `{training["by_task"]}`，终止原因 `{training["outcomes"]}`。',
        f'- 完整回合包含 {training["steps_in_complete_episodes"]:,} 步，另有 {training["steps_in_partial_episodes"]} 步属于预算结束时未完成回合。训练随机动作的回合结果不能替代确定性验证。',
        '- 全部采样中恢复尝试 1 次、成功推进参考 0 次；停滞奖励触发 0 次。墙面惩罚累计约 -130.257。',
        '- 恢复机制已在训练环境启用，但本轮缺少成功恢复后的采样经验，不能据此声称 PPO 学会恢复。',
        '- 随机训练动作与确定性评估的状态分布不同；后续宜先检查停滞状态采样覆盖，再决定定向课程或恢复片段训练。此处未追加训练。','',
        '## 实现与复现',
        '- ConditionalTrainingEnv 组合条件恢复与既有墙面/停滞奖励，物理推进和动作投影只执行一次。评估使用 ConditionalRecoveryEnv，不加额外训练奖励。',
        '- 保留 77 维观测、7 维动作、0.30 普通恢复半径、0.20 终点半径、受限 0.45 恢复和 240 步时限。当前是一障碍静态课程，基础 YAML 的动态参数不代表本轮启用动态场景。',
        '- 四工作进程，每个 rollout 250 步；批次 100、3 epochs、学习率 1e-4、clip 0.05、target KL 0.005，固定探索标准差 0.10。54 个训练布局，10 个教师保留布局继续排除。',
        '- 从原 PPO 权重和 PPO 优化器状态继续，无 BC 更新；基线模型文件未覆盖，冻结先验参数未改变，最终候选确实产生参数更新。',
        '- 当前保留模型的 PPO 内部计数仍为 25,000；未选中的最终候选为 35,000。本轮另外采样了 10,000 步。',
        '- 集成前完整测试 115 项通过；实际触发恢复的案例验证了训练/评估观测、动作和终止结果一致，奖励差恰为训练附加项。','',
        '## 文件与动画',
        '- `selected_model.zip` 必须配合 `controller_config.json` 或 `manifest.json` 中的条件恢复规则使用；单独模型 zip 不包含环境逻辑。',
        '- `final_candidate.zip` 是未选中的 10k PPO 候选；`selection.json` 与 `promotion_assessment.json` 分别记录选模和阶段门槛。',
        '- `overview.png` / `validation_curve.csv`：验证曲线；`training_episodes.csv`：训练完整回合；`training_episode_summary.json`：采样汇总。',
        '- `summary.json`：完整汇总；`verification.json`：输入、训练代码、冻结权重与逐回合核验；两组子目录含回归明细。',
        '', '```bash',
        'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/demo_autonomous_ppo.py --seed 1000000 --playback-speed 0.6 --hold-seconds 30',
        '```','',
        '该默认入口已读取本轮导出模型与条件恢复配置，无在线 RRT 专家避障。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'training':training,'selected_additional_steps':summary['selected_additional_steps'],
        'verification':verification,'report':str(report)},indent=2))


if __name__=='__main__':main()
