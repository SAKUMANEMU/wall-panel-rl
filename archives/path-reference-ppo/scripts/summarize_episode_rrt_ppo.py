"""Publish completed episode-level imitation/PPO training and validation."""
import _bootstrap
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/episode_rrt_ppo.yaml');args=parser.parse_args()
    torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text())
    root=Path(study['output_root']);report=Path(study['report_root']);ppo=root/'ppo'
    if read(report/'status.json')['status']!='completed':raise RuntimeError('Training/evaluation is not complete')
    summary=read(report/'summary.json');manifest=read(report/'manifest.json')
    eligible=bool(summary['selection']['selection_score'][0])
    history=json_lines(ppo/'validation.jsonl');updates=json_lines(root/'auxiliary_bc/updates.jsonl')
    records=[]
    for path in sorted(root.glob('episodes_worker_*.jsonl')):
        records.extend(dict(row,worker=path.stem) for row in json_lines(path))
    table=[]
    for event in history:
        added=event['additional_steps'];mixed=read(ppo/'mixed_evaluations'/f'episodes_{added}.json')
        for name,metrics in [('ppo',event['metrics']['dynamic_ppo']),('mixture',mixed['metrics'])]:
            table.append({'additional_steps':added,'controller':name,**metrics['stage_target']})
    # The pure PPO rows lack planning_failure, which is always zero there.
    fields=['additional_steps','controller','episodes','success','collision','timeout','joint_limit','planning_failure']
    with (report/'validation_curve.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows(dict(row,planning_failure=row.get('planning_failure',0)) for row in table)
    if records:
        keys=sorted(set().union(*(row.keys() for row in records)))
        with (report/'training_episodes.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)

    initial=PPO.load(study['initial_model'],device='cpu')
    selected=PPO.load(report/'selected_model.zip',device='cpu')
    final=PPO.load(report/'final_candidate.zip',device='cpu')
    selected_step=summary['selection']['additional_steps']
    auxiliary_path=root/'auxiliary_bc'/(f'aux_{selected_step}.pt' if selected_step<study['ppo_steps'] else 'final_aux_state.pt')
    auxiliary_state=torch.load(auxiliary_path,map_location='cpu',weights_only=False)
    shutil.copy2(auxiliary_path,report/'selected_aux_state.pt')
    frozen=[name for name,param in initial.policy.named_parameters() if not param.requires_grad]
    a=initial.policy.state_dict();b=final.policy.state_dict()
    sampling=read(ppo/'sampling_events.json')
    rrt_rows=[r for r in records if r['controller']=='rrt'];ppo_rows=[r for r in records if r['controller']=='ppo']
    accepted_labels=sum(r['steps'] for r in rrt_rows if r['accepted_for_imitation'])
    verification={
        'source_file_unchanged':sha(study['initial_model'])==manifest['source_sha256'],
        'source_code_unchanged':all(sha(p)==v for p,v in manifest['code_hashes'].items()),
        'export_hash_matches':sha(report/'selected_model.zip')==summary['model_sha256']==manifest['model_sha256'],
        'exact_ppo_budget':final.num_timesteps-initial.num_timesteps==study['ppo_steps']==sampling['observed_transitions'],
        'frozen_prior_parameters_unchanged':bool(frozen) and all(torch.equal(a[n],b[n]) for n in frozen),
        'rrt_step_accounting':sum(r['steps'] for r in rrt_rows)==summary['rrt_transitions'],
        'pool_contains_successful_teacher_labels_only':min(accepted_labels,study['auxiliary_bc']['capacity'])==summary['imitation_pool_size'],
        'auxiliary_update_accounting':sum(u['accepted'] for u in updates)==summary['imitation_accepted_updates'] and
            sum(not u['accepted'] for u in updates)==summary['imitation_rejected_updates'],
        'distinct_checkpoints':[r['additional_steps'] for r in history]==[0,5000,10000],
        'selected_auxiliary_state_matches_model':auxiliary_state['ppo_num_timesteps']==selected.num_timesteps,
        'selected_checkpoint_bc_updates':auxiliary_state['accepted_updates'],
        'selected_auxiliary_sha256':sha(report/'selected_aux_state.pt'),
        'stored_ppo_counters':{'initial':initial.num_timesteps,'selected':selected.num_timesteps,'final':final.num_timesteps},
        'successful_rrt_label_transitions':accepted_labels,
        'ppo_steps_in_incomplete_episodes':summary['ppo_transitions']-sum(r['steps'] for r in ppo_rows),
        'ppo_recovery_events':sampling,
    }
    assert all(v for k,v in verification.items() if isinstance(v,bool)),verification
    dump(report/'verification.json',verification)
    outcome={name:dict(Counter(r['reason'] for r in records if r['controller']==name)) for name in ('rrt','ppo')}

    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'success':'#2166ac','collision':'#b2182b','timeout':'#d68b00','joint_limit':'#777777','planning_failure':'#6a3d9a'}
    for ax,name,title in [(axes[0,0],'ppo','Pure PPO validation (20 targets)'),
                          (axes[0,1],'mixture','Episode mixture validation (20 targets)')]:
        rows=[r for r in table if r['controller']==name]
        for key in ('success','collision','timeout'):
            ax.plot([r['additional_steps'] for r in rows],[100*r[key]/r['episodes'] for r in rows],
                    marker='o',color=colors[key],label=key)
        ax.set(title=title,xlabel='Additional PPO transitions',ylabel='Rate (%)',ylim=(-3,105),xticks=[0,5000,10000])
        ax.grid(alpha=.2);ax.legend(loc='center right')
    bottom=np.zeros(2)
    for key,color in colors.items():
        values=np.array([outcome[name].get(key,0) for name in ('rrt','ppo')])
        axes[1,0].bar(['RRT','PPO'],values,bottom=bottom,color=color,label=key)
        for i,value in enumerate(values):
            if value>=3:axes[1,0].text(i,bottom[i]+value/2,str(value),ha='center',va='center',color='white')
        bottom+=values
    axes[1,0].set(title='Completed training episodes (all tasks)',ylabel='Episodes');axes[1,0].legend(fontsize=8)
    if updates:
        axes[1,1].plot([u['ppo_additional_steps'] for u in updates],[u['loss'] for u in updates],marker='o',color='#238b45')
    axes[1,1].set(title='Separate auxiliary BC updates',xlabel='Additional PPO transitions',ylabel='Weighted imitation MSE')
    axes[1,1].grid(alpha=.2)
    fig.suptitle('RRT/PPO episode mixture: known validation and training diagnostics')
    fig.savefig(report/'overview.png',dpi=160);plt.close(fig)

    final_evaluation=read(report/'final_evaluation/summary.json')['controllers']
    chosen=final_evaluation['ppo']['metrics']['stage_target']
    mixed_chosen=final_evaluation['mixture']['metrics']['stage_target']
    lines=['# 每回合 RRT/PPO 50% 混合训练结果','',
        f'已完成 **{summary["ppo_transitions"]:,} 个 PPO 采样步**，另执行 **{summary["rrt_transitions"]:,} 个 RRT 步**；合计 {summary["total_physics_transitions"]:,} 个控制步。',
        f'当前课程为 {manifest["evaluation_settings"]["active_stage"]["count"]} 个静态障碍物、零传感器噪声。RRT/PPO每回合独立抽签，整回合保持所选控制器。','',
        '| 追加PPO步数 | 控制方式 | 成功 | 碰撞 | 超时 |','|---:|---|---:|---:|---:|']
    for row in table:lines.append(f'| {row["additional_steps"]:,} | {row["controller"]} | {row["success"]}/{row["episodes"]} | {row["collision"]} | {row["timeout"]} |')
    lines.extend(['',f'**最终选择追加 {selected_step:,} 步检查点**。纯PPO验证：{chosen["success"]}/20成功、{chosen["collision"]}碰撞、{chosen["timeout"]}超时。',
        f'同权重的混合执行验证：{mixed_chosen["success"]}/20成功、{mixed_chosen["collision"]}碰撞、{mixed_chosen["timeout"]}超时。',
        '选择标准先保留最低成功率、对准与终点收敛能力，再优先减少碰撞；混合执行成绩不参与纯PPO选模。',
        '上述是已见验证场景，混合成功包含在线RRT的贡献，不是纯PPO能力或新独立测试。',
        '纯RRT及两组远近对准的结果见 `final_evaluation/summary.json`。','',
        '## 训练样本与更新',
        f'- 实际开始的回合：`{summary["episode_starts"]}`；完成的回合及终止原因：`{outcome}`。',
        '- 抽签概率50%不等于有限样本回合数各半，也不等于两种控制器的执行步数各半。规划失败保持RRT标签并计入分母。',
        f'- 成功RRT回合提供 {accepted_labels:,} 个监督标签；最终监督池 {summary["imitation_pool_size"]:,} 条。',
        f'- 独立BC更新接受 {summary["imitation_accepted_updates"]} 次、KL超限回滚 {summary["imitation_rejected_updates"]} 次。BC仅更新残差/门控网络，PPO缓冲区不含RRT动作。',
        f'- 所选检查点实际已经历 {auxiliary_state["accepted_updates"]} 次BC更新；对应的优化器、采样RNG和监督池另存为 `selected_aux_state.pt`，其步数已与所选PPO模型核对。',
        f'- PPO完整回合以外还包含 {verification["ppo_steps_in_incomplete_episodes"]} 个未完成回合步；训练回合结果含探索随机性，不能替代确定性验证。',
        f'- PPO采样中的恢复/奖励事件：`{sampling}`。',
        '- 原模型文件、冻结先验参数、运行时代码、步数、监督标签来源和导出哈希均已核验。','',
        '## 文件',
        '- `validation_curve.csv`：各检查点纯PPO及混合执行结果；`overview.png`：曲线与采样诊断。',
        '- `summary.json`、`training_episodes.csv`：训练汇总与完整回合；`verification.json`：完整性核验。',
        '- `selected_model.zip` 配合 `manifest.json` / `controller_config.json` 使用；`final_candidate.zip` 为预算结束时模型。',
        '- `final_evaluation/`：所选模型的纯PPO、纯RRT和混合执行逐回合结果。',
        '- 本轮尚无新的独立测试或匹配的纯PPO训练对照，不能把已见验证变化单独归因于BC/RRT。',
        f'- 原始训练、BC优化器/监督池快照与检查点：`{root.resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/episode_rrt_ppo_v1_seed61.log`。','',
        '## 终端演示所选模型','', '```bash',
        'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_episode_mixture.py \\',
        f'  --model {report.resolve()}/selected_model.zip \\',
        f'  --manifest {report.resolve()}/manifest.json --playback-speed 0.6',
        '```','', '追加 `--controller ppo` 可只查看PPO；不指定时每次演示随机选择一个控制器。'])
    if not eligible:
        lines[1:1]=['','**本轮没有通过最低能力门槛的模型。所选文件仅是诊断候选，不用于替换当前默认模型。**']
    dump(report/'candidate_eligibility.json',{'eligible':eligible,
        'all_checkpoints_ineligible':all(not e['selection_score'][0] for e in history),
        'diagnostic_only':not eligible,'thresholds':manifest['training_settings']['gate']})
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():
        for path in report.rglob('*'):
            if not path.is_file():continue
            target=visible/path.relative_to(report)
            if target.exists() and target.read_bytes()!=path.read_bytes():raise RuntimeError(f'Mirror conflict: {target}')
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    print(json.dumps({'report':str(visible),'selected_additional_steps':selected_step,
        'pure_ppo':chosen,'mixture':mixed_chosen,'training':summary,'verification':verification},indent=2))


if __name__=='__main__':main()
