"""Summarize saved BC budget study artifacts without changing any models/data."""
import _bootstrap
import argparse
import json
from pathlib import Path
import csv
import hashlib
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


def summarize(config):
    spec=yaml.safe_load(Path(config).read_text());out=Path(spec['report_root'])
    status=json.loads((out/'status.json').read_text());root=Path(spec['output_root'])
    cases=[];losses={};metadata={}
    for name in ('same_data_10k','expanded_10k'):
        bc=root/name/'bc';hist=bc/'closed_loop_history.jsonl'
        if hist.exists():
            for line in hist.read_text().splitlines():
                event=json.loads(line);m=event['validation']['metrics']['dynamic_ppo']
                target=m['stage_target']
                cases.append({'variant':name,'update':event['update'],'passed':event['validation']['passed'],
                    **{k:target[k] for k in ('episodes','success','collision','timeout','joint_limit')},
                    'alignment_success':m['alignment_far']['success']+m['alignment_near']['success'],
                    'alignment_episodes':m['alignment_far']['episodes']+m['alignment_near']['episodes']})
        p=bc/'loss_history.json'
        if p.exists():losses[name]=json.loads(p.read_text())
        p=bc/'metadata.json'
        if p.exists():metadata[name]=json.loads(p.read_text())
    if cases:
        with (out/'checkpoint_summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
        fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
        colors={'same_data_10k':'#457b9d','expanded_10k':'#2a9d8f'}
        labels={'same_data_10k':'Fixed 100 episodes','expanded_10k':'Expanded + corrections'}
        for name,history in losses.items():
            axes[0].plot([v['update'] for v in history],[v['heldout_task_mse']['avoidance'] for v in history],
                color=colors[name],label=labels[name])
        for name in colors:
            group=[r for r in cases if r['variant']==name]
            if not group:continue
            axes[1].plot([r['update'] for r in group],[100*r['success']/r['episodes'] for r in group],
                color=colors[name],marker='o',label=labels[name]+' success')
            axes[1].plot([r['update'] for r in group],[100*r['collision']/r['episodes'] for r in group],
                color=colors[name],linestyle='--',marker='x',label=labels[name]+' collisions')
        axes[0].set(xlabel='BC updates',ylabel='Held-out avoidance action MSE',title='Supervised fit (different datasets)')
        axes[1].axhline(25,color='gray',alpha=.5,linestyle=':',label='Collision release limit')
        axes[1].set(xlabel='BC updates',ylabel='Episodes (%)',ylim=(-3,103),title='Same target validation bank (20 cases)')
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('BC budget study | '+status['status']+' | '+status['phase'])
        fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
    data=Path(spec['data_root'])/'expanded'
    progress={}
    for filename in ('train.json','validation_full.json'):
        p=data/filename
        if p.exists():progress[filename]=len(json.loads(p.read_text()))
    p=data/'teacher/attempts.json'
    if p.exists():
        attempts=json.loads(p.read_text());counts={k:sum(v['reason']=='success' and v['group']==k for v in attempts)
            for k in ('alignment','avoidance','correction')}
        progress.update(attempts=len(attempts),successful_groups=counts,successful_total=sum(counts.values()),
            failures=sum(v['reason']!='success' for v in attempts),
            unique_avoidance_scene_seeds=len({v['seed'] for v in attempts
                if v['reason']=='success' and v['group'] in ('avoidance','correction')}))
    p=data/'teacher/metadata.json'
    teacher=json.loads(p.read_text()) if p.exists() else None
    summary={'status':status,'checkpoints':cases,'selected_updates':{k:v['selected_update'] for k,v in metadata.items()},
        'expanded_progress':progress,'expanded_teacher':teacher,'variants':{}}
    for name in ('same_data_10k','expanded_10k'):
        p=out/(name+'.json')
        if p.exists():summary['variants'][name]=json.loads(p.read_text())
    if summary['variants']:
        def ranking(item):
            name,v=item;g=v['bc_gate'];t=g['metrics']['dynamic_ppo']['stage_target'];c=g['checks']
            retained=all(c[k] for k in ('alignment_samples','alignment_retained','alignment_collision_free','goal_convergence'))
            return (int(g['passed']),int(retained),t['success'],-t['collision'],-t['joint_limit'],
                    -t['timeout'],-v['metadata']['heldout_selection_objective'])
        name,best=max(summary['variants'].items(),key=ranking)
        source=root/name/'bc/model.zip';export=out/'best_bc_model.zip'
        shutil.copyfile(source,export)
        summary['best_bc']={'variant':name,'selected_update':best['metadata']['selected_update'],
            'passed_bc_gate':best['bc_gate']['passed'],'source_checkpoint':str(source.resolve()),
            'export':str(export),'sha256':hashlib.sha256(export.read_bytes()).hexdigest(),
            'metrics':best['bc_gate']['metrics']['dynamic_ppo']['stage_target'],
            'definition':'ReferencePreservingPolicy in .worktrees/path-reference-ppo; no new PPO training'}
        (out/'best_bc_model.json').write_text(json.dumps(summary['best_bc'],indent=2))
    p=out/'ppo_assessment.json'
    if p.exists():summary['ppo_assessment']=json.loads(p.read_text())
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    table='\n'.join(f"| {r['variant']} | {r['update']} | {r['success']}/{r['episodes']} | {r['collision']}/{r['episodes']} | {r['timeout']} | {r['alignment_success']}/{r['alignment_episodes']} | {'通过' if r['passed'] else '未通过'} |" for r in cases)
    verdict=('仍在运行，具体阶段见下方。' if status['status']=='running' else
        '两组 BC 均未放行，PPO 未启动。' if status['status']=='stopped_bc_gate' else
        '已完成一轮有上限的 PPO；阶段验收见 ppo_assessment.json。' if status['status']=='completed' else
        '运行异常停止，错误详情见 status.json 和日志。')
    selected_table='\n'.join(f"| {name} | {v['metadata']['selected_update']} | {v['bc_gate']['metrics']['dynamic_ppo']['stage_target']['success']}/20 | {v['bc_gate']['metrics']['dynamic_ppo']['stage_target']['collision']}/20 | {'通过' if v['bc_gate']['passed'] else '未通过'} |" for name,v in summary['variants'].items())
    data_note=(f"实际扩充数据：{teacher['attempts']} 次尝试，1000 个成功回合，{teacher['transitions']} 条专家标签，"
        f"{teacher['unique_scene_seeds']} 个布局种子（避障 {teacher['unique_avoidance_scene_seeds']} 个）。"
        if teacher is not None else '扩充数据仍在准备或采集中。')
    content=f'''# BC 预算、扩充示范与纠偏实验

当前状态：**{status['status']}**；阶段：`{status['phase']}`。{verdict}

## BC 检查点

| 数据组 | BC 更新 | 避障成功 | 碰撞 | 超时 | 对准成功 | BC 放行 |
|---|---:|---:|---:|---:|---:|---|
{table}

固定数据组复用原 100 个成功回合与同一初始化，更新上限 10000。首次 2000 次闭环结果
复现旧运行；10000 次已有 10/20 成功，但 9/20 碰撞超过 25% 上限，因此继续扩充。
扩充组计划 256 个布局、1000 个成功回合：200 对准、600 普通避障、200 纠偏。
纠偏前缀只用于产生真实偏离状态，不作为动作标签；离线专家负责后续标签，执行策略无专家。
扩充组更新上限同为 10000，至少覆盖 200 个不同避障布局；数据进度见 summary.json。
{data_note}

## 实际保留的模型

| 数据组 | 选中 BC 更新数 | 避障成功 | 碰撞 | 放行 |
|---|---:|---:|---:|---|
{selected_table}

`best_bc_model.zip` / `best_bc_model.json` 导出两组中闭环评分最高的 BC 模型和指标。
这是便于查看的本地快照，不表示已通过放行，也不是新 PPO 模型；定义代码位于工作副本。
扩充组 5000→10000 次更新期间，监督误差下降但成功率从 2/20 降为 0/20，因此保留 5000 次模型。
固定数据组证明增加更新次数有益；扩充与纠偏的组合在本预算内未提高闭环成绩，不能据此
断言单独增加数据无效。两组数据量和平均状态采样次数不同，后续需分别考察。

选模优先通过放行与保留对准，再考虑成功/碰撞/越界/超时，最后比较监督误差。
训练数据不同的两组 MSE 不能当作同一检查集的直接对比；图中闭环成绩来自相同种子范围
的 20 个目标验证场景。重复验证用于工程选模，不是独立测试证明。

## 文件

- `overview.png`：BC 检查集误差和闭环成功/碰撞曲线。
- `checkpoint_summary.csv`：检查点结果；`summary.json`：状态、覆盖与选模结果。
- `same_data_10k.json` / `expanded_10k.json`：完成阶段的配置、选中模型和 BC 放行结果。
- `manifest.json`：源模型、原数据绑定、物理配置、代码哈希；`status.json`：流程状态。
- 原始日志：`/home/johnsonchen/wall-panel-rl/logs/bc_budget_v3_seed61.log`。
- 模型根目录：`{root.resolve()}`；扩充数据：`{data.resolve()}`。

任一组放行后最多 10000 个 PPO 环境步；两组均失败则停止。每回合上限 240 步、
安装成功位置 <1cm / 姿态 <2° 和阶段推进阈值均未放宽，旧结果保留。
'''
    (out/'README.md').write_text(content)
    print(json.dumps({'status':status,'checkpoints':cases,'expanded_progress':progress},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/bc_budget_study.yaml')
    summarize(parser.parse_args().config)
