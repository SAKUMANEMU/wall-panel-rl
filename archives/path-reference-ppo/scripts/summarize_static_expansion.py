"""Publish prospective one-static verification and bounded three-static expansion."""
import _bootstrap
import csv
from collections import Counter
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from evaluate_reference_handoff import tensor_fingerprint


def main():
    torch.set_num_threads(1)
    study=yaml.safe_load(Path('configs/static_expansion.yaml').read_text())
    report=Path(study['report_root']);data=Path(study['data_root'])
    if read(report/'status.json')['status']!='completed':raise RuntimeError('Experiment is not complete')
    s=read(report/'summary.json');banks=read(report/'frozen_banks.json');rows=[]
    def add(name,metrics):
        row={'group':name,**metrics['stage_target']}
        row.setdefault('planning_failure',0)
        rows.append(row)
    add('one_fresh_before',s['one_fresh'])
    if s['three_training_started']:
        add('three_validation_before',s['three_validation_before'])
        add('three_validation_selected',s['three_validation_selected'])
        add('three_holdout_before',s['three_holdout_before']);add('three_holdout_after',s['three_holdout_after'])
        add('one_regression_after',s['one_regression_after'])
    with (report/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    groups=[('Single-obstacle: 100 new / regression',
        [r for r in rows if r['group'] in ('one_fresh_before','one_regression_after')])]
    if s['three_training_started']:
        groups.extend([('Three-obstacle validation (20)',
            [r for r in rows if r['group'] in ('three_validation_before','three_validation_selected')]),
            ('Three-obstacle held-out (100)',[r for r in rows if r['group'].startswith('three_holdout')])])
    fig,axes=plt.subplots(1,len(groups),figsize=(5.7*len(groups),4.8),layout='constrained')
    colors={'success':'#2166ac','collision':'#b2182b','timeout':'#d68b00','joint_limit':'#777777','planning_failure':'#6a3d9a'}
    for ax,(title,part) in zip(np.atleast_1d(axes),groups):
        bottom=np.zeros(len(part))
        labels=['Source' if 'before' in r['group'] else
            'Source retained (step 0)' if s.get('three_selected_additional_steps')==0 else 'Selected 3-obstacle' for r in part]
        for key,color in colors.items():
            values=np.array([r[key] for r in part]);ax.bar(labels,values,bottom=bottom,label=key,color=color)
            for i,value in enumerate(values):
                if value:ax.text(i,bottom[i]+value/2,str(value),ha='center',va='center',color='white',fontsize=9)
            bottom+=values
        ax.set(title=title,ylabel='Episodes',ylim=(0,max(r['episodes'] for r in part)*1.12))
    axes=np.atleast_1d(axes)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,-.015),ncol=5,fontsize=9)
    title='Static expansion: pure PPO performance on paired scenes'
    if s.get('three_training_started') and not s['three_selected_gate']['passed']:
        title+='\nNo eligible three-obstacle checkpoint; diagnostic selection retains source weights'
    fig.suptitle(title)
    fig.savefig(report/'overview.png',dpi=160,bbox_inches='tight');plt.close(fig)

    fresh_ids=set(banks['seeds'])
    train_config=yaml.safe_load(Path(study['three_training_config']).read_text())
    train_ids={r['seed'] for r in read(train_config['train_scenes'])}
    validation_ids={c['scene']['seed'] for c in read(data/'three_validation.json')}
    checks={
        'source_file_unchanged':sha(study['initial_model'])==read(report/'protocol.json')['source_sha256'],
        'frozen_banks_unchanged':all(sha(p)==v for p,v in banks['hashes'].items()),
        'fresh_train_validation_disjoint':not bool(fresh_ids&(train_ids|validation_ids)),
        'one_three_have_identical_scene_ids':{c['scene']['seed'] for c in read(data/'one_fresh.json') if c['suite']=='stage_target'}==
            {c['scene']['seed'] for c in read(data/'three_fresh.json') if c['suite']=='stage_target'}==fresh_ids,
        'fresh_target_count':len(fresh_ids),
        'three_bank_used_for_checkpoint_selection':False,
        'default_one_model_preserved':s['default_model_preserved'],
        'automatic_dynamic_stage':False,
    }
    assert all(checks[k] for k in ('source_file_unchanged','frozen_banks_unchanged','fresh_train_validation_disjoint',
        'one_three_have_identical_scene_ids','default_one_model_preserved'))
    dump(report/'verification.json',checks)
    names={'one_fresh_before':'单障碍新100（训练前）','three_validation_before':'三障碍验证（训练前）',
        'three_validation_selected':'三障碍验证（所选模型）','three_holdout_before':'三障碍留出100（训练前模型）',
        'three_holdout_after':'三障碍留出100（所选模型）','one_regression_after':'单障碍同100回归（三障碍所选模型）'}
    lines=['# 单障碍新场景验证与三障碍静态扩展','',
        '先冻结100个新的完整场景，各自投影成单障碍和三障碍版本；同一场景的起点、目标、参考路径和中间障碍一致。',
        '保持目标精度、零噪声、77维观测、7维动作、奖励和条件恢复规则；增加静态障碍数。',
        f'场景种子从 {min(fresh_ids)} 开始，均与既有场景及训练/验证数据分开。几何生成失败 {banks["generation_failures"]} 次，单独记录；未按策略或RRT成败筛选场景。','',
        '| 场景与模型 | 成功 | 碰撞 | 超时 | 越限 | 规划失败 |','|---|---:|---:|---:|---:|---:|']
    for row in rows:lines.append(f'| {names[row["group"]]} | {row["success"]}/{row["episodes"]} | {row["collision"]} | {row["timeout"]} | {row["joint_limit"]} | {row["planning_failure"]} |')
    lines.extend(['','单障碍新100是当前固定模型的前瞻检查；三障碍100在训练和选模结束后评估，未传入训练或选模。',
        '后续用三障碍所选模型重放单障碍同100属于回归检查，不是第二批新独立测试。',''])
    if not s['three_training_started']:
        lines.extend(['## 阶段结果','', '单障碍新场景未达到既有门槛，按预定协议没有启动三障碍训练。',
            f'检查结果：`{s["one_gate"]}`。原默认模型保持不变，未启动动态阶段。'])
    else:
        three_report=Path(study['three_report_root']);training=read(three_report/'summary.json')
        initial=PPO.load(study['initial_model'],device='cpu')
        selected=PPO.load(three_report/'selected_model.zip',device='cpu')
        same=tensor_fingerprint(initial.policy)==tensor_fingerprint(selected.policy)
        if s['three_selected_additional_steps']==0:assert same
        checks['diagnostic_candidate_keeps_source_weights']=same
        dump(report/'verification.json',checks)
        logs=[json.loads(line) for path in Path(study['three_output_root']).glob('episodes_worker_*.jsonl')
              for line in path.read_text().splitlines() if line]
        tasks=[{'controller':key[0],'task':key[1],'reason':key[2],'episodes':n}
               for key,n in sorted(Counter((r['controller'],r['task'],r['reason']) for r in logs).items())]
        dump(report/'training_by_task.json',tasks)
        static_success=sum(r['controller']=='ppo' and r['task']=='static' and r['reason']=='success' for r in logs)
        alignment_success=sum(r['controller']=='ppo' and r['task']=='alignment' and r['reason']=='success' for r in logs)
        final=read(three_report/'final_evaluation/summary.json')['controllers']
        lines.extend(['## 三障碍短预算训练','',
            f'- 追加 {training["ppo_transitions"]:,} 个PPO步及 {training["rrt_transitions"]:,} 个RRT步；选择追加 {s["three_selected_additional_steps"]:,} 步的检查点。',
            '- 从原模型继续PPO权重和PPO优化器。新阶段重新建立三障碍RRT示范池，并初始化本阶段独立BC优化器；辅助学习率/权重等数值保持原配置。',
            f'- 实际开始的回合：`{training["episode_starts"]}`；完成的回合：`{training["completed_by_controller"]}`。',
            f'- BC更新接受 {training["imitation_accepted_updates"]} 次、KL超限回滚 {training["imitation_rejected_updates"]} 次，监督池 {training["imitation_pool_size"]:,} 条。',
            f'- **PPO训练回合中，三障碍静态成功 {static_success} 次，无障碍对准成功 {alignment_success} 次。对准成功不能当作三障碍成功。** 详细分类见 `training_by_task.json`。',
            f'- 三障碍纯RRT验证：`{final["rrt"]["metrics"]["stage_target"]}`；混合执行：`{final["mixture"]["metrics"]["stage_target"]}`。混合成绩包含RRT贡献。','',
            '## 扩展门槛与当前默认模型','',
            f'- 两个不同更新检查点连续达标：**{s["three_stage_gate"]["passed"]}**。',
            f'- 所选三障碍模型验证达标：**{s["three_selected_gate"]["passed"]}**；三障碍100留出检查达标：**{s["three_holdout_gate"]["passed"]}**。',
            f'- 单障碍回归的碰撞优先评分不下降：**{s["one_regression_safety_not_worse"]}**。',
            f'- 三障碍阶段整体准备就绪：**{s["three_stage_ready"]}**。',
            '- 默认演示与模型继续保留此前已验证的单障碍版本。未自动进入缓慢移动障碍物阶段。',
            '- 未达到门槛的三障碍“所选模型”仅是本轮诊断候选，不能等同于已通过部署/课程升级检查。',
            '- 选模仍采用既定最低能力及碰撞优先规则，没有依据留出100的成绩重新选模型或调整超参数。','',
            '## 三障碍演示','', '```bash',
            'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
            'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_episode_mixture.py \\',
            f'  --model {three_report.resolve()}/selected_model.zip \\',
            f'  --manifest {three_report.resolve()}/manifest.json \\',
            f'  --scene-bank {(data/"three_validation.json").resolve()} --seed 900000 --playback-speed 0.6',
            '```','', '追加 `--controller ppo` 可单独查看PPO，追加 `--controller rrt` 可单独查看专家。'])
        diagnosis=report/'slot_order_diagnosis.json'
        if diagnosis.exists():
            order=read(diagnosis)['metrics']['stage_target']
            lines.extend(['','## 固定权重的输入排列诊断','',
                f'在相同20例物理场景中，仅将障碍输入顺序从[1,2,3]改为[2,1,3]，固定源模型结果为 '
                f'{order["success"]}/20成功、{order["collision"]}碰撞、{order["timeout"]}超时。',
                '原始顺序为0/20成功、1碰撞、17超时、2越限。两种顺序均未恢复成功，且行为差异明显。',
                '该诊断提示模型对障碍槽位排列敏感，但不能单独证明失败的唯一原因；诊断顺序未用于训练、选模或新100场景。'])
    lines.extend(['','## 文件','',
        '- `summary.json` / `comparison.csv` / `overview.png`：整体结果。',
        '- `one_before.json` / `one_after.json`：单障碍前瞻检查和回归逐回合结果（若执行训练）。',
        '- `three_before_validation.json`、`three_holdout_before.json`、`three_holdout_after.json`：三障碍结果（若执行训练）。',
        '- `protocol.json`、`frozen_banks.json`、`verification.json`：预定规则、冻结数据和完整性核验。',
        f'- 三障碍训练报告：`{Path(study["three_report_root"]).resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/static_expansion_v1_seed61.log`。'])
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    for path in report.rglob('*'):
        if not path.is_file():continue
        target=visible/path.relative_to(report)
        if target.exists() and target.read_bytes()!=path.read_bytes() and path.name not in {
                'README.md','overview.png','comparison.csv','verification.json','training_by_task.json'}:
            raise RuntimeError(f'Mirror conflict: {target}')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    print(json.dumps({'report':str(visible),'results':rows,'verification':checks,
        'three_stage_ready':s.get('three_stage_ready',False)},indent=2))


if __name__=='__main__':main()
