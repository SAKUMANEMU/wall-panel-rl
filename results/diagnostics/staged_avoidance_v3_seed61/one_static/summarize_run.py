"""Rebuild the first v3 stage report from saved artifacts; no learning."""
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
os.environ.setdefault('MPLCONFIGDIR','/tmp/wall-panel-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

run=Path(__file__).resolve().parent
work=Path('/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo')
data=work/'data/staged_avoidance_v3_seed61/one_static'
models=work/'models/staged_avoidance_v3_seed61/one_static'
status=json.loads((run/'status.json').read_text())
if status['status']=='running':raise RuntimeError('Wait for the stage pipeline to stop before summarizing')
teacher=json.loads((data/'teacher/metadata.json').read_text())
bc=json.loads((models/'bc/metadata.json').read_text())
gate=json.loads((models/'bc/closed_loop_gate.json').read_text())
training_path=models/'ppo/training.json'
training=json.loads(training_path.read_text()) if training_path.exists() else {'additional_steps':0}
assessment_path=models/'ppo/assessment.json'
assessment=json.loads(assessment_path.read_text()) if assessment_path.exists() else None
records=[]
for method,suites in gate['metrics'].items():
    for suite,values in suites.items():records.append({'checkpoint':'after_bc','additional_steps':0,
        'method':method,'suite':suite,**values})
validation=models/'ppo/validation.jsonl'
if validation.exists():
    for line in validation.read_text().splitlines():
        event=json.loads(line)
        for method,suites in event['metrics'].items():
            for suite,values in suites.items():records.append({'checkpoint':event['kind'],
                'additional_steps':event['additional_steps'],'method':method,'suite':suite,**values})
with (run/'validation_summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
for source,name in [(data/'teacher/metadata.json','teacher_metadata.json'),
    (models/'bc/metadata.json','bc_metadata.json'),(models/'bc/closed_loop_gate.json','bc_gate.json'),
    (models/'bc/closed_loop_episodes.json','bc_validation_episodes.json')]:shutil.copyfile(source,run/name)
if assessment is not None:shutil.copyfile(assessment_path,run/'stage_assessment.json')
hist=bc['history'];current=gate['metrics']['dynamic_ppo']
if validation.exists():current=json.loads(validation.read_text().splitlines()[-1])['metrics']['dynamic_ppo']
suites=['stage_target','alignment_far','alignment_near']
fig,axes=plt.subplots(1,3,figsize=(15,4.4),constrained_layout=True)
updates=[v['update'] for v in hist]
for task in ('alignment','avoidance'):
    axes[0].semilogy(updates,[max(v['heldout_task_mse'][task],1e-12) for v in hist],label=task)
axes[0].semilogy(updates,[max(v['heldout_terminal_mse'],1e-12) for v in hist],label='terminal',linestyle='--')
axes[0].set(title='BC held-out errors (scene split)',xlabel='BC optimizer updates',ylabel='Action MSE')
axes[0].legend();axes[0].grid(alpha=.2)
x=np.arange(3);bottom=np.zeros(3)
for reason,color in [('success','#2a9d8f'),('collision','#e76f51'),('timeout','#e9c46a'),('joint_limit','#9b5de5')]:
    values=np.array([100*current[s][reason]/current[s]['episodes'] for s in suites])
    axes[1].bar(x,values,bottom=bottom,label=reason,color=color);bottom+=values
axes[1].set(xticks=x,xticklabels=['1 static','Far alignment','Near alignment'],ylim=(0,100),
    ylabel='Episodes (%)',title='Latest deterministic validation')
axes[1].legend(fontsize=8,loc='upper right')
for i,suite in enumerate(suites):axes[1].text(i,3,f"{current[suite]['success']}/{current[suite]['episodes']} success",ha='center',fontsize=9)
probes=gate['goal_probes']
axes[2].bar(np.arange(len(probes)),[max(v['max_abs_action'],1e-10) for v in probes],color='#457b9d')
axes[2].axhline(gate['thresholds']['maximum_goal_action'],color='#e76f51',linestyle='--',label='Gate limit')
axes[2].set(yscale='log',xticks=[0,1],xticklabels=['No obstacle','Obstacle present'],
    ylabel='Maximum absolute action',title='At-goal deterministic actions')
axes[2].legend();axes[2].grid(axis='y',alpha=.2)
fig.suptitle('V3 one-static stage | BC gate '+('PASS' if gate['passed'] else 'FAIL')+
             f" | PPO additional steps: {training['additional_steps']:,}")
fig.savefig(run/'overview.png',dpi=150);plt.close(fig)
summary={'status':status,'teacher':{k:teacher[k] for k in ('counts','attempts','transitions','unique_scene_seeds')},
    'bc_updates':bc['bc_updates'],'bc_heldout_selection_objective':bc['heldout_selection_objective'],
    'bc_gate_passed':gate['passed'],'bc_gate_checks':gate['checks'],'bc_metrics':gate['metrics']['dynamic_ppo'],
    'ppo_training':training,'stage_assessment':assessment}
(run/'summary.json').write_text(json.dumps(summary,indent=2))
elapsed=(datetime.fromisoformat(status['finished_at'])-datetime.fromisoformat(status['started_at'])).total_seconds()
rows='\n'.join(f"| {label} | {current[suite]['success']}/{current[suite]['episodes']} | {current[suite]['collision']} | {current[suite]['timeout']} | {current[suite]['joint_limit']} |"
    for suite,label in zip(suites,['单静态障碍物','远距离对准','近距离对准']))
outcome=('BC 放行未通过，按预设规则停止；没有启动 PPO，也未进入动态障碍物阶段。'
    if not gate['passed'] else 'BC 已放行，已完成当前阶段的既定 PPO 预算；没有自动进入下一阶段。')
text=f'''# v3 第一阶段结果：单静态障碍物、无噪声

{outcome}

- 开始：{status['started_at']}；结束：{status['finished_at']}；总耗时约 {elapsed/60:.2f} 分钟。
- 专家示范：{teacher['counts']['alignment']} 个对准成功回合、{teacher['counts']['avoidance']} 个避障成功回合；
  {teacher['attempts']} 次尝试，{teacher['transitions']} 条状态—动作数据，{teacher['unique_scene_seeds']} 个不同场景种子。
- BC 更新：{bc['bc_updates']} 次；检查集选模目标：{bc['heldout_selection_objective']:.8f}。
- PPO 新增环境步：{training['additional_steps']}。

| 最新验证场景 | 成功/回合 | 碰撞 | 超时 | 关节越界 |
|---|---:|---:|---:|---:|
{rows}

BC 放行：**{'通过' if gate['passed'] else '未通过'}**。
最低要求：当前阶段至少 20 例、成功率 ≥50%、碰撞率 ≤25%、无关节越界；
远/近对准各至少 10 例、成功率各 ≥90%、无碰撞/越界；目标处最大动作 ≤0.005。
最终 PPO 阶段验收仍为原来的成功率 ≥80%、碰撞率 ≤10% 等条件。
这些是重复使用的工程验证场景，不是独立测试成绩。

## 查看结果

- `overview.png`：BC 误差曲线、最新回合结果、目标动作检查。
- `summary.json` / `validation_summary.csv`：结构化结果。
- `bc_gate.json` / `bc_validation_episodes.json`：放行检查及逐回合结果。
- `teacher_metadata.json` / `bc_metadata.json`：示范统计、训练设置和数据来源。
- `status.json`：流水线结束状态；`prepare.log`、`teacher.log`、`bc.log` 为原始日志。
- 工作副本模型目录：`{models}`。
- 示范目录：`{data/'teacher'}`。

旧版本模型和结果保留。未降低安装精度（位置 <1 cm、姿态 <2°），未自动追加训练预算。
'''
(run/'README.md').write_text(text)
print(json.dumps(summary,indent=2))
