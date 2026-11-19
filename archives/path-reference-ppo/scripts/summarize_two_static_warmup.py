"""Report strong two-static BC, optional gated PPO, and untouched test results."""
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
from wall_panel_rl.rl.episode_policy import safety_first_score
from wall_panel_rl.rl.avoidance_curriculum import gate_result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/two_static_warmup.yaml')
    args=parser.parse_args();torch.set_num_threads(1);study=yaml.safe_load(Path(args.config).read_text())
    report=Path(study['report_root']);models=Path(study['model_root']);data=Path(study['data_root'])
    if read(report/'status.json')['status']!='completed':raise RuntimeError('Study is not complete')
    summary=read(report/'summary.json');warm=read(models/'bc/metadata.json')
    teacher_root=Path(study.get('teacher_root',data/'teacher'));teacher=read(teacher_root/'metadata.json')
    source=PPO.load(study['initial_model'],device='cpu');candidate=PPO.load(report/'candidate_model.zip',device='cpu')
    frozen=[n for n,p in source.policy.named_parameters() if not p.requires_grad]
    source_state=source.policy.state_dict();candidate_state=candidate.policy.state_dict()
    held=set(read(study['heldout_metadata'])['heldout_scene_seeds'])
    snapshots=read(report/'code_snapshot.json')['code_hashes']
    checks={'source_file_unchanged':sha(study['initial_model'])==read(report/'protocol.json')['source_sha256'],
        'executed_code_unchanged':all(sha(p)==h for p,h in snapshots.items()),
        'data_banks_unchanged':all(sha(p)==h for p,h in read(report/'data_hashes.json').items()),
        'teacher_unchanged':sha(teacher_root/'teacher.npz')==teacher['teacher_sha256'],
        'teacher_heldout_not_in_gradients':not bool(set(warm['train_scene_seeds'])&held),
        'teacher_replay_passed':all(r['reason']=='success' and r['max_error']<=1e-6 for r in read(report/'teacher_replay.json')),
        'bc_preserved_prior_critic_std':warm['prior_critic_and_std_unchanged'],
        'final_preserved_frozen_prior':bool(frozen) and all(torch.equal(source_state[n],candidate_state[n]) for n in frozen),
        'candidate_sha_matches':sha(report/'candidate_model.zip')==summary['candidate_sha256'],
        'default_model_preserved':summary['default_model_preserved'],
        'fresh_test_used_for_selection':False}
    fresh_ids={c['scene']['seed'] for c in read(data/'two_test.json') if c['suite']=='stage_target'}
    validation_ids={c['scene']['seed'] for c in read(data/'two_validation.json')}
    checks['fresh_test_disjoint_from_teacher_and_validation']=not bool(fresh_ids&(set(warm['train_scene_seeds'])|held|validation_ids))
    if summary.get('original_actor_preserved'):
        from wall_panel_rl.rl.added_slot_adapter import original_actor_preserved
        checks['entire_original_one_actor_preserved']=original_actor_preserved(source.policy,candidate.policy)
        if summary['ppo_started']:
            raw=PPO.load(report/'ppo_run/final_candidate.zip',device='cpu')
            checks['raw_final_ppo_one_actor_preserved']=original_actor_preserved(source.policy,raw.policy)
    required=[k for k in checks if k!='fresh_test_used_for_selection'];assert all(checks[k] for k in required),checks
    dump(report/'verification.json',checks)
    rows=[]
    for name,metrics in [('two_before',summary['baseline']['two']),('one_before_validation',summary['baseline']['one'])]:
        rows.append({'phase':name,'updates':0,**metrics['stage_target']})
    for event in warm['history']:
        for task in ('two','one'):
            rows.append({'phase':'bc_'+task,'updates':event['update'],**event[task]['stage_target']})
    ppo_history=[]
    if summary['ppo_started']:
        root=models/'ppo_run/ppo'
        ppo_history=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines() if line]
        for event in ppo_history:
            for task,m in [('two',event['metrics']['dynamic_ppo']),('one',event['gate']['one_static_metrics'])]:
                rows.append({'phase':'ppo_'+task,'updates':event['additional_steps'],**m['stage_target']})
        selected_step=read(report/'ppo_run/summary.json')['selection']['additional_steps']
        mixture=read(root/'mixed_evaluations'/f'episodes_{selected_step}.json')
        dump(report/'selected_mixture_validation.json',mixture)
    for name in ('two_holdout_before','two_holdout_after','one_before','one_after'):
        rows.append({'phase':name,'updates':None,**summary[name]['stage_target']})
    if 'two_initialized_holdout' in summary:
        rows.append({'phase':'two_after_initialization_before_learning','updates':0,**summary['two_initialized_holdout']['stage_target']})
    keys=['phase','updates','episodes','success','collision','timeout','joint_limit','planning_failure']
    with (report/'comparison.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        writer.writerows(dict(r,planning_failure=r.get('planning_failure',0)) for r in rows)
    parent=read(Path(study['source_report'])/'manifest.json');gate=parent['evaluation_settings']['gate']
    def rank(metrics):
        return safety_first_score({'metrics':{'dynamic_ppo':metrics},'checks':{'alignment_collision_free':all(
            metrics[k]['collision']==0 and metrics[k]['joint_limit']==0 for k in ('alignment_far','alignment_near')),
            'goal_convergence':summary['final_validation']['checks']['goals']}},gate)
    regression=rank(summary['one_after'])>=rank(summary['one_before'])
    consecutive=(len(ppo_history)>=3 and all(gate_result(e['metrics']['dynamic_ppo'],gate)['passed'] and
        e['gate']['checks']['one_static_retained'] for e in ppo_history[-2:])) if summary['ppo_started'] else False
    ready=summary['final_validation']['eligible'] and summary['heldout_gate']['passed'] and regression
    assessment={'candidate_validation_eligible':summary['final_validation']['eligible'],
        'two_holdout_passed':summary['heldout_gate']['passed'],'one_regression_safety_not_worse':regression,
        'two_updated_ppo_checks_passed':consecutive,'two_task_ready':ready,
        'dynamic_progression_ready':ready and consecutive,'default_model_preserved':True}
    dump(report/'assessment.json',assessment)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'success':'#2166ac','collision':'#b2182b','timeout':'#d68b00'}
    for key,color in colors.items():
        axes[0,0].plot([e['update'] for e in warm['history']],
            [100*e['two']['stage_target'][key]/20 for e in warm['history']],marker='o',color=color,label=key)
    axes[0,0].set(title='BC warm-up: two-static validation (20)',xlabel='BC updates',ylabel='Rate (%)',ylim=(-3,105));axes[0,0].legend()
    if ppo_history:
        for key,color in colors.items():
            axes[0,1].plot([e['additional_steps'] for e in ppo_history],
                [100*e['metrics']['dynamic_ppo']['stage_target'][key]/20 for e in ppo_history],marker='o',color=color,label=key)
        axes[0,1].legend()
    else:axes[0,1].text(.5,.5,'PPO not started: BC start gate not met',ha='center',va='center',transform=axes[0,1].transAxes)
    axes[0,1].set(title='PPO adaptation: two-static validation (20)',xlabel='Additional PPO transitions',ylabel='Rate (%)',ylim=(-3,105))
    for ax,names,title in [(axes[1,0],('two_holdout_before','two_holdout_after'),'Two-static frozen test (100)'),
                           (axes[1,1],('one_before','one_after'),'One-static known regression (100)')]:
        labels=['Source','Candidate']
        if 'two_initialized_holdout' in summary and names[0]=='two_holdout_before':
            names=('two_holdout_before','two_initialized_holdout','two_holdout_after');labels=['Source','Init only','Selected']
        bottom=np.zeros(len(names))
        for key,color in [*colors.items(),('joint_limit','#777777')]:
            values=np.array([summary[n]['stage_target'][key] for n in names]);ax.bar(labels,values,bottom=bottom,color=color,label=key)
            for i,value in enumerate(values):
                if value:ax.text(i,bottom[i]+value/2,str(value),ha='center',va='center',color='white',fontsize=9)
            bottom+=values
        ax.set(title=title,ylabel='Episodes',ylim=(0,112))
    title='Two-static bridge: stronger RRT supervision, gated PPO, separate hold-out checks'
    if summary.get('original_actor_preserved'):
        title='Protected two-static: initialization and learning evaluated separately'
    fig.suptitle(title)
    fig.savefig(report/'overview.png',dpi=160);plt.close(fig)
    ppo_selected=None
    if summary['ppo_started']:
        p=read(report/'ppo_run/summary.json');ppo_selected=p['selection']['additional_steps']
        aux_path=models/'ppo_run/auxiliary_bc'/(f'aux_{ppo_selected}.pt' if ppo_selected<study['ppo_steps'] else 'final_aux_state.pt')
    else:aux_path=models/'bc/selected_bc_state.pt'
    shutil.copy2(aux_path,report/'candidate_aux_state.pt')
    selected_bc=warm['selected'];final=summary['final_validation']
    lines=['# 双静态障碍：强化BC预热与PPO适配','',
        f'源模型保持不变；新增双障碍场景使用原始障碍索引[2,3]，原中间障碍仍在第一个槽位。测试场景从种子 {study["fresh_seed_start"]} 起预先固定。','',
        '## RRT示范与BC预热',
        f'- {"复用已验证的" if "teacher_root" in study else "采集"} {teacher["episodes"]} 条成功轨迹，分组 `{teacher["group_counts"]}`，总计 {teacher["transitions"]:,} 个监督状态。',
        '- 对准轨迹使用冻结参考伺服；单/双障碍轨迹使用现有RRT专家。失败尝试保留在teacher/attempts.json。',
        f'- BC训练布局 {len(warm["train_scene_seeds"])} 个、留出布局 {len(warm["heldout_scene_seeds"])} 个；训练批次权重为双障60%、单障30%、对准10%。',
        '- 交换两个活动障碍的槽位作数据增强，提升相对冻结参考动作偏离较大的状态的监督权重。只更新残差与门控，先验、价值网络和标准差不变。',
        f'- 执行 {study["bc"]["updates"]:,} 次BC更新，选择更新 {selected_bc["update"]:,} 的模型；PPO启动资格：**{selected_bc["ppo_ready"]}**。',
        '', '| BC更新 | 双障成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |','|---:|---:|---:|---:|---:|---:|']
    for event in warm['history']:
        a=event['two']['stage_target'];b=event['one']['stage_target']
        lines.append(f'| {event["update"]:,} | {a["success"]}/20 | {a["collision"]} | {a["timeout"]} | {b["success"]}/20 | {b["collision"]} |')
    lines.extend(['','## PPO适配',f'- 是否启动：**{summary["ppo_started"]}**。',
        '- 启动要求：双障成功≥50%、碰撞≤25%、无越限，同时单障成功≥80%且验证中无碰撞，并保留对准与目标收敛。最终放行要求更严格。'])
    if ppo_history:
        lines.extend([f'- 执行 {p["ppo_transitions"]:,} 个PPO步、额外 {p["rrt_transitions"]:,} 个RRT步，选择追加 {ppo_selected:,} 步。',
            '- BC预热后PPO使用新的Adam和5e-5学习率；每回合50%RRT/50%PPO。辅助BC用训练示范初始化监督池，每rollout更新3批，权重.2。',
            '', '| PPO追加步 | 双障成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |','|---:|---:|---:|---:|---:|---:|'])
        for event in ppo_history:
            a=event['metrics']['dynamic_ppo']['stage_target'];b=event['gate']['one_static_metrics']['stage_target']
            lines.append(f'| {event["additional_steps"]:,} | {a["success"]}/20 | {a["collision"]} | {a["timeout"]} | {b["success"]}/20 | {b["collision"]} |')
        lines.append(f'\n所选模型的50/50混合执行验证：`{mixture["metrics"]["stage_target"]}`；该成绩包含在线RRT贡献。')
    else:lines.append('- 预热未达到启动门槛，按预定协议没有执行PPO；不把无效预热模型作为新的默认策略。')
    lines.extend(['','## 冻结测试和回归','', '| 数据 | 模型 | 成功 | 碰撞 | 超时 | 越限 |','|---|---|---:|---:|---:|---:|'])
    for title,names in [('双障新100',('two_holdout_before','two_holdout_after')),('单障旧100回归',('one_before','one_after'))]:
        labels=('原模型','候选模型')
        if 'two_initialized_holdout' in summary and title=='双障新100':
            names=('two_holdout_before','two_initialized_holdout','two_holdout_after');labels=('原模型','初始化修正、未学习','所选候选')
        for label,name in zip(labels,names):
            a=summary[name]['stage_target'];lines.append(f'| {title} | {label} | {a["success"]}/100 | {a["collision"]} | {a["timeout"]} | {a["joint_limit"]} |')
    lines.extend(['',f'资格检查：`{assessment}`。',
        '双障新100未参与训练或选模；单障100是既有回归库。混合执行成绩不能替代纯PPO成绩。',
        '当前默认单障模型始终保留；未启动三障或动态阶段。候选是否可用以assessment.json为准。','',
        '## 文件',
        '- comparison.csv / overview.png：阶段成绩和曲线；verification.json：数据、代码及冻结参数核验。',
        '- candidate_model.zip / candidate_manifest.json / candidate_aux_state.pt：候选与匹配配置、辅助优化器状态。',
        '- baseline.json、bc_validation/、ppo_run/：各检查点；two_test_before/after.json、one_before/after.json：逐场景结果。',
        '- teacher_replay.json：三组示范的动作/观测物理重放检查；data_hashes.json记录冻结测试及验证场景哈希。',
        f'- 原始示范：`{teacher_root.resolve()}`；BC/PPO模型与优化器：`{models.resolve()}`。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/two_static_warmup_v1_seed61.log`。'])
    if summary.get('original_actor_preserved'):
        init=summary['initialization_validation']['two']['stage_target'];test_init=summary['two_initialized_holdout']['stage_target']
        lines[1:1]=['','**本报告为初始化修正及原单障策略保护后的独立试验。第一轮失败记录完整保留。**',
            f'仅将原来未激活槽位的输入列归零，未训练时双障验证变为 {init["success"]}/20，碰撞 {init["collision"]}；单障保持原行为。',
            f'初始化修正模型在新100场景中为 {test_init["success"]}/100成功、{test_init["collision"]}碰撞、{test_init["timeout"]}超时。',
            '动作网络只允许新增槽位输入列更新，所有原单障动作连接保持不变；PPO价值网络可单独更新。',
            '最终测试同时列出原模型、初始化修正模型和学习后模型，区分初始化与训练的贡献。']
        if selected_bc['update']==0 and ppo_selected==0:
            lines[2:2]=['','**本轮未采纳任何BC/PPO更新。最终候选仍是仅做初始化修正的模型；新增槽位输入权重仍为零，尚未学会利用第二障碍信息。这些测试成绩不能归因于本轮训练。**']
        lines=[line.replace('/home/johnsonchen/wall-panel-rl/logs/two_static_warmup_v1_seed61.log',
            '/home/johnsonchen/wall-panel-rl/logs/protected_two_static_v2_seed61.log') for line in lines]
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    for path in report.rglob('*'):
        if not path.is_file():continue
        target=visible/path.relative_to(report)
        if target.exists() and target.read_bytes()!=path.read_bytes() and path.name not in {
                'README.md','overview.png','comparison.csv','verification.json','assessment.json'}:
            raise RuntimeError(f'Mirror conflict: {target}')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    print(json.dumps({'report':str(visible),'assessment':assessment,'verification':checks,
        'selected_bc_update':selected_bc['update'],'selected_ppo_steps':ppo_selected,
        'two_test':summary['two_holdout_after']['stage_target'],'one_regression':summary['one_after']['stage_target']},indent=2))


if __name__=='__main__':main()
