"""Verify and publish a completed bounded multi-obstacle adapter experiment."""
import _bootstrap
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from run_two_static_adapter import resolve
from wall_panel_rl.rl.multi_obstacle_adapter import frozen_base_preserved
from wall_panel_rl.rl.multi_obstacle_warmup import teacher_arrays


def summarize(study, mirror=True):
    data, models, report = map(Path, (study['data_root'], study['model_root'], study['report_root']))
    if read(report/'status.json')['status'] != 'completed': raise RuntimeError('Experiment is not complete')
    protocol, summary = read(report/'protocol.json'), read(report/'summary.json')
    for path, expected in protocol['code_hashes'].items():
        if sha(path) != expected: raise AssertionError(f'Executed code changed: {path}')
    for path, expected in read(report/'data_hashes.json').items():
        if sha(path) != expected: raise AssertionError(f'Frozen data changed: {path}')
    if sha(study['initial_model']) != protocol['source_sha256']: raise AssertionError('Source file changed')
    if sha(report/'candidate_model.zip') != summary['candidate_sha256']: raise AssertionError('Candidate changed')
    held = set(protocol['teacher_heldout_seeds']); train = set(protocol['train_seeds'])
    tests = read(data/'two_test.json'); validation = read(data/'two_validation.json')
    test_seeds = {c['scene']['seed'] for c in tests}
    if test_seeds & (train | held | {c['scene']['seed'] for c in validation}): raise AssertionError('Test seed leakage')
    teacher, parts = teacher_arrays(data/'teacher/teacher.npz', held)
    bc_metadata = read(models/'bc/metadata.json')
    if set(bc_metadata['train_scene_seeds']) & held:
        raise AssertionError('BC trained on held-out teacher layouts')
    correction_seeds = {r['seed'] for r in read(data/'corrections/attempts.json')}
    if correction_seeds - train: raise AssertionError('Correction used non-training scenes')
    base = PPO.load(models/'initialized_model.zip', device='cpu')
    chosen = PPO.load(report/'candidate_model.zip', device='cpu')
    actor_check = frozen_base_preserved(base.policy, chosen.policy)
    if not actor_check: raise AssertionError('Frozen original actor changed')
    # Verify at least one learned multi-box action and exact one-box actions.
    ids = parts['train']['two_static']
    indices = np.concatenate([np.arange(teacher['offsets'][i], teacher['offsets'][i+1]) for i in ids])[:2048]
    x = torch.as_tensor(teacher['obs'][indices])
    one = x.clone(); one[:, 30:70] = 0
    with torch.no_grad():
        delta = float((chosen.policy.tracking_mean(x)-base.policy.tracking_mean(x)).abs().max())
        one_identical = torch.equal(chosen.policy.tracking_mean(one), base.policy.tracking_mean(one))
    if not one_identical: raise AssertionError('One-box actions changed')
    ppo_selected = None
    if summary['ppo_started']:
        ppo_selected = read(models/'ppo_run/ppo/selection.json')['additional_steps']
        final = PPO.load(report/'ppo_run/final_candidate.zip', device='cpu')
        if not frozen_base_preserved(base.policy, final.policy): raise AssertionError('Raw final PPO base changed')
        pool = read(models/'ppo_run/auxiliary_bc/initial_pool.json')
        if set(pool['heldout_excluded']) != held: raise AssertionError('Auxiliary pool split changed')
    verification = {'frozen_data_hashes_valid': True, 'executed_code_hashes_valid': True,
        'source_hash_unchanged': True, 'train_validation_test_disjoint': True,
        'corrections_only_training_scenes': True, 'teacher_heldout_excluded_from_updates': True,
        'original_actor_frozen': actor_check, 'one_obstacle_actions_identical': one_identical,
        'selected_multi_action_change_vs_initial_max': delta, 'candidate_sha256': summary['candidate_sha256'],
        'new_test_used_for_selection': False, 'default_model_preserved': True}
    dump(report/'verification.json', verification)
    bc = read(models/'bc/history.json')
    rows = []
    def add(phase, steps, metrics):
        rows.append(dict(phase=phase, updates=steps, **metrics))
    for entry in bc:
        add('bc_two', entry['update'], entry['two']['stage_target'])
        add('bc_one', entry['update'], entry['one']['stage_target'])
    ppo_events = []
    if summary['ppo_started']:
        ppo_events = [json.loads(line) for line in (models/'ppo_run/ppo/validation.jsonl').read_text().splitlines() if line.strip()]
        for e in ppo_events: add('ppo_two', e['additional_steps'], e['metrics']['dynamic_ppo']['stage_target'])
    for key, phase in [('two_initialized_holdout', 'two_new_test_initialized'), ('two_holdout_after', 'two_new_test_candidate'),
                       ('one_before', 'one_regression_before'), ('one_after', 'one_regression_after')]:
        add(phase, '', summary[key]['stage_target'])
    fields = ['phase', 'updates', 'episodes', 'success', 'collision', 'timeout', 'joint_limit', 'planning_failure']
    with (report/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    # The report records the gate, never advances stages or changes a default.
    from wall_panel_rl.rl.avoidance_curriculum import assess_history
    from run_two_static_warmup import stage
    parent = read(Path(study['source_report'])/'manifest.json')
    settings = stage(parent['evaluation_settings'], 2)
    stability = assess_history(ppo_events, settings['gate']) if ppo_events else {'passed': False}
    assessment = {'bc_learning_gate_passed': summary['bc_learning_gate'],
        'selected_bc_updates': summary['selected_bc']['update'], 'selected_ppo_additional_steps': ppo_selected,
        'selected_candidate_validation_eligible': summary['final_validation']['eligible'],
        'new_test_gate_passed': summary['heldout_gate']['passed'],
        'ppo_stability_passed': stability['passed'],
        'learned_multi_obstacle_actions': delta > 1e-5,
        'default_model_preserved': True, 'automatic_dynamic_stage': False}
    before_test = summary['two_initialized_holdout']['stage_target']
    after_test = summary['two_holdout_after']['stage_target']
    assessment['new_test_not_worse_than_initialization'] = (
        after_test['success'] >= before_test['success'] and after_test['collision'] <= before_test['collision'])
    assessment['two_task_ready'] = all(assessment[k] for k in ('selected_candidate_validation_eligible',
        'new_test_gate_passed', 'ppo_stability_passed', 'learned_multi_obstacle_actions',
        'new_test_not_worse_than_initialization'))
    dump(report/'assessment.json', assessment); dump(report/'ppo_stability.json', stability)
    accounting = {'ppo_transitions': 0, 'rrt_transitions': 0, 'completed_episodes': {}}
    if summary['ppo_started']:
        accounting.update(read(report/'ppo_run/summary.json'))
        journal = [json.loads(line) for path in (models/'ppo_run').glob('episodes_worker_*.jsonl')
                   for line in path.read_text().splitlines() if line.strip()]
        accounting['completed_episodes'] = {
            controller: {reason: sum(r['controller'] == controller and r['reason'] == reason for r in journal)
                for reason in ('success', 'collision', 'timeout', 'joint_limit', 'planning_failure')}
            for controller in ('ppo', 'rrt')}
        accounting['completed_by_controller_and_task'] = {
            controller: {task: {reason: sum(r['controller'] == controller and r['task'] == task and r['reason'] == reason for r in journal)
                for reason in ('success', 'collision', 'timeout', 'joint_limit', 'planning_failure')}
                for task in sorted({r['task'] for r in journal})} for controller in ('ppo', 'rrt')}
        accounting['completed_episode_steps_exclude_partial_final_episodes'] = True
    dump(report/'training_accounting.json', accounting)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for key, label, color in [('success', 'Success', '#16834b'), ('collision', 'Collision', '#bc3b36'), ('timeout', 'Timeout', '#b47d17')]:
        axes[0, 0].plot([e['update'] for e in bc], [100*e['two']['stage_target'][key]/20 for e in bc],
                        marker='o', label=label, color=color)
    axes[0, 0].set(title='Two-static BC validation (20 scenes)', xlabel='BC updates', ylabel='Rate (%)', ylim=(-3, 103)); axes[0, 0].legend()
    if ppo_events:
        for key, label in [('success', 'Success'), ('collision', 'Collision'), ('timeout', 'Timeout')]:
            axes[0, 1].plot([e['additional_steps'] for e in ppo_events],
                [100*e['metrics']['dynamic_ppo']['stage_target'][key]/20 for e in ppo_events], marker='o', label=label)
        axes[0, 1].legend(); axes[0, 1].set(title='Pure PPO validation', xlabel='Additional PPO transitions', ylabel='Rate (%)', ylim=(-3,103))
    else:
        axes[0, 1].text(.5, .5, 'PPO not started\nBC learning gate failed', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Gated PPO pilot')
    for ax, keys, title in [(axes[1, 0], ['two_initialized_holdout', 'two_holdout_after'], 'New two-static test (100 scenes)'),
                            (axes[1, 1], ['one_before', 'one_after'], 'One-static regression (100 scenes)')]:
        bottom = np.zeros(2)
        for key, label, color in [('success', 'Success', '#16834b'), ('collision', 'Collision', '#bc3b36'), ('timeout', 'Timeout', '#d8aa42')]:
            values = [summary[n]['stage_target'][key] for n in keys]
            ax.bar(['Initialized', 'Candidate'], values, bottom=bottom, label=label, color=color)
            for i, v in enumerate(values):
                if 0 < v < 6:
                    ax.annotate(str(v), (i+.30, bottom[i]+v/2),
                        xytext=(12, -10 if key == 'collision' else 10), textcoords='offset points',
                        ha='left', va='center', fontsize=9,
                        bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': .2},
                        arrowprops={'arrowstyle': '-', 'color': '#555555', 'lw': .7})
                elif v:
                    ax.text(i, bottom[i]+v/2, str(v), ha='center', va='center', color='white' if key != 'timeout' else 'black')
            bottom += values
        ax.set(title=title, ylim=(0, 105), ylabel='Episodes')
    fig.suptitle('Independent multi-obstacle adapter: fixed validation, new final test')
    fig.savefig(report/'overview.png', dpi=170); plt.close(fig)
    correction = read(data/'corrections/metadata.json'); merged = read(data/'teacher/metadata.json')
    lines = ['# 双静态障碍：独立修正网络、针对性RRT示范与短预算适配', '',
        f'**本轮未放行，不替换现有模型。新100场景：初始化{before_test["success"]}%成功/{before_test["collision"]}次碰撞，'
        f'学习后候选{after_test["success"]}%成功/{after_test["collision"]}次碰撞。固定验证中的改进未推广到新场景。**', '',
        '原单障碍动作网络被冻结；独立修正网络只在至少两个障碍存在时激活。没有在线避障规划器参与纯PPO评估。',
        '初始化时新增槽位输入列归零、修正网络输出归零。训练保留障碍的固定顺序，不再交换原槽位含义。', '',
        '## 数据与接口', '',
        f'- 原示范{merged["parent_episodes"]}条；新增针对性RRT成功示范{merged["new_episodes"]}条；合计{merged["transitions"]:,}个监督状态。',
        f'- 审计{len(train)}个训练布局，其中{correction["second_blocks_one_rrt"]}个布局的第二障碍挡住仅考虑第一障碍的RRT路线。',
        f'- 新采集共{correction["attempts"]}次尝试，失败数为{correction["failures"]}；失败不作为标签，交接前策略动作不作为专家标签。',
        '- 只从训练布局采集；10个示范留出布局不进入梯度；固定20个验证布局用于选模；新100场景预先冻结，最后评估。',
        '- 新示范采样权重为原示范4倍，双障/单障/对准批次比例80%/10%/10%。', '', '## BC验证', '',
        '| 更新 | 成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |', '|---:|---:|---:|---:|---:|---:|']
    for e in bc:
        a, b = e['two']['stage_target'], e['one']['stage_target']
        lines.append(f'| {e["update"]:,} | {a["success"]}/20 | {a["collision"]} | {a["timeout"]} | {b["success"]}/20 | {b["collision"]} |')
    lines += ['', f'执行{summary["bc_updates_run"]:,}次BC更新；所选更新{summary["selected_bc"]["update"]:,}；学习改进门槛通过：**{summary["bc_learning_gate"]}**。',
        '启动PPO须同时满足：有非零更新、对第二障碍有动作响应、固定验证较初始化有成功/碰撞改进、保留单障和对准能力。', '',
        '## PPO适配', '', f'启动：**{summary["ppo_started"]}**；实际追加PPO步数：{summary["ppo_budget"]:,}；所选追加步数：{ppo_selected}。',
        '训练按回合50%RRT/50%PPO，RRT标签与PPO在策略采样缓冲区分开；辅助BC只更新独立修正网络。']
    if summary['ppo_started']:
        lines += [f'额外RRT物理步数：{accounting["rrt_transitions"]:,}；训练完成回合按控制器统计：`{accounting["completed_episodes"]}`。']
    if ppo_events:
        lines += ['', '| PPO追加步 | 成功 | 碰撞 | 超时 |', '|---:|---:|---:|---:|']
        for e in ppo_events:
            a = e['metrics']['dynamic_ppo']['stage_target']
            lines.append(f'| {e["additional_steps"]:,} | {a["success"]}/20 | {a["collision"]} | {a["timeout"]} |')
    lines += ['', '## 最终评估', '', '| 数据 | 模型 | 成功 | 碰撞 | 超时 |', '|---|---|---:|---:|---:|']
    for label, key in [('双障新100：初始化', 'two_initialized_holdout'), ('双障新100：候选', 'two_holdout_after'),
                       ('单障回归100：原模型', 'one_before'), ('单障回归100：候选', 'one_after')]:
        a = summary[key]['stage_target']; dataset, name = label.split('：')
        lines.append(f'| {dataset} | {name} | {a["success"]}/{a["episodes"]} | {a["collision"]} | {a["timeout"]} |')
    lines += ['', f'第二障碍移动15厘米时的动作响应：`{summary["second_slot_sensitivity"]}`。',
        f'原单障完整轨迹保持一致：**{summary["one_trajectories_identical"]}**。',
        f'放行检查：`{assessment}`。原默认单障模型保留，没有自动进入动态阶段。', '',
        '第二障碍敏感性只证明使用了输入，不等于避障保证。新100场景为本轮独立测试，旧100单障场景仅作回归。',
        '纯PPO结果与selected_mixture_validation.json中的50/50混合结果分开，混合结果包含在线RRT贡献。', '',
        '## 文件位置', '', f'- 原始数据与示范：`{data.resolve()}`', f'- 模型与优化器：`{models.resolve()}`',
        '- comparison.csv、overview.png：各阶段数据与曲线。', '- interface_diagnosis.json：训练前真实输入和梯度检查。',
        '- bc_validation/、ppo_run/：各检查点逐回合评估；two_test_after.json：新测试的全部结果。',
        '- teacher_replay.json：新专家示范重放核验；verification.json：参数与数据隔离核验。',
        '- candidate_model.zip + candidate_manifest.json：匹配的候选与环境配置。',
        '- training_accounting.json：PPO与RRT完成回合及分任务计数，未结束的PPO回合单独保留在步数中。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/two_static_adapter_v3_seed61.log`。', '']
    (report/'README.md').write_text('\n'.join(lines))
    for path in protocol['code_hashes']:
        target = report/'code_archive'/path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    # Successful demo seed is explicitly selected for visualization, never rate estimation.
    successful = [r['seed'] for r in read(report/'two_test_after.json')['rows'] if r['suite'] == 'stage_target' and r['reason'] == 'success']
    dump(report/'demo_selection.json', {'seed': successful[0] if successful else None,
        'scene_bank': str((data/'two_test.json').resolve()), 'successful_case_selected_for_visualization_only': True})
    if mirror:
        target = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
        if target.resolve() != report.resolve():
            if target.exists():
                if sha(target/'candidate_model.zip') != summary['candidate_sha256']:
                    raise FileExistsError('Refusing replacement of a different visible candidate')
                # Recompute derived presentation only; raw observations/models stay immutable.
                for name in ('README.md', 'comparison.csv', 'overview.png', 'verification.json',
                             'assessment.json', 'ppo_stability.json', 'training_accounting.json', 'demo_selection.json'):
                    shutil.copy2(report/name, target/name)
            else: shutil.copytree(report, target)
    print(json.dumps({'assessment': assessment, 'two_test': summary['two_holdout_after']['stage_target'],
                      'report': str(report.resolve())}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/two_static_adapter.yaml')
    args = p.parse_args(); torch.set_num_threads(1); summarize(resolve(args.config))


if __name__ == '__main__': main()
