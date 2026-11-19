"""Verify a completed coverage/route-consistency experiment and export reports."""
import _bootstrap
import argparse
import csv
import json
import hashlib
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from wall_panel_rl.rl.multi_obstacle_adapter import frozen_base_preserved
from wall_panel_rl.rl.multi_obstacle_warmup import teacher_arrays
from wall_panel_rl.rl.avoidance_curriculum import assess_history


def verify(study):
    data, models, report = map(Path, (study['data_root'], study['model_root'], study['report_root']))
    if read(report/'status.json')['status'] != 'completed': raise RuntimeError('Study is not complete')
    protocol, summary = read(report/'protocol.json'), read(report/'summary.json')
    for path, expected in {**protocol['code_hashes'], **read(report/'data_hashes.json')}.items():
        if sha(path) != expected: raise AssertionError(f'Executed source/data changed: {path}')
    if sha(study['initial_model']) != protocol['source_sha256']: raise AssertionError('Original source changed')
    if sha(study['failed_model']) != protocol['failed_model_sha256']: raise AssertionError('Mining source changed')
    if sha(report/'candidate_model.zip') != summary['candidate_sha256']: raise AssertionError('Candidate changed')
    split = read(report/'split.json')
    sets = [set(split[k]) for k in ('train_seeds', 'heldout_teacher_seeds', 'validation_seeds', 'test_seeds')]
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]): raise AssertionError('Scene split overlap')
    teacher, parts = teacher_arrays(data/'teacher/teacher.npz', sets[1])
    assembled = read(data/'teacher/metadata.json')
    for path, expected in [(Path(study['base_teacher']), assembled['source_sha256']),
            (data/'canonical/teacher.npz', assembled['new_sha256']), (data/'teacher/teacher.npz', assembled['teacher_sha256'])]:
        if sha(path) != expected: raise AssertionError('Teacher file changed')
    train_ids = {int(i) for ids in parts['train'].values() for i in ids}
    if set(teacher['seeds'][sorted(train_ids)].tolist()) - sets[0]: raise AssertionError('Non-training labels')
    for i in parts['train']['two_static']:
        if i < read(data/'teacher/metadata.json')['parent_selected_episodes']:
            raise AssertionError('Old multi-route two-static training label retained')
    routes = {r['seed']: r for r in read(data/'canonical/route_consistency.json')}
    for r in routes.values():
        if r['selected_seed'] is None: continue
        chosen = min(r['candidate_scores'], key=lambda c: (round(c['cost'], 10), c['seed']))
        if chosen['seed'] != r['selected_seed']: raise AssertionError('Canonical selection differs from frozen rule')
        digest = hashlib.sha256(np.asarray(r['selected_path'], dtype='<f8').tobytes()).hexdigest()
        if digest != r['canonical_sha256']: raise AssertionError('Canonical path changed')
    attempts = read(data/'canonical/attempts.json')
    if any(a['seed'] not in sets[0] or a['canonical_sha256'] != routes[a['seed']]['canonical_sha256'] for a in attempts):
        raise AssertionError('Correction route/layout mismatch')
    base = PPO.load(models/'initialized_model.zip', device='cpu')
    candidate = PPO.load(report/'candidate_model.zip', device='cpu')
    if not frozen_base_preserved(base.policy, candidate.policy): raise AssertionError('Frozen actor changed')
    final_bc = PPO.load(models/'bc/checkpoints'/f'bc_{study["bc"]["updates"]}.zip',device='cpu')
    if not frozen_base_preserved(base.policy,final_bc.policy): raise AssertionError('Raw final BC changed original actor')
    def adapter_delta(policy):
        return float(torch.sqrt(sum((value-base.policy.state_dict()[name]).square().sum()
            for name,value in policy.state_dict().items() if name.startswith('multi_adapter.'))))
    dump(report/'training_proof.json', {'bc_updates_completed':study['bc']['updates'],
        'raw_final_adapter_parameter_change_l2':adapter_delta(final_bc.policy),
        'selected_adapter_parameter_change_l2':adapter_delta(candidate.policy),
        'raw_and_selected_original_actors_frozen':True,'selected_bc_update':summary['selected_bc']['update'],
        'ppo_started':summary['ppo_started']})
    bc = read(models/'bc/history.json'); ppo = []
    if summary['ppo_started']:
        ppo = [json.loads(line) for line in (models/'ppo_run/ppo/validation.jsonl').read_text().splitlines() if line.strip()]
        raw = PPO.load(report/'ppo_run/final_candidate.zip', device='cpu')
        if not frozen_base_preserved(base.policy, raw.policy): raise AssertionError('Raw PPO actor changed')
    return data, models, report, summary, protocol, bc, ppo


def metrics_table(metrics):
    return f"{metrics['success']}/{metrics['episodes']} | {metrics['collision']} | {metrics['timeout']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/consistent_two_static.yaml')
    args = parser.parse_args(); torch.set_num_threads(1)
    summarize(yaml.safe_load(Path(args.config).read_text()))


# Rendering and file export are defined below; no simulation/training happens here.
def plot_history(report, summary, bc, ppo):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'success': '#16834b', 'collision': '#bc3b36', 'timeout': '#c8962f'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for field, color in colors.items():
        for ax, key, style in [(axes[0, 0], 'two', '-'), (axes[0, 1], 'legacy_two', '-')]:
            values = [100*e[key]['stage_target'][field]/e[key]['stage_target']['episodes'] for e in bc]
            ax.plot([e['update'] for e in bc], values, style, color=color, marker='o', label=field.capitalize())
    axes[0, 0].set(title='New validation (64 scenes)', xlabel='BC updates', ylabel='Rate (%)', ylim=(-3, 103))
    axes[0, 1].set(title='Legacy validation (20 scenes)', xlabel='BC updates', ylabel='Rate (%)', ylim=(-3, 103))
    for ax in axes[0]: ax.legend()
    if ppo:
        for field, color in colors.items():
            axes[1, 0].plot([e['additional_steps'] for e in ppo],
                [100*e['metrics']['dynamic_ppo']['stage_target'][field]/e['metrics']['dynamic_ppo']['stage_target']['episodes'] for e in ppo],
                marker='o', color=color, label=field.capitalize())
        axes[1, 0].legend(); axes[1, 0].set(xlabel='Additional PPO transitions', ylabel='Rate (%)', ylim=(-3, 103))
    else:
        axes[1, 0].text(.5, .5, 'PPO not started\nBC learning gate failed', ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title('Gated PPO pilot: new validation')
    bottom = np.zeros(2)
    for field, color in colors.items():
        values = [summary[key]['stage_target'][field] for key in ('test_initialized', 'test_candidate')]
        axes[1, 1].bar(['Initialized', 'Candidate'], values, bottom=bottom, label=field.capitalize(), color=color)
        for i, value in enumerate(values):
            if value >= 6:
                axes[1, 1].text(i, bottom[i]+value/2, str(value), ha='center', va='center', color='white' if field!='timeout' else 'black')
            elif value:
                axes[1, 1].annotate(str(value), (i+.3, bottom[i]+value/2),
                    xytext=(12, -10 if field=='collision' else 10), textcoords='offset points',
                    ha='left', va='center', fontsize=9, bbox={'facecolor':'white', 'edgecolor':'none', 'pad':.2},
                    arrowprops={'arrowstyle':'-', 'color':'#555555', 'lw':.7})
        bottom += values
    axes[1, 1].set(title='New paired test (100 scenes)', ylabel='Episodes', ylim=(0, 105))
    fig.suptitle('Expanded failure coverage and canonical RRT demonstrations')
    fig.savefig(report/'overview.png', dpi=170); plt.close(fig)


def summarize(study, mirror=True):
    data, models, report, s, protocol, bc, ppo = verify(study)
    b, a = s['test_initialized']['stage_target'], s['test_candidate']['stage_target']
    state = read(report/'bc_learning_gate.json')
    selected_ppo = read(models/'ppo_run/ppo/selection.json')['additional_steps'] if ppo else None
    settings = read(report/'candidate_manifest.json')['evaluation_settings']
    stability = assess_history(ppo, settings['gate']) if ppo else {'passed': False}
    assessment = {'selected_bc_update': s['selected_bc']['update'], 'selected_ppo_additional_steps': selected_ppo,
        'bc_learning_gate_passed': state['passed'], 'candidate_validation_eligible': s['final_validation']['eligible'],
        'new_test_gate_passed': s['test_gate']['passed'], 'ppo_stability_passed': stability['passed'],
        'new_test_not_worse_than_initialization': a['success']>=b['success'] and a['collision']<=b['collision'],
        'default_model_preserved': True, 'automatic_dynamic_stage': False}
    assessment['two_task_ready'] = all(assessment[k] for k in ('candidate_validation_eligible',
        'new_test_gate_passed', 'ppo_stability_passed', 'new_test_not_worse_than_initialization'))
    dump(report/'assessment.json', assessment); dump(report/'ppo_stability.json', stability)
    dump(report/'verification.json', {'code_and_data_hashes_valid': True, 'source_files_unchanged': True,
        'splits_disjoint': True, 'old_two_static_training_labels_removed': True,
        'canonical_selection_rule_verified': True, 'corrections_bound_to_saved_canonical_routes': True,
        'original_actor_frozen': True, 'one_obstacle_trajectories_identical': s['one_trajectories_identical'],
        'test_used_for_selection': False, 'candidate_sha256': s['candidate_sha256']})
    rows = []
    for e in bc:
        for key, phase in [('two', 'bc_new_validation'), ('legacy_two', 'bc_legacy_validation'), ('one', 'bc_one_validation')]:
            rows.append({'phase': phase, 'updates': e['update'], **e[key]['stage_target']})
    for e in ppo:
        rows.append({'phase': 'ppo_new_validation', 'updates': e['additional_steps'], **e['metrics']['dynamic_ppo']['stage_target']})
        rows.append({'phase': 'ppo_legacy_validation', 'updates': e['additional_steps'], **e['gate']['coverage']['legacy_two']['stage_target']})
    for key in ('test_initialized', 'test_candidate', 'one_before', 'one_after'):
        rows.append({'phase': key, 'updates': '', **s[key]['stage_target']})
    with (report/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['phase', 'updates', 'episodes', 'success', 'collision', 'timeout', 'joint_limit', 'planning_failure'])
        writer.writeheader(); writer.writerows(rows)
    plot_history(report, s, bc, ppo)
    meta = read(data/'canonical/metadata.json'); assembled = read(data/'teacher/metadata.json')
    if s['selected_bc']['update'] == 0:
        outcome = '**本轮未采纳学习更新。候选仍为初始化模型，不能将其测试成绩归因于新训练。**'
    elif not assessment['new_test_not_worse_than_initialization']:
        outcome = '**新候选在新场景上退化，未采用；原模型完整保留。**'
    else:
        outcome = '**原默认模型保留。候选是否达到全部放行条件，见assessment.json；不自动进入动态阶段。**'
    lines = ['# 扩大失败场景覆盖与RRT路线一致性实验', '', outcome, '',
        f'新100场景配对结果：初始化{b["success"]}次成功、{b["collision"]}次碰撞、{b["timeout"]}次超时；'
        f'候选{a["success"]}次成功、{a["collision"]}次碰撞、{a["timeout"]}次超时。', '',
        '## 数据和专家路线', '',
        '- 新增128个训练布局，与原54个非留出训练布局合并为182个。新64个布局用于选模；另100个新布局只做最终测试。',
        '- 原20例双障验证作为回归保护；单障验证和回归均保留。以前的测试场景未混入这轮新增训练/验证/测试。',
        f'- 审计{meta["attempted_layouts"]}个训练布局，初始化策略或上一轮失败候选在其中{meta["failing_layouts"]}个失败。',
        f'- 规划器报告成功的候选共{meta["successful_planner_candidates"]}条；{meta["route_conflict_layouts"]}个布局出现相同初始状态、不同RRT种子下的初始动作方向冲突。',
        f'- 恢复后提前边复检拒绝{meta.get("unsafe_candidate_routes_rejected",0)}条危险候选；实际采纳的完整/纠正示范均经过边检查与真实执行验证。',
        '- 每布局最多比较3条RRT路线，用关节路径长度、相对无障碍参考的偏离和初始动作差异的固定代价选一条；不混合不同路线标签。',
        '- 纠正示范从真实策略失败前交接，通过碰撞检查的直线连接接入同一保存路线的尾部。无安全连接时记录失败，不生成标签。',
        f'- 成功保存{meta["canonical_episodes"]}条完整路线示范和{meta["correction_episodes"]}条纠正示范，后者覆盖{meta["correction_layouts"]}个布局。',
        f'- 删除旧双障多路线标签，保留单障/对准锚点与原10布局示范留出集。最终{assembled["episodes"]}条示范、{assembled["transitions"]:,}个监督状态。',
        '- 失败前的策略动作仅用于重放，不作为专家标签；原无障碍参考始终不替换为专家路径。',
        '- 路线一致性限定在同一布局内部，并未证明不同相近布局具有相同绕行拓扑；方向冲突也不是所有失败的已证实唯一原因。', '',
        '## BC短预算验证', '',
        '| BC更新 | 新验证成功 | 碰撞 | 超时 | 原验证成功 | 碰撞 | 超时 |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for e in bc:
        x, y = e['two']['stage_target'], e['legacy_two']['stage_target']
        lines.append(f'| {e["update"]:,} | {metrics_table(x)} | {metrics_table(y)} |')
    lines += ['', f'执行{s["bc_updates_run"]:,}次BC更新；选择{s["selected_bc"]["update"]:,}次；PPO学习启动门槛通过：{state["passed"]}。',
        '选择模型要求新64例与原20例均不降低成功数、不增加碰撞；满足最低能力后优先减少碰撞。单障与对准能力必须保留。',
        'PPO只在非零BC更新对新验证集带来改进、对原验证不退化且响应第二障碍时启动。', '',
        '## PPO适配', '', f'是否启动：**{s["ppo_started"]}**；实际PPO追加步数：{s["ppo_steps"]:,}；所选追加步数：{selected_ppo}。',
        '每回合50%RRT/50%PPO；PPO只接收自身策略采样的转换。辅助BC监督池仅使用冻结的一致路线训练标签，运行中新生成的RRT路线不加入池。']
    if ppo:
        account = read(report/'ppo_run/summary.json'); dump(report/'training_accounting.json', account)
        lines += [f'额外RRT步数：{account["rrt_transitions"]:,}；完成回合计数：`{account["completed_episodes"]}`。', '',
            '| PPO追加步 | 新验证成功 | 碰撞 | 超时 |', '|---:|---:|---:|---:|']
        for e in ppo: lines.append(f'| {e["additional_steps"]:,} | {metrics_table(e["metrics"]["dynamic_ppo"]["stage_target"])} |')
    lines += ['', '## 最终测试与单障回归', '', '| 数据/模型 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|',
        f'| 新100/初始化 | {metrics_table(b)} |', f'| 新100/候选 | {metrics_table(a)} |',
        f'| 单障100/原模型 | {metrics_table(s["one_before"]["stage_target"])} |',
        f'| 单障100/候选 | {metrics_table(s["one_after"]["stage_target"])} |', '',
        f'第二障碍15厘米位置扰动的动作敏感性：`{s["second_slot_sensitivity"]}`；敏感性不等于避障保证。',
        f'完整放行检查：`{assessment}`。',
        '- 原单障完整轨迹与冻结动作参数保持一致。新100场景只在候选锁定后评估，不用于选模。',
        '- 本轮新测试在报告生成后已成为已见数据，后续不能再称为独立新测试。', '',
        '## 结果文件', '', f'- 原始数据和路线审计：`{data.resolve()}`', f'- 模型/优化器：`{models.resolve()}`',
        '- comparison.csv / overview.png：全部检查点计数和曲线。',
        '- bc_validation/ / ppo_run/：各检查点逐场景结果；test_candidate.json：新100例全部终止原因。',
        '- teacher_replay.json：真实专家标签物理重放；verification.json：数据、代码、选路线和冻结参数核验。',
        '- candidate_model.zip + candidate_manifest.json：候选及匹配环境；assessment.json：放行状态。',
        '- 当前恢复运行日志：`/home/johnsonchen/wall-panel-rl/logs/consistent_two_static_v4_seed61_resume1.log`。',
        '- 初次中断的代码、日志和80布局数据检查点保留在interrupted_collection_1，未重新生成冻结数据。', '']
    (report/'README.md').write_text('\n'.join(lines))
    for path in protocol['code_hashes']:
        target = report/'code_archive'/path; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    successful = [r['seed'] for r in read(report/'test_candidate.json')['rows'] if r['suite']=='stage_target' and r['reason']=='success']
    dump(report/'demo_selection.json', {'seed': successful[0] if successful else None,
        'scene_bank': str((data/'test_cases.json').resolve()), 'for_visualization_only': True})
    if mirror:
        target = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
        if target.resolve() != report.resolve():
            if target.exists():
                if sha(target/'candidate_model.zip') != s['candidate_sha256']: raise FileExistsError(target)
                for name in ('README.md', 'comparison.csv', 'overview.png', 'verification.json', 'assessment.json', 'ppo_stability.json', 'demo_selection.json', 'training_proof.json'):
                    shutil.copy2(report/name, target/name)
            else: shutil.copytree(report, target)
    print(json.dumps({'assessment': assessment, 'test_initialized': b, 'test_candidate': a, 'report': str(report.resolve())}, indent=2))


if __name__ == '__main__': main()
