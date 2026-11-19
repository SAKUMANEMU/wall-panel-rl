"""Export BC checkpoint counts/curves and provenance into the visible project."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from run_targeted_bc import read, dump, sha


def main():
    root = Path('results/diagnostics/four_obstacle_bc_v1_seed61')
    if read(root / 'status.json')['status'] != 'completed': raise RuntimeError('BC pilot incomplete')
    summary = read(root / 'summary.json'); protocol = read(root / 'protocol.json')
    study = protocol['study']; model_root = Path(study['model_root'])
    for path, expected in {**protocol['code_hashes'], **protocol['input_hashes']}.items():
        if sha(path) != expected: raise AssertionError(f'Executed source/input changed: {path}')
    assert sha(model_root / 'training_pool.npz') == summary['training_pool_sha256']
    assert sha(root / 'selected_model.zip') == summary['selected_model_sha256']
    assert sha(root / 'final_candidate.zip') == summary['final_candidate_sha256']
    with np.load(model_root / 'training_pool.npz', allow_pickle=False) as pool:
        assert not set(pool['seeds'].tolist()) & set(protocol['excluded_scene_seeds'])
        assert set(pool['seeds'].tolist()) == set(protocol['train_scene_seeds'])
    history = read(model_root / 'bc/history.json')
    if [h['update'] for h in history] != [0, *study['bc']['checkpoints']]: raise AssertionError('Missing checkpoint')
    rows, episodes = [], []
    for h in history:
        row = {'bc_updates': h['update'], 'selected': h['update'] == summary['selected_bc']['update'], 'eligible': h['eligible']}
        for group in ('four', 'two'):
            row.update({f'{group}_{k}': h[group][k] for k in ('episodes', 'success', 'collision', 'timeout', 'joint_limit')})
        row.update({f'{group}_success': h['one'][suite]['success'] for group, suite in
                    [('one', 'stage_target'), ('alignment_far', 'alignment_far'), ('alignment_near', 'alignment_near')]})
        row.update({f'train_mse_{g}': v for g, v in h['training_mse'].items()})
        row['one_trajectories_identical'] = h['one_rows_identical']; rows.append(row)
        report = read(root / 'validation' / f'update_{h["update"]}.json')
        for task in ('four', 'two', 'one'):
            for raw in report[task]['rows']:
                episodes.append({'bc_updates': h['update'], 'validation_task': task, 'suite': raw['suite'],
                    'seed': raw['seed'], 'reason': raw['reason'], 'steps': raw['steps'],
                    'position_m': raw['position_m'], 'orientation_deg': raw['orientation_deg'],
                    'controller': 'deterministic_policy'})
    for name, values in [('checkpoints.csv', rows), ('episodes.csv', episodes)]:
        with (root / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    baseline_rows = read(root / 'validation/update_0.json')['four']['rows']
    final_rows = read(root / f'validation/update_{study["bc"]["updates"]}.json')['four']['rows']
    assert [r['seed'] for r in baseline_rows] == [r['seed'] for r in final_rows]
    paired = dict(Counter(f'{a["reason"]} -> {b["reason"]}' for a, b in zip(baseline_rows, final_rows)))
    dump(root / 'paired_baseline_to_final.json', paired)
    loss = [json.loads(line) for line in (model_root / 'bc/loss.jsonl').read_text().splitlines()]
    assert len(loss) == study['bc']['updates'] and all(np.isfinite(r['loss']) for r in loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), layout='constrained')
    updates = [h['update'] for h in history]; selected = summary['selected_bc']['update']
    smooth = np.convolve([r['loss'] for r in loss], np.ones(25)/25, mode='valid')
    axes[0, 0].plot(range(25, len(loss)+1), smooth, color='#287299')
    axes[0, 0].set(title='BC training loss (25-update mean)', ylabel='Loss', xlabel='BC optimizer updates')
    for ax, group, total in [(axes[0, 1], 'four', 32), (axes[1, 0], 'two', 20)]:
        for key, color in [('success', '#218554'), ('collision', '#bb3d42'), ('timeout', '#bd8f2b')]:
            ax.plot(updates, [h[group][key] for h in history], 'o-', label=key, color=color, markersize=4)
        ax.axvline(selected, color='#555555', ls='--', label=f'Selected: {selected}')
        ax.set(title=f'{group.capitalize()} static: frozen validation', ylabel=f'Episodes (of {total})',
               xlabel='BC optimizer updates', ylim=(-.8, total+1))
        ax.legend(fontsize=8, frameon=False)
    for group in ('four_static', 'two_static'):
        axes[1, 1].plot(updates, [max(1e-8, h['training_mse'][group]) for h in history], 'o-', label=group)
    axes[1, 1].set(yscale='log', title='Training action MSE (not validation labels)', ylabel='MSE (floor 1e-8)', xlabel='BC optimizer updates')
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle('Four-obstacle BC pilot: no PPO updates', fontsize=14)
    fig.savefig(root / 'curves.png', dpi=175); plt.close(fig)
    b, chosen, last = history[0], summary['selected_bc'], history[-1]
    lines = ['# 四静态障碍：BC短预算实验', '',
        f'完成{study["bc"]["updates"]}次BC参数更新；PPO新增步数0。每个episode沿原策略参考路径观测，纯策略输出动作，无在线RRT或碰撞动作拒绝。', '',
        '| 32个四障碍验证场景 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|']
    for label, h in [('初始化', b), (f'选中BC {chosen["update"]}', chosen), (f'最终BC {last["update"]}', last)]:
        a = h['four']; lines.append(f'| {label} | {a["success"]} | {a["collision"]} | {a["timeout"]} |')
    lines.extend(['', f'- 选中检查点：{chosen["update"]}次BC更新；短PPO适配准备条件通过：{summary["ready_for_ppo_pilot"]}。本轮未启动PPO。',
        f'- 旧双障碍验证（20例）：初始化`{b["two"]}`，选中`{chosen["two"]}`。',
        '- 短PPO准备条件要求：选中非零更新、有验证改进、保留旧任务，四障碍成功率至少50%且碰撞率不超过25%。',
        '- 旧单障碍/无障碍对准验证40例的轨迹保持完全一致；初始化网络、价值网络和动作标准差冻结，只训练multi_adapter。',
        '- 训练抽样比例：80%四障碍RRT示范、10%旧双障碍、5%旧单障碍、5%对准。旧任务目标是初始化策略动作，用于技能保持。',
        '- 新任务64个训练布局；原验证/测试种子及旧heldout种子排除在梯度之外。各任务可用episode数量见protocol.json。',
        '- 模型选择优先减少碰撞，四障碍成功数最多下降2例，允许避免的碰撞转为超时；同时要求旧双障碍成功/碰撞不退步及单障碍/对准轨迹一致。相同分数保留较早检查点。',
        '- 这32个场景用于检查点选择，因此成绩属于验证结果。64个新测试场景尚未评估。',
        '- 原始评估JSON的mode/controller字段沿用ppo名称，表示纯策略分支；本轮训练算法仅为BC。CSV使用deterministic_policy以明确这一点。',
        '- 初始化模型文件和默认动画模型保持原样。selected_model.zip为选中检查点；final_candidate.zip为最后1000次更新候选。', '',
        '## 文件', '', '- curves.png：BC损失、四障碍与旧双障碍验证曲线。',
        '- checkpoints.csv：每个检查点的成功/碰撞/超时计数及训练MSE。',
        '- episodes.csv / validation/：逐场景结果、误差与完整验证日志。',
        '- summary.json / protocol.json / code_archive/：选型、配置、训练种子与代码来源。',
        f'- 训练与优化器检查点：`{(model_root / "bc").resolve()}`。',
        f'- 初始化至最终候选的四障碍终止原因变化：`{paired}`。', ''])
    if last['two']['success'] < b['two']['success']:
        lines.extend(['观察到四障碍学习与旧双障碍保持之间的干扰：最终四障碍成功数上升，但旧双障碍成功数下降。',
                      'multi_adapter在两个及以上障碍时共同激活，且旧双障碍动作MSE逐步上升；这提示共享修正分支是干扰来源。',
                      '后续优先考虑只对四障碍激活的独立修正分支，保留原双障碍动作，再做短预算验证。', ''])
    if selected == 0:
        lines.extend(['选中初始化检查点，表示没有更新候选满足本轮验证改进与旧技能保持规则。BC确实执行了完整预算，但没有把更新候选作为推荐模型。', ''])
    (root / 'README.md').write_text('\n'.join(lines))
    dump(root / 'verification.json', {'input_and_code_hashes_match': True, 'bc_updates': len(loss),
        'ppo_updates': 0, 'new_test_evaluated': False, 'frozen_state_unchanged': summary['frozen_state_unchanged'],
        'zero_one_actions_identical': summary['zero_one_actions_identical'], 'source_unchanged': summary['source_unchanged']})
    shutil.copy2('/tmp/four_obstacle_bc_interface_tests.log', root / 'interface_tests.txt')
    dest = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics') / root.name
    shutil.copytree(root, dest, dirs_exist_ok=True)
    print(json.dumps({'baseline': b['four'], 'selected_update': selected, 'selected': chosen['four'],
                      'final': last['four'], 'ready_for_ppo_pilot': summary['ready_for_ppo_pilot'], 'report': str(dest)}, indent=2))


if __name__ == '__main__': main()
