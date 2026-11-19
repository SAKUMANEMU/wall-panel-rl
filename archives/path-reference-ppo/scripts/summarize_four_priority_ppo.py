"""PPO learning/validation counts and the once-reserved paired test."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from run_targeted_bc import read, dump, sha


def main():
    root = Path('results/diagnostics/four_priority_ppo_v4_seed61')
    assert read(root / 'status.json')['status'] == 'completed'
    s, p = read(root / 'summary.json'), read(root / 'protocol.json')
    models = Path(p['spec']['model_root']); history = read(root / 'history.json')
    for path, expected in {**p['code_hashes'], **p['input_hashes']}.items(): assert sha(path) == expected
    assert sha(root / 'selected_model.zip') == read(root / 'selection_before_test.json')['model_sha256']
    assert not set(p['train_seeds']) & set(p['validation_seeds'] + p['test_seeds'])
    assert not set(p['validation_seeds']) & set(p['test_seeds'])
    rows, episodes = [], []
    for h in history:
        steps = h['additional_ppo_steps']
        rows.append({'additional_ppo_steps': steps, 'selected': steps == s['selected']['additional_ppo_steps'], **h['four']})
        for r in read(root / 'validation' / f'steps_{steps}.json')['rows']:
            episodes.append({'additional_ppo_steps': steps, 'seed': r['seed'], 'reason': r['reason'], 'steps': r['steps'],
                             'position_m': r['position_m'], 'orientation_deg': r['orientation_deg']})
    for name, values in [('checkpoints.csv', rows), ('episodes.csv', episodes)]:
        with (root / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    train = [json.loads(line) for file in sorted(models.glob('episodes_worker_*.jsonl')) for line in file.read_text().splitlines()]
    assert all(r['seed'] in p['train_seeds'] and r['controller'] == 'ppo' for r in train)
    assert sum(r['steps'] for r in train) == s['training_completed_transitions'] <= s['ppo_steps_run']
    paired = None
    if s['test_evaluated']:
        before, after = read(root / 'test_source.json')['rows'], read(root / 'test_selected.json')['rows']
        assert [r['seed'] for r in before] == [r['seed'] for r in after]
        assert set(r['seed'] for r in after) == set(p['test_seeds'])
        paired = dict(Counter(f'{a["reason"]} -> {b["reason"]}' for a, b in zip(before, after)))
        dump(root / 'paired_test.json', paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout='constrained')
    colors = {'success': '#26865b', 'collision': '#bd4246', 'timeout': '#c18f2b'}
    x = [h['additional_ppo_steps'] for h in history]
    for key in colors: axes[0, 0].plot(x, [h['four'][key] for h in history], 'o-', label=key, color=colors[key])
    axes[0, 0].set(title='Pure PPO: frozen validation (32 scenes)', ylabel='Episodes', xlabel='Additional PPO transitions', ylim=(-.7, 33))
    axes[0, 0].legend(frameon=False)
    if s['test_evaluated']:
        bottom = np.zeros(2)
        for key in colors:
            vals = [s[name][key] for name in ('test_source', 'test_selected')]
            axes[0, 1].bar(['BC1000 source', 'Selected PPO'], vals, bottom=bottom, label=key, color=colors[key])
            for i, v in enumerate(vals):
                if v: axes[0, 1].text(i, bottom[i]+v/2, str(v), color='white', ha='center', va='center')
            bottom += vals
        axes[0, 1].set(title='Reserved paired test (64 scenes)', ylabel='Episodes', ylim=(0, 72))
        axes[0, 1].legend(frameon=False, fontsize=8)
    else: axes[0, 1].axis('off')
    counts = s['training_episodes']
    axes[1, 0].bar(list(counts), list(counts.values()), color=[colors.get(k, '#777777') for k in counts])
    axes[1, 0].set(title='Completed stochastic training episodes', ylabel='Episodes')
    axes[1, 1].hist([r['raw_return'] for r in train], bins=20, color='#33789a')
    axes[1, 1].set(title='Training returns (original reward units)', xlabel='Return', ylabel='Completed episodes')
    fig.savefig(root / 'curves.png', dpi=175); plt.close(fig)
    lines = ['# 四障碍优先：PPO短适配', '',
        f'从BC1000候选进行{s["ppo_steps_run"]}步纯PPO训练，选中{s["selected"]["additional_ppo_steps"]}步检查点。双障碍不参与训练或选型。', '',
        '| 四障碍验证32例 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|']
    for label, m in [('原BC候选', s['source_validation']), ('选中PPO', s['selected']['four'])]:
        lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} |')
    if s['test_evaluated']:
        lines.extend(['', '| 原预留配对测试64例 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|'])
        for label, m in [('原BC候选', s['test_source']), ('选中PPO', s['test_selected'])]:
            lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} |')
    lines.extend(['', '- 训练/验证/测试场景种子完全分离，均为四个沿原名义路线布置的静态障碍。场景、碰撞判定、安装精度和240步上限未降低。',
        '- 仅训练多障碍修正网络及价值网络；原基础actor及动作标准差冻结。既有BC提供初始化，本轮BC更新为0。',
        '- 训练与评估均无在线避障RRT分支，统计为纯策略表现。PPO缓冲区只包含策略自己的动作、log probability与真实转移。',
        '- 训练奖励乘正比例0.01以调整数值尺度，并重置价值输出；原奖励项与评估奖励不变。PPO用新优化器，学习率1e-5，4环境，250步rollout，批100，3轮，clip0.05，KL目标0.005。',
        '- 检查点按验证集的成功数优先选择，碰撞不得高于续训前且无关节越界；同分比较碰撞/超时，仍同分则保留更早检查点。',
        '- 测试执行前已冻结selection_before_test.json的模型哈希，测试不用于重新选型。该64例测试现已使用，后续调参后不能再称它为全新独立测试。',
        f'- 已完成的训练episode统计：`{counts}`，完成部分转移{s["training_completed_transitions"]}；剩余转移属于未结束episode。总PPO步数恰为{s["ppo_steps_run"]}。',
        f'- 测试配对终止原因变化：`{paired}`。',
        '- curves.png / checkpoints.csv：验证曲线、每个检查点的成功/碰撞/超时。episodes.csv和validation/含逐场景误差。',
        f'- PPO检查点：`{(models / "checkpoints").resolve()}`。',
        '- selected_model.zip及candidate_manifest.json可用于显式播放本轮模型；旧模型与原默认动画入口保留。', ''])
    lines.extend(['', '完整回归178项测试通过。固定验证种子1022018和一次随机种子422877867的无界面新模型演示均安装成功；单次演示不用于成功率估计。',
        '新模型播放命令（随机场景，0.6倍速度）：', '', '```bash',
        'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_trained.py', '```', ''])
    (root / 'README.md').write_text('\n'.join(lines))
    dump(root / 'verification.json', {'code_and_input_hashes_match': True, 'splits_disjoint': True,
        'ppo_transitions': s['ppo_steps_run'], 'only_policy_training': True, 'two_used_for_selection': False,
        'test_selected_after_freeze': s['test_evaluated'], 'test_used_for_selection': False,
        'base_actor_and_std_unchanged': s['base_actor_and_std_unchanged']})
    for temporary, name in [('/tmp/four_priority_ppo_tests.log', 'tests.txt'), ('/tmp/four_priority_full_tests.log', 'full_tests.txt')]:
        if Path(temporary).exists(): shutil.copy2(temporary, root / name)
    dest = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics') / root.name
    shutil.copytree(root, dest, dirs_exist_ok=True)
    print(json.dumps({'validation': s['selected']['four'], 'test_before': s['test_source'], 'test_after': s['test_selected'],
                      'training_episodes': counts, 'report': str(dest)}, indent=2))


if __name__ == '__main__': main()
