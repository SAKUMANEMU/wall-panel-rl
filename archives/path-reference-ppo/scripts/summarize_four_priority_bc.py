"""Render four-priority or corrective BC results without changing selection."""
import _bootstrap
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from run_targeted_bc import read, dump, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', default='results/diagnostics/four_corrective_bc_v3_seed61')
    root = Path(p.parse_args().report)
    if read(root / 'status.json')['status'] != 'completed': raise RuntimeError('Study incomplete')
    s = read(root / 'summary.json'); protocol = read(root / 'protocol.json'); study = protocol['study']
    models = Path(study['model_root']); history = read(models / 'bc/history.json')
    changed = []
    for path, expected in protocol['code_hashes'].items():
        if sha(path) != expected:
            assert sha(root / 'code_archive' / path) == expected
            changed.append(path)
    for path, expected in protocol['input_hashes'].items(): assert sha(path) == expected
    assert sha(root / 'selected_model.zip') == s['selected_model_sha256']
    assert [h['additional_bc_updates'] for h in history] == [0, *study['bc']['checkpoints']]
    rows, episodes = [], []
    selected = s['selected_bc']['additional_bc_updates']
    for h in history:
        n = h['additional_bc_updates']; metrics = h['four']
        rows.append({'additional_bc_updates': n, 'total_four_bc_updates': n+study['initial_bc_updates'],
                     'selected': n == selected, **metrics, 'training_mse': h['training_mse']})
        for r in read(root / 'validation' / f'update_{n}.json')['rows']:
            episodes.append({'additional_bc_updates': n, 'seed': r['seed'], 'reason': r['reason'],
                             'steps': r['steps'], 'position_m': r['position_m'], 'orientation_deg': r['orientation_deg']})
    for name, values in [('checkpoints.csv', rows), ('episodes.csv', episodes)]:
        with (root / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(values[0])); w.writeheader(); w.writerows(values)
    losses = [json.loads(line)['loss'] for line in (models / 'bc/loss.jsonl').read_text().splitlines()]
    assert len(losses) == study['bc']['updates'] and np.isfinite(losses).all()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), layout='constrained')
    rolling = np.convolve(losses, np.ones(50)/50, mode='valid')
    axes[0, 0].plot(range(50, len(losses)+1), rolling)
    axes[0, 0].set(title='Four-only BC loss (50-update mean)', xlabel='Additional BC updates', ylabel='Loss')
    x = [h['additional_bc_updates'] for h in history]
    colors = {'success': '#26865b', 'collision': '#bd4246', 'timeout': '#c18f2b'}
    for key in colors: axes[0, 1].plot(x, [h['four'][key] for h in history], 'o-', label=key, color=colors[key], markersize=4)
    axes[0, 1].axvline(selected, color='#666666', ls='--', label=f'Selected +{selected}')
    axes[0, 1].set(title='Frozen four-box validation (32 scenes)', xlabel='Additional BC updates', ylabel='Episodes', ylim=(-.7, 33))
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(x, [h['training_mse'] for h in history], 'o-')
    axes[1, 0].set(title='Training action MSE', xlabel='Additional BC updates', ylabel='MSE')
    if s['test_evaluated']:
        bottom = np.zeros(2)
        for key in colors:
            vals = [s[name][key] for name in ('test_source', 'test_selected')]
            axes[1, 1].bar(['BC1000 source', 'Selected policy'], vals, bottom=bottom, label=key, color=colors[key])
            for i, v in enumerate(vals):
                if v: axes[1, 1].text(i, bottom[i]+v/2, str(v), ha='center', va='center', color='white')
            bottom += vals
        axes[1, 1].set(title='Reserved paired test (64 scenes)', ylabel='Episodes', ylim=(0, 72)); axes[1, 1].legend(frameon=False, fontsize=8)
    else:
        axes[1, 1].axis('off'); axes[1, 1].text(.5, .5, 'No validation gain\n64 test scenes remain reserved', ha='center', va='center')
    fig.savefig(root / 'curves.png', dpi=170); plt.close(fig)
    lines = ['# 四障碍优先BC结果', '',
        f'从上一轮BC1000候选继续，新增{study["bc"]["updates"]}次更新，选中累计{s["selected_total_bc_updates"]}次更新的检查点。PPO新增步数0。', '',
        '| 32个验证场景 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|']
    for label, m in [('续训前BC1000', s['source_validation']), ('选中模型', s['selected_bc']['four']), ('最后候选', history[-1]['four'])]:
        lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} |')
    if s['test_evaluated']:
        lines.extend(['', '| 独立配对测试64例 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|'])
        for label, m in [('续训前BC1000', s['test_source']), ('选中模型', s['test_selected'])]:
            lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} |')
    else: lines.extend(['', '未超过续训前验证成绩，因此64个预留测试场景未评估。'])
    lines.extend(['', '- 四障碍示范采样占100%；双障碍不参与训练或检查点筛选。',
        '- 按四障碍成功数优先、碰撞/超时为同分次序选择；碰撞不高于起点、无关节越界。原物理碰撞、安装精度和240步上限保持原样。',
        '- 双障碍仅作附加诊断：`'+str(s['two_diagnostic'])+'`，没有据此否决候选。',
        '- 仅训练multi_adapter；原网络、价值网络和动作标准差保持不变；零/单障碍函数检查一致。',
        '- 每次验证均使用纯策略，无在线RRT或碰撞动作拒绝。评估器ppo字段表示策略分支，本轮优化为BC。',
        '- 检查点仅由验证集选定，配对测试若执行则发生在选择冻结之后；测试不用于重新选型。',
        '- curves.png / checkpoints.csv：曲线、每个检查点成功/碰撞/超时。episodes.csv / validation/：逐场景详情。',
        f'- 模型及BC优化器状态：`{(models / "bc").resolve()}`。selected_model.zip为本报告的选中候选。',
        '- 原默认演示模型与旧报告均保留。'])
    if study.get('allow_corrections'):
        lines.extend(['', '本轮数据包含仅由64个训练布局采集的策略偏离纠正后缀。安全连接到原有专家路线，执行前后复检及实际成功均通过才入库。',
                      '训练时60%概率抽取所选episode前8步，另外40%覆盖完整episode，强化安全接管状态；原参考路径未替换。'])
    (root / 'README.md').write_text('\n'.join(lines)+'\n')
    dump(root / 'verification.json', {'executed_archive_matches': True, 'current_source_differences': changed,
        'input_hashes_match': True, 'bc_updates': len(losses), 'ppo_steps': 0, 'frozen_state_unchanged': s['frozen_state_unchanged'],
        'two_used_for_selection': False, 'test_used_for_selection': False})
    dest = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics') / root.name
    shutil.copytree(root, dest, dirs_exist_ok=True)
    print(json.dumps({'validation': s['selected_bc']['four'], 'test': s['test_selected'], 'report': str(dest)}, indent=2))


if __name__ == '__main__': main()
