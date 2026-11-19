"""Summarize training-only correction quality and exact base preservation."""
import _bootstrap
from collections import Counter
from pathlib import Path
import shutil
import numpy as np
from run_targeted_bc import read, dump, sha


def main():
    root = Path('results/diagnostics/four_policy_corrections_v3_seed61')
    data = Path('data/four_policy_corrections_v3_seed61')
    assert read(root / 'status.json')['status'] == 'completed'
    summary = read(root / 'summary.json'); meta = summary['teacher']; protocol = read(root / 'protocol.json')
    assert sha(data / 'teacher.npz') == meta['teacher_sha256']
    assert sha(root / 'collector.py') == protocol['code_sha256']
    attempts = read(data / 'attempts.json'); audits = read(data / 'audit.json'); routes = read(data / 'canonical_routes.json')
    accepted = [a for a in attempts if a['accepted']]
    assert len(accepted) == meta['correction_episodes']
    assert all(a['pre_verified'] and a['post_verified'] and a['reason'] == 'success' and a['prefix_error'] <= 1e-6 for a in accepted)
    assert all(r['maximum_error'] <= 1e-6 and r['reason'] == 'success' for r in summary['replay'])
    splits = read(root / 'split.json')
    with np.load(data / 'teacher.npz', allow_pickle=False) as z:
        assert set(z['seeds'].tolist()) == set(splits['train'])
        assert not set(z['seeds'].tolist()) & set(splits['validation'] + splits['test'])
        original_path = Path(protocol['spec']['data_root']) / 'teacher/teacher.npz'
        with np.load(original_path, allow_pickle=False) as old:
            assert all(np.array_equal(z[k][:len(old[k])], old[k]) for k in old.files)
    outcomes = dict(Counter(a['reason'] for a in attempts))
    source = {name: dict(Counter(a['reason'] for a in audits if a['source'] == name)) for name in ('bc1000', 'bc5000')}
    report = {'teacher': meta, 'attempt_outcomes': outcomes, 'source_train_outcomes': source,
              'canonical_routes_verified': sum(r['verified'] for r in routes), 'original_arrays_exact': True,
              'train_only': True, 'replay': summary['replay']}
    dump(root / 'verification.json', report)
    for name in ('attempts.json', 'audit.json', 'canonical_routes.json', 'metadata.json'): shutil.copy2(data / name, root / name)
    lines = ['# 四障碍策略纠正示范（仅训练布局）', '',
        f'- 原64条示范不变，新增{meta["correction_episodes"]}个成功纠正episode，覆盖{meta["corrected_layouts"]}个训练布局。',
        f'- 合并数据：{meta["episodes"]}个episode、{meta["transitions"]}条状态动作转移；原所有数组前缀逐项完全一致。',
        f'- 全部尝试（失败不作标签）：`{outcomes}`。', f'- 两个挖掘策略在训练布局的终止统计：`{source}`。',
        f'- 可验证的保存专家路线：{report["canonical_routes_verified"]}/64。实际路线或保存命令重建，采用原10毫米规划间距、0.005弧度细采样。',
        '- 重放策略到接管点，连接到同场景原专家路线的字面后缀；不另选随机绕行路线，不更换策略观测中的原参考路径。',
        '- 只有连接/执行前复检、实际物理安装成功及执行后复检全部通过的轨迹入库。',
        '- 前3条新增示范含策略前缀进行物理重放，误差不超过1e-6。',
        '- 训练布局仅64个；验证32和测试64未参与纠正采集。布局数与episode数不同，同一布局可有多个安全接管点。',
        f'- 合并数据：`{(data / "teacher.npz").resolve()}`。',
        '- attempts.json / canonical_routes.json / verification.json：全部尝试、原路线与质量证明。', '']
    (root / 'README.md').write_text('\n'.join(lines))
    dest = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics') / root.name
    shutil.copytree(root, dest, dirs_exist_ok=True)
    print(report)


if __name__ == '__main__': main()
