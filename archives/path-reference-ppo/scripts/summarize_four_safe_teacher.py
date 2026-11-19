"""Audit the frozen four-box teacher package; never fit or evaluate a policy."""
import _bootstrap
from collections import Counter
import csv
from pathlib import Path
import shutil
import numpy as np
from run_targeted_bc import read, dump, sha


def main():
    root = Path('results/diagnostics/four_safe_teacher_v1_seed61')
    if read(root / 'status.json')['status'] != 'completed':
        raise RuntimeError('Data preparation is incomplete')
    protocol = read(root / 'protocol.json')
    spec = protocol['spec']
    data = Path(spec['data_root'])
    summary = read(root / 'summary.json')
    metadata = summary['teacher']
    for path, expected in protocol['code_hashes'].items():
        if sha(path) != expected:
            raise AssertionError(f'Collection source changed: {path}')
    for path, expected in read(root / 'data_hashes.json').items():
        if sha(path) != expected:
            raise AssertionError(f'Frozen bank changed: {path}')
    if sha(data / 'teacher/teacher.npz') != metadata['teacher_sha256']:
        raise AssertionError('Teacher bytes changed')
    source = Path(spec['source_report']) / 'selected_model.zip'
    if sha(source) != protocol['source_model_sha256']:
        raise AssertionError('Initialization weights changed')
    banks = {name: read(data / f'{name}_cases.json') for name in ('train', 'validation', 'test')}
    for cases in banks.values():
        for case in cases:
            scene = case['scene']
            assert len(scene['obstacles']) == 4 and not scene['obstacle_motion']
            placement = scene['on_path_placement']
            assert placement['start_goal_valid'] and placement['all_obstacles_intersect_nominal']
            assert len(placement['obstacles']) == 4
            assert all(w['intersects_panel'] or w['intersects_arm'] for w in placement['obstacles'])
    seeds = {name: {int(c['scene']['seed']) for c in cases} for name, cases in banks.items()}
    regression = {int(c['scene']['seed']) for c in read(spec['regression_cases'])}
    assert all(len(seeds[k]) == len(banks[k]) == spec[f'{k}_count'] for k in banks)
    assert all(not seeds[a] & seeds[b] for a in seeds for b in seeds if a != b)
    assert all(not values & regression for values in seeds.values())
    split = read(root / 'split.json')
    assert all(set(split[k]) == seeds[k] for k in banks)
    attempts = read(data / 'teacher/attempts.json')
    accepted = [a for a in attempts if a['accepted']]
    with np.load(data / 'teacher/teacher.npz', allow_pickle=False) as archive:
        d = {k: archive[k] for k in archive.files}
    resumed = (root / 'resume_protocol.json').exists()
    if resumed:
        resume = read(root / 'resume_protocol.json')
        assert sha(root / 'protocol.json') == resume['original_protocol_sha256']
        for path, expected in resume['code_hashes'].items(): assert sha(path) == expected
        checkpoint = root / 'interrupted_checkpoint/teacher'
        assert sha(checkpoint / 'teacher.npz') == resume['partial_teacher_sha256']
        assert sha(checkpoint / 'attempts.json') == resume['partial_attempts_sha256']
        with np.load(checkpoint / 'teacher.npz', allow_pickle=False) as old:
            assert all(np.array_equal(d[key][:len(old[key])], old[key]) for key in old.files)
        assert metadata['resumed_prefix_arrays_exactly_preserved']
    n, episodes = len(d['obs']), len(d['seeds'])
    assert d['obs'].shape == (n, 77)
    assert d['actions'].shape == d['raw_actions'].shape == (n, 7)
    assert d['terminal_obs'].shape == (episodes, 77)
    assert d['terminal_actions'].shape == (episodes, 7)
    assert len(d['offsets']) == episodes + 1 and d['offsets'][0] == 0 and d['offsets'][-1] == n
    assert np.all(np.diff(d['offsets']) > 0)
    for key in ('obs', 'actions', 'raw_actions', 'terminal_obs', 'terminal_actions'):
        assert np.all(np.isfinite(d[key])), key
    assert np.max(np.abs(d['actions']), initial=0) <= 1.000001
    assert np.max(np.abs(d['raw_actions']), initial=0) <= 1.000001
    assert episodes == metadata['episodes'] == len(accepted)
    assert n == metadata['transitions']
    assert len(set(d['seeds'].tolist())) == episodes
    assert set(d['seeds'].tolist()).issubset(seeds['train'])
    assert d['seeds'].tolist() == metadata['accepted_seeds'] == [a['seed'] for a in accepted]
    assert d['tasks'].tolist() == ['four_static'] * episodes
    lengths = np.diff(d['offsets'])
    rows = []
    for i, a in enumerate(accepted):
        assert a['reason'] == 'success' and a['pre_execution_verified'] and a['post_execution_verified']
        assert a['saved_episode'] == i and a['steps'] == int(lengths[i])
        rows.append({'episode': i, 'scene_seed': a['seed'], 'planner_seed': a['planner_seed'],
                     'steps': a['steps'], 'planning_seconds': a['planning_seconds'],
                     'position_m': a['position_m'], 'orientation_deg': a['orientation_deg']})
    replays = read(root / 'teacher_replay.json')
    assert len(replays) == min(3, episodes)
    assert all(r['reason'] == 'success' and r['maximum_error'] <= 1e-6 for r in replays)
    if resumed:
        added_replays = read(root / 'resumed_segment_replay.json')
        assert len(added_replays) == min(3, episodes - resume['preserved_episodes'])
        assert all(r['reason'] == 'success' and r['maximum_error'] <= 1e-6 for r in added_replays)
    with (root / 'demonstrations.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'scene_seed', 'planner_seed', 'steps',
                                             'planning_seconds', 'position_m', 'orientation_deg'])
        writer.writeheader(); writer.writerows(rows)
    counts = dict(Counter(a['reason'] for a in attempts))
    generation_rejections = read(data / 'generation_rejections.json')
    dump(root / 'generation_rejections.json', generation_rejections)
    verification = {'arrays_and_offsets_valid': True, 'splits_disjoint': True,
                    'all_scenes_have_four_static_nominal_route_blockers': True,
                    'original_regression_excluded': True, 'source_weights_unchanged': True,
                    'accepted_routes_physically_successful_and_pre_post_verified': True,
                    'one_route_per_training_layout': True, 'replay_checks': replays,
                    'episodes': episodes, 'transitions': n, 'attempt_outcomes': counts,
                    'ready_for_bc': metadata['ready_for_bc'], 'training_steps': 0}
    if resumed:
        verification.update(resumed_prefix_arrays_exactly_preserved=True, resumed_segment_replay_checks=added_replays)
    dump(root / 'verification.json', verification)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    axes[0].plot(range(1, episodes + 1), lengths, '.', color='#286f9c')
    axes[0].set(xlabel='Accepted demonstration', ylabel='Control steps', title=f'{episodes} successful RRT demonstrations')
    axes[1].bar(list(counts), list(counts.values()), color='#388268')
    axes[1].set(ylabel='Planner/execution attempts', title='All attempts (retries included)')
    for i, value in enumerate(counts.values()): axes[1].text(i, value, str(value), ha='center', va='bottom')
    fig.savefig(root / 'overview.png', dpi=170); plt.close(fig)
    lines = ['# 新四障碍 RRT 示范数据（未训练）', '',
             f'- 独立场景：训练{len(banks["train"])}、验证{len(banks["validation"])}、测试{len(banks["test"])}，种子互斥且不含旧99例。',
             f'- 保存{episodes}个成功episode、{n}条状态动作转移；每个训练布局最多一条路线。',
             f'- 全部尝试结果（包含重试）：`{counts}`。这不是PPO成功率，也不是独立测试成绩。',
             f'- 未解决训练种子：`{metadata["unresolved_train_seeds"]}`；数据完整性标志ready_for_bc={metadata["ready_for_bc"]}。',
             '- 标签同时满足：执行前路线复检通过、真实物理执行成功、执行后路线复检通过。失败尝试不加入标签。',
             '- 新示范使用10毫米规划间距、0.005弧度采样；原99例回归仍为5毫米。物理碰撞阈值、执行器、奖励未变。',
             '- 有限采样不等于连续安全证明；规划时仿真时钟暂停。',
             '- 验证和测试只生成并冻结场景，没有执行含障碍专家episode，也没有用于示范筛选或训练。',
             '- 场景沿冻结初始化策略的无障碍运动布置；生成时的无障碍轨迹仅用于几何布置。生成拒绝见generation_rejections.json。',
             f'- 生成阶段另有{len(generation_rejections)}个候选因无障碍名义轨迹未完整结束而拒绝，已保留原因；这些不属于含障碍RRT执行的失败尝试。',
             '- 策略参考路径保持原值。独立保存raw_actions和执行后的actions，观测77维、动作7维；前三个episode物理重放误差均不超过1e-6。',
             '- 本轮BC/PPO优化步数为0，初始化权重和默认演示策略均未替换。', '',
             '## 文件', '',
             f'- 数据根目录：`{data.resolve()}`',
             '- train_cases.json / validation_cases.json / test_cases.json：冻结场景。',
             '- teacher/teacher.npz：obs、actions、raw_actions、offsets、seeds、tasks、terminal_obs、terminal_actions。',
             '- teacher/attempts.json / metadata.json：全部尝试、接收规则和数据哈希。',
             '- demonstrations.csv / overview.png：逐episode数据概况，不是学习曲线。',
             '- protocol.json / code_archive/ / verification.json：配置、代码、划分和质量验证。', '',
             '后续接入BC前需明确四障碍任务分组和旧技能保留比例；此独立数据集不能直接套用旧混合任务分组。', '']
    if resumed:
        lines.extend([f'中断续采：原{resume["preserved_episodes"]}个episode、{resume["preserved_transitions"]}条转移的所有数组前缀逐项完全一致，原文件备份在interrupted_checkpoint/。',
                      '续采规则与原配置相同；新增段前三个episode另外进行了物理重放。详见resume_protocol.json及resumed_segment_replay.json。', ''])
    (root / 'README.md').write_text('\n'.join(lines))
    shutil.copy2(Path(spec['verified_regression']) / 'tests.txt', root / 'tests.txt')
    dest = Path('/home/johnsonchen/wall-panel-rl/results/diagnostics') / root.name
    if dest.resolve() != root.resolve(): shutil.copytree(root, dest, dirs_exist_ok=True)
    print(verification)
    print('Report:', dest)


if __name__ == '__main__': main()
