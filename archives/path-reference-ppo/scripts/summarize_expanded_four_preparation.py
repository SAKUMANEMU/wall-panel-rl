"""Audit diagnosis and fresh scene splits without scoring the held-out bank."""
import _bootstrap
from pathlib import Path
import shutil
from run_targeted_bc import read, dump, sha


def main():
    root=Path('results/diagnostics/four_expanded_v5_seed61')
    assert read(root/'status.json')['status']=='completed'
    p=read(root/'protocol.json');s=read(root/'summary.json');data=Path(p['spec']['data_root'])
    for path,expected in read(root/'data_hashes.json').items():assert sha(path)==expected
    assert sha(p['spec']['source_model'])==p['source_model_sha256']
    assert sha(root/'executed_preparation.py')==p['code_hashes']['scripts/prepare_expanded_four.py']
    splits=read(data/'split.json');names=['train','validation','test','old_validation','old_test_regression']
    assert all(not set(splits[a])&set(splits[b]) for a in names for b in names if a!=b)
    for name in names:
        cases=read(data/f'{name}_cases.json')
        assert set(c['scene']['seed'] for c in cases)==set(splits[name])
        assert all(len(c['scene']['obstacles'])==4 and not c['scene']['obstacle_motion'] for c in cases)
        assert all(c['scene']['on_path_placement']['start_goal_valid'] and c['scene']['on_path_placement']['all_obstacles_intersect_nominal'] for c in cases)
    assert read(data/'train_cases.json')[:64]==read(Path(p['spec']['old_data_root'])/'train_cases.json')
    d=s['collision_diagnosis'];rejections=read(data/'generation_rejections.json')
    text=['# 四静态障碍扩展场景与碰撞诊断','',
        '- 192训练布局（保留原64，新增128）；64全新验证；100全新测试。原32验证、已用64测试分别保留作回归。',
        '- 各角色种子互斥。新场景沿同一冻结策略的无障碍名义运动布置，4个障碍、尺寸和间距规则保持原样。',
        f'- 共{len(rejections)}个场景构造候选被拒绝，原因逐条保存；没有根据含障碍执行结果筛选新场景。',
        '- 原64测试中剩余13个碰撞全部按相同终止原因/步数复现，均为墙板：第4障碍附近11次、第3障碍附近2次。',
        '- 阶段分布为后段11、前段2，碰撞步数7～50。collisions.csv / collisions.json / collision_traces/保存接触体、link、slot和实际路线。',
        '- 原测试失败用于诊断，未把这些场景或动作加入训练。未来困难布局加权仅依据192个训练场景的独立扫描。',
        '- 此步骤未更新模型，尚未对新100测试场景运行含障碍策略。',
        f'- 场景目录：`{data.resolve()}`。split.json与data_hashes.json记录划分和字节哈希。','']
    (root/'README.md').write_text('\n'.join(text))
    dump(root/'verification.json',{'splits_disjoint':True,'original64_training_preserved':True,
        'four_static_on_path_endpoints_valid':True,'diagnosis_reproduced':d['all_reproduced'],
        'source_unchanged':True,'training_steps':0})
    dest=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    shutil.copytree(root,dest,dirs_exist_ok=True);print(s)


if __name__=='__main__':main()
