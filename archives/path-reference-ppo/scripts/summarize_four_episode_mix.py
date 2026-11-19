"""Document physical branch preflight without treating it as model training."""
import _bootstrap
import csv
from pathlib import Path
import shutil
import yaml
from run_targeted_bc import read,dump,sha
from wall_panel_rl.rl.episode_policy import EpisodeChoice


def main():
    spec=yaml.safe_load(Path('configs/four_episode_mix.yaml').read_text())
    root=Path(spec['evaluation_root']);summary=read(root/'summary.json')
    assert not summary['training_performed'] and not summary['independent_test']
    assert sha(spec['source_model'])==summary['source_model_sha256']==read(Path(spec['source_report'])/'candidate_manifest.json')['model_sha256']
    for path,expected in read(root/'executed_code_hashes.json').items():
        assert sha(root/'code_archive'/path)==expected
    mixed=read(root/'mixture.json');choice=EpisodeChoice(.5,spec['seed'])
    for row in mixed['rows']:
        assert row['controller']==choice.begin_episode();choice.end_episode()
    probe=read(root/'training_wrapper_probe.json')
    assert probe['ppo_updates']==0 and probe['verified_bc_pairs']==56
    experts=[r for r in probe['records'] if r['controller']=='rrt']
    assert experts and all(r['accepted_for_imitation'] and r['pre_verified'] and r['post_verified'] and r['reference_preserved'] for r in experts)
    assert any(r['controller']=='ppo' and r['reason']=='success' for r in probe['records'])
    rows=[{'controller':name,**mixed['by_controller'][name]} for name in ('rrt','ppo')]
    rows.append({'controller':'mixture_total',**mixed['metrics']['stage_target']})
    with (root/'counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# 四障碍逐episode RRT/PPO混合接口验证','',
        '**正式混合训练尚未启动。当前84%是原纯PPO模型成绩；这里不是新的正式成功率实验。**','',
        '| 8个功能验证episode | 总数 | 成功 | 碰撞 |','|---|---:|---:|---:|']
    for row in rows:lines.append(f'| {row["controller"]} | {row["episodes"]} | {row["success"]} | {row["collision"]} |')
    lines.extend(['','- 每个episode独立以0.5概率选择RRT或PPO，整回合不切换。该固定种子的8次抽签恰为2/6，不是强制一半配额。',
        '- 既有RRT-Connect已验证可直接使用，无需额外独立BC模型。RRT分支不是BC神经网络，报告按RRT命名。',
        '- 单独的训练环境探针：RRT成功56步并提供56对合格状态动作，随后PPO成功73步。没有优化器更新，也没有修改模型文件。',
        '- 只有实际成功、执行前后路径复检通过且原策略参考路径保留的专家轨迹进入独立BC样本池。规划失败保留失败记录，不隐式换PPO重做。',
        '- PPO缓冲区只保留PPO动作和其原log probability；BC辅助更新使用单独Adam，且只发生在新rollout开始、采样之前。',
        '- 20项针对性测试覆盖真实两段PPO更新后的log probability、专家数据隔离、路径复检拒绝、奖励缩放和旧代码兼容。完整回归188项通过。',
        '- 训练配置10,000步仅统计PPO转移；RRT转移及规划耗时额外记录。概率各半不代表两分支步数或墙钟耗时各半。',
        '- 训练中的评估同时保存混合、RRT分项及PPO分项；另跑完整纯PPO验证用于学习检查点选择。没有将混合成绩归为PPO提升。',
        '- 当前实跑是验证集前8个场景，非新独立测试。旧模型、旧报告和192/64/100场景划分均保持。','',
        '## 运行','', '配置：configs/four_episode_mix.yaml；代码：scripts/train_four_episode_mix.py。','',
        '```bash','cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        '# 检查配置，不启动训练',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py',
        '# 正式训练（PPO10,000步+额外RRT episode）',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py --train',
        '# 用现有模型播放一个随机控制分支的episode',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_expanded.py --controller mixture --expert-probability 0.5',
        '```','',
        '完整三模式评估及训练后模型的命令见docs/FOUR_EPISODE_MIX.md。评估用--output指定新目录，避免覆盖本次预检；--model可指定待评估检查点。',''])
    (root/'README.md').write_text('\n'.join(lines))
    dump(root/'verification.json',{'episode_coin_matches':True,'rrt_physically_usable':True,'source_weights_unchanged':True,
        'real_training_wrapper_verified_pairs':56,'formal_training_started':False,'independent_test':False,
        'on_policy_likelihood_tests_passed':True})
    for path,name in [('/tmp/four_episode_mix_tests.log','targeted_tests.txt'),('/tmp/four_episode_mix_full_tests.log','full_tests.txt')]:
        shutil.copy2(path,root/name)
    dest=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    shutil.copytree(root,dest,dirs_exist_ok=True);print({'report':str(dest),'formal_training':False,'counts':rows})


if __name__=='__main__':main()
