# 四障碍优先：PPO短适配

从BC1000候选进行10000步纯PPO训练，选中10000步检查点。双障碍不参与训练或选型。

| 四障碍验证32例 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 原BC候选 | 17 | 15 | 0 |
| 选中PPO | 22 | 10 | 0 |

| 原预留配对测试64例 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 原BC候选 | 37 | 27 | 0 |
| 选中PPO | 51 | 13 | 0 |

- 训练/验证/测试场景种子完全分离，均为四个沿原名义路线布置的静态障碍。场景、碰撞判定、安装精度和240步上限未降低。
- 仅训练多障碍修正网络及价值网络；原基础actor及动作标准差冻结。既有BC提供初始化，本轮BC更新为0。
- 训练与评估均无在线避障RRT分支，统计为纯策略表现。PPO缓冲区只包含策略自己的动作、log probability与真实转移。
- 训练奖励乘正比例0.01以调整数值尺度，并重置价值输出；原奖励项与评估奖励不变。PPO用新优化器，学习率1e-5，4环境，250步rollout，批100，3轮，clip0.05，KL目标0.005。
- 检查点按验证集的成功数优先选择，碰撞不得高于续训前且无关节越界；同分比较碰撞/超时，仍同分则保留更早检查点。
- 测试执行前已冻结selection_before_test.json的模型哈希，测试不用于重新选型。该64例测试现已使用，后续调参后不能再称它为全新独立测试。
- 已完成的训练episode统计：`{'success': 37, 'collision': 132, 'timeout': 14}`，完成部分转移9662；剩余转移属于未结束episode。总PPO步数恰为10000。
- 测试配对终止原因变化：`{'success -> success': 37, 'collision -> success': 14, 'collision -> collision': 13}`。
- curves.png / checkpoints.csv：验证曲线、每个检查点的成功/碰撞/超时。episodes.csv和validation/含逐场景误差。
- PPO检查点：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/four_priority_ppo_v4_seed61/checkpoints`。
- selected_model.zip及candidate_manifest.json可用于显式播放本轮模型；旧模型与原默认动画入口保留。


完整回归178项测试通过。固定验证种子1022018和一次随机种子422877867的无界面新模型演示均安装成功；单次演示不用于成功率估计。
新模型播放命令（随机场景，0.6倍速度）：

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_trained.py
```
