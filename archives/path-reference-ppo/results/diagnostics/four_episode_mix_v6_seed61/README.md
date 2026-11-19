# 四障碍RRT/PPO正式混合训练已完成

**100%完成：PPO10,000步 + RRT10,131步 =20,131次物理转移。后台进程已退出。**

| 新增PPO步数 | 纯PPO成功/碰撞/超时 | 混合成功/碰撞/超时 |
|---:|---|---|
| 0 | 56/8/0 | 60/4/0 |
| 5000 | 55/8/1 | 60/4/0 |
| 10000 | 50/11/3 | 55/7/2 |

**没有选中本轮更新候选；保留0步原模型。选中权重与原PPOv5全部逐项相同。**
- 选中模型的混合验证为60/64（93.75%），纯PPO为56/64（87.5%）。混合优势在训练前已存在，不是本轮训练带来的提升。
- 最后候选的混合验证为55/64（85.94%）：RRT24/24成功，PPO31/40成功、7碰撞、2超时。完整纯PPO验证50/64成功、11碰撞、3超时。
- 每episode独立50%概率抽签，整回合固定；固定评估种子得到24个RRT和40个PPO，同一分配用于全部检查点。
- 正式训练启动149个RRT和145个PPO回合；完成149个RRT（全成功）、141个PPO（57成功/73碰撞/11超时），另4个PPO在预算结束时未完成。
- 独立BC辅助更新10次均通过KL检查，没有拒绝。专家轨迹只在执行成功且前后复检通过时入池，PPO缓冲区不混入专家动作。
- 这些为64例验证结果，不是全新独立测试；原84/100独立测试成绩没有在本轮重新测量，也不能把93.75%归为纯PPO。
- 源模型及旧演示保持不变。selected_model.zip是原策略的选中副本；final_candidate.zip是退化的末候选，不能与选中模型混用。
- curves.png/checkpoints.csv：完整三检查点曲线和计数。history.json及validation/：原始逐场景评估。
- summary.json：回合启动/完成数、两分支物理步数及辅助更新次数。paired_ppo.json/paired_mixture.json：初末场景变化。
- 训练日志：`/home/johnsonchen/wall-panel-rl/logs/four_episode_mix_v6_seed61.log`。
- 模型与优化器：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/four_episode_mix_v6_seed61`。
