# 四静态障碍：BC短预算实验

完成1000次BC参数更新；PPO新增步数0。每个episode沿原策略参考路径观测，纯策略输出动作，无在线RRT或碰撞动作拒绝。

| 32个四障碍验证场景 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 初始化 | 4 | 28 | 0 |
| 选中BC 25 | 7 | 25 | 0 |
| 最终BC 1000 | 17 | 15 | 0 |

- 选中检查点：25次BC更新；短PPO适配准备条件通过：False。本轮未启动PPO。
- 旧双障碍验证（20例）：初始化`{'episodes': 20, 'success': 15, 'collision': 4, 'timeout': 1, 'joint_limit': 0, 'planning_failure': 0}`，选中`{'episodes': 20, 'success': 16, 'collision': 4, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}`。
- 短PPO准备条件要求：选中非零更新、有验证改进、保留旧任务，四障碍成功率至少50%且碰撞率不超过25%。
- 旧单障碍/无障碍对准验证40例的轨迹保持完全一致；初始化网络、价值网络和动作标准差冻结，只训练multi_adapter。
- 训练抽样比例：80%四障碍RRT示范、10%旧双障碍、5%旧单障碍、5%对准。旧任务目标是初始化策略动作，用于技能保持。
- 新任务64个训练布局；原验证/测试种子及旧heldout种子排除在梯度之外。各任务可用episode数量见protocol.json。
- 模型选择优先减少碰撞，四障碍成功数最多下降2例，允许避免的碰撞转为超时；同时要求旧双障碍成功/碰撞不退步及单障碍/对准轨迹一致。相同分数保留较早检查点。
- 这32个场景用于检查点选择，因此成绩属于验证结果。64个新测试场景尚未评估。
- 原始评估JSON的mode/controller字段沿用ppo名称，表示纯策略分支；本轮训练算法仅为BC。CSV使用deterministic_policy以明确这一点。
- 初始化模型文件和默认动画模型保持原样。selected_model.zip为选中检查点；final_candidate.zip为最后1000次更新候选。

## 文件

- curves.png：BC损失、四障碍与旧双障碍验证曲线。
- checkpoints.csv：每个检查点的成功/碰撞/超时计数及训练MSE。
- episodes.csv / validation/：逐场景结果、误差与完整验证日志。
- summary.json / protocol.json / code_archive/：选型、配置、训练种子与代码来源。
- 训练与优化器检查点：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/four_obstacle_bc_v1_seed61/bc`。
- 初始化至最终候选的四障碍终止原因变化：`{'success -> success': 4, 'collision -> collision': 15, 'collision -> success': 13}`。

观察到四障碍学习与旧双障碍保持之间的干扰：最终四障碍成功数上升，但旧双障碍成功数下降。
multi_adapter在两个及以上障碍时共同激活，且旧双障碍动作MSE逐步上升；这提示共享修正分支是干扰来源。
后续优先考虑只对四障碍激活的独立修正分支，保留原双障碍动作，再做短预算验证。
