# 双静态障碍：强化BC预热与PPO适配


**本轮未采纳任何BC/PPO更新。最终候选仍是仅做初始化修正的模型；新增槽位输入权重仍为零，尚未学会利用第二障碍信息。这些测试成绩不能归因于本轮训练。**
**本报告为初始化修正及原单障策略保护后的独立试验。第一轮失败记录完整保留。**
仅将原来未激活槽位的输入列归零，未训练时双障验证变为 15/20，碰撞 4；单障保持原行为。
初始化修正模型在新100场景中为 85/100成功、10碰撞、5超时。
动作网络只允许新增槽位输入列更新，所有原单障动作连接保持不变；PPO价值网络可单独更新。
最终测试同时列出原模型、初始化修正模型和学习后模型，区分初始化与训练的贡献。

源模型保持不变；新增双障碍场景使用原始障碍索引[2,3]，原中间障碍仍在第一个槽位。测试场景从种子 1010000 起预先固定。

## RRT示范与BC预热
- 复用已验证的 256 条成功轨迹，分组 `{'alignment': 64, 'one_static': 64, 'two_static': 128}`，总计 12,215 个监督状态。
- 对准轨迹使用冻结参考伺服；单/双障碍轨迹使用现有RRT专家。失败尝试保留在teacher/attempts.json。
- BC训练布局 54 个、留出布局 10 个；训练批次权重为双障60%、单障30%、对准10%。
- 交换两个活动障碍的槽位作数据增强，提升相对冻结参考动作偏离较大的状态的监督权重。只更新残差与门控，先验、价值网络和标准差不变。
- 执行 2,500 次BC更新，选择更新 0 的模型；PPO启动资格：**True**。

| BC更新 | 双障成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |
|---:|---:|---:|---:|---:|---:|
| 0 | 15/20 | 4 | 1 | 18/20 | 0 |
| 500 | 10/20 | 8 | 2 | 18/20 | 0 |
| 1,000 | 7/20 | 11 | 2 | 18/20 | 0 |
| 2,500 | 9/20 | 9 | 2 | 18/20 | 0 |

## PPO适配
- 是否启动：**True**。
- 启动要求：双障成功≥50%、碰撞≤25%、无越限，同时单障成功≥80%且验证中无碰撞，并保留对准与目标收敛。最终放行要求更严格。
- 执行 10,000 个PPO步、额外 7,130 个RRT步，选择追加 0 步。
- BC预热后PPO使用新的Adam和5e-5学习率；每回合50%RRT/50%PPO。辅助BC用训练示范初始化监督池，每rollout更新3批，权重.2。

| PPO追加步 | 双障成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |
|---:|---:|---:|---:|---:|---:|
| 0 | 15/20 | 4 | 1 | 18/20 | 0 |
| 5,000 | 15/20 | 4 | 1 | 18/20 | 0 |
| 10,000 | 14/20 | 4 | 2 | 18/20 | 0 |

所选模型的50/50混合执行验证：`{'episodes': 20, 'success': 17, 'collision': 2, 'timeout': 1, 'joint_limit': 0, 'planning_failure': 0}`；该成绩包含在线RRT贡献。

## 冻结测试和回归

| 数据 | 模型 | 成功 | 碰撞 | 超时 | 越限 |
|---|---|---:|---:|---:|---:|
| 双障新100 | 原模型 | 0/100 | 100 | 0 | 0 |
| 双障新100 | 初始化修正、未学习 | 85/100 | 10 | 5 | 0 |
| 双障新100 | 所选候选 | 85/100 | 10 | 5 | 0 |
| 单障旧100回归 | 原模型 | 95/100 | 2 | 3 | 0 |
| 单障旧100回归 | 候选模型 | 95/100 | 2 | 3 | 0 |

资格检查：`{'candidate_validation_eligible': False, 'two_holdout_passed': True, 'one_regression_safety_not_worse': True, 'two_updated_ppo_checks_passed': False, 'two_task_ready': False, 'dynamic_progression_ready': False, 'default_model_preserved': True}`。
双障新100未参与训练或选模；单障100是既有回归库。混合执行成绩不能替代纯PPO成绩。
当前默认单障模型始终保留；未启动三障或动态阶段。候选是否可用以assessment.json为准。

## 文件
- comparison.csv / overview.png：阶段成绩和曲线；verification.json：数据、代码及冻结参数核验。
- candidate_model.zip / candidate_manifest.json / candidate_aux_state.pt：候选与匹配配置、辅助优化器状态。
- baseline.json、bc_validation/、ppo_run/：各检查点；two_test_before/after.json、one_before/after.json：逐场景结果。
- teacher_replay.json：三组示范的动作/观测物理重放检查；data_hashes.json记录冻结测试及验证场景哈希。
- 原始示范：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/two_static_warmup_v1_seed61/teacher`；BC/PPO模型与优化器：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/protected_two_static_v2_seed61`。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/protected_two_static_v2_seed61.log`。
