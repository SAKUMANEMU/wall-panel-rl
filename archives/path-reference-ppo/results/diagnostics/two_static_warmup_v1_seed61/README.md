# 双静态障碍：强化BC预热与PPO适配

源模型保持不变；新增双障碍场景使用原始障碍索引[2,3]，原中间障碍仍在第一个槽位。测试场景从种子 1008000 起预先固定。

## RRT示范与BC预热
- 采集 256 条成功轨迹，分组 `{'alignment': 64, 'one_static': 64, 'two_static': 128}`，总计 12,215 个监督状态。
- 对准轨迹使用冻结参考伺服；单/双障碍轨迹使用现有RRT专家。失败尝试保留在teacher/attempts.json。
- BC训练布局 54 个、留出布局 10 个；训练批次权重为双障60%、单障30%、对准10%。
- 交换两个活动障碍的槽位作数据增强，提升相对冻结参考动作偏离较大的状态的监督权重。只更新残差与门控，先验、价值网络和标准差不变。
- 执行 10,000 次BC更新，选择更新 2,500 的模型；PPO启动资格：**False**。

| BC更新 | 双障成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0/20 | 20 | 0 | 18/20 | 0 |
| 1,000 | 2/20 | 17 | 1 | 2/20 | 16 |
| 2,500 | 7/20 | 12 | 1 | 6/20 | 13 |
| 5,000 | 3/20 | 16 | 1 | 2/20 | 16 |
| 10,000 | 4/20 | 16 | 0 | 2/20 | 16 |

## PPO适配
- 是否启动：**False**。
- 启动要求：双障成功≥50%、碰撞≤25%、无越限，同时单障成功≥80%且验证中无碰撞，并保留对准与目标收敛。最终放行要求更严格。
- 预热未达到启动门槛，按预定协议没有执行PPO；不把无效预热模型作为新的默认策略。

## 冻结测试和回归

| 数据 | 模型 | 成功 | 碰撞 | 超时 | 越限 |
|---|---|---:|---:|---:|---:|
| 双障新100 | 原模型 | 0/100 | 100 | 0 | 0 |
| 双障新100 | 候选模型 | 51/100 | 44 | 5 | 0 |
| 单障旧100回归 | 原模型 | 95/100 | 2 | 3 | 0 |
| 单障旧100回归 | 候选模型 | 42/100 | 52 | 6 | 0 |

资格检查：`{'candidate_validation_eligible': False, 'two_holdout_passed': False, 'one_regression_safety_not_worse': False, 'two_updated_ppo_checks_passed': False, 'two_task_ready': False, 'dynamic_progression_ready': False, 'default_model_preserved': True}`。
双障新100未参与训练或选模；单障100是既有回归库。混合执行成绩不能替代纯PPO成绩。
当前默认单障模型始终保留；未启动三障或动态阶段。候选是否可用以assessment.json为准。

## 文件
- comparison.csv / overview.png：阶段成绩和曲线；verification.json：数据、代码及冻结参数核验。
- candidate_model.zip / candidate_manifest.json / candidate_aux_state.pt：候选与匹配配置、辅助优化器状态。
- baseline.json、bc_validation/、ppo_run/：各检查点；two_test_before/after.json、one_before/after.json：逐场景结果。
- teacher_replay.json：三组示范的动作/观测物理重放检查；data_hashes.json记录冻结测试及验证场景哈希。
- 原始示范：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/two_static_warmup_v1_seed61/teacher`；BC/PPO模型与优化器：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/two_static_warmup_v1_seed61`。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/two_static_warmup_v1_seed61.log`。
