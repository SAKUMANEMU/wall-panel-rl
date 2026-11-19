# 针对性纠偏与 BC 续训对照结果

两组均从原 50% BC 检查点恢复权重、BC Adam 与采样随机状态，各追加 10,000 次更新；PPO 环境步数为 0。
原 20 个验证场景用于选模，独立 100 个场景仅在两组选模完成后评估。

| 模型 | 选中追加更新 | 原验证成功 | 原验证碰撞 | 独立成功 | 独立碰撞 |
|---|---:|---:|---:|---:|---:|
| 原 BC 基线 | 0 | 10/20 | 9/20 | 36/100 | 61/100 |
| 原数据续训 | 0 | 10/20 | 9/20 | 36/100 | 61/100 |
| 原数据＋针对性纠偏 | 10000 | 13/20 | 2/20 | 60/100 | 13/100 |

## 诊断和数据
- 原验证集碰撞步数：[25, 162, 21, 88, 27, 22, 180, 80, 152]。
- 原验证集碰撞对象（每回合、每类型计一次）：{'panel_obstacle': 6, 'arm_self': 2, 'arm_arm_panel': 1}；未能分类：0。
- 纠偏：107 次专家尝试，100 条成功恢复轨迹，3523 条动作标签。
- 接管发生在策略执行 11–123 步后，其中 99 条超过 12 步；失败前缀不作为示范标签。
- 抽样回放：10/10 成功，最大观测差 0.0。
- 纠偏仅来自训练布局；原监督 held-out 布局保持隔离。两组使用相同 held-out 专家状态，因此 MSE 可直接对照。

## 检查点

| 数据组 | 追加更新 | 成功 | 碰撞 | 超时 | 对准成功 | BC 放行 |
|---|---:|---:|---:|---:|---:|---|
| original_only | 0 | 10/20 | 9/20 | 1 | 20/20 | False |
| original_only | 2000 | 0/20 | 20/20 | 0 | 20/20 | False |
| original_only | 4000 | 0/20 | 20/20 | 0 | 20/20 | False |
| original_only | 6000 | 0/20 | 19/20 | 1 | 20/20 | False |
| original_only | 8000 | 1/20 | 17/20 | 2 | 20/20 | False |
| original_only | 10000 | 0/20 | 19/20 | 1 | 20/20 | False |
| targeted_mix | 0 | 10/20 | 9/20 | 1 | 20/20 | False |
| targeted_mix | 2000 | 0/20 | 19/20 | 1 | 20/20 | False |
| targeted_mix | 4000 | 13/20 | 7/20 | 0 | 20/20 | False |
| targeted_mix | 6000 | 1/20 | 17/20 | 2 | 20/20 | False |
| targeted_mix | 8000 | 5/20 | 13/20 | 2 | 20/20 | False |
| targeted_mix | 10000 | 13/20 | 2/20 | 5 | 20/20 | True |

## 文件与解释
- 按原验证集预定规则选出的模型：`selected_bc_model.zip`，来源 `targeted_mix`；独立成绩不参与选模。
- `summary.json`：选模和独立评估；`collision_diagnosis.json`、`validation_traces.json`、`train_traces.json`：碰撞与完整执行轨迹。
- `checkpoint_summary.csv`、`overview.png`：检查点曲线；`correction_replay.json`：物理回放核对。
- `original_only.json`、`targeted_mix.json`：完整 BC 元数据与保留原先验/critic 的检查。
- `independent_*.json`：冻结独立场景上的逐回合结果；这些场景今后已不再是未见测试集。
- 这仍是单个静态障碍物阶段，不代表五个动态障碍物的成绩。只有一个训练随机种子，不能据此宣称统计显著。
- 若选中追加更新为 0，表示训练后检查点未超过初始 BC；完整预算已执行，训练前模型被保留。
- 模型目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/targeted_bc_v3_seed61`；数据目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/targeted_bc_v3_seed61`。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/targeted_bc_v3_seed61.log`。
