# 停滞恢复：低学习率 BC 与短预算 PPO 对照

两组从同一个原验证 65% 的 BC 模型独立出发。回合上限 240 步，成功标准和碰撞判定保持原定义。
PPO 工作者使用额外停滞奖励；选模与报告中的验证使用原奖励。新独立测试在两组选模完成后执行。

| 模型 | 原验证成功 | 碰撞 | 超时 | 新独立成功 | 碰撞 | 超时 |
|---|---:|---:|---:|---:|---:|---:|
| 当前 BC 基线 | 13/20 | 2/20 | 5/20 | 67/100 | 9/100 | 24/100 |
| 低学习率 BC＋停滞示范 | 13/20 | 2/20 | 5/20 | 67/100 | 9/100 | 24/100 |
| PPO＋停滞反馈 | 14/20 | 3/20 | 3/20 | 78/100 | 13/100 | 9/100 |

## 诊断、数据与预算
- 54 个训练布局的初始结果：{'success': 39, 'collision': 8, 'timeout': 7}。停滞分解见 `train_diagnosis.json`，多个回合基础动作与修正动作近乎抵消。
- 40 次恢复尝试，40 条成功示范、339 条专家动作标签。
- 回放检查：10/10 成功，最大观测差 0.0。
- 示范只来自训练场景；原监督 held-out 布局不参与恢复采集。前缀动作不作为专家标签。
- BC：完成 10000 次追加更新，学习率 1e-4；选中追加更新 0。
- PPO：完成 10000 个追加环境步；选中追加步数 10000。完整回合统计见 `ppo_completed_episodes.json`，不与评估成功率混淆。
- 选中追加步数为 0 表示该组完成训练但保留初始模型，避免以退化的最终检查点覆盖它。
- 按原验证规则锁定的总体模型：`ppo_stall`，导出为 `selected_model.zip`。独立测试不参与选模。

## 检查点

| 组别 | 追加更新或环境步 | 成功 | 碰撞 | 超时 | BC 门槛 |
|---|---:|---:|---:|---:|---|
| bc_low_lr | 0 | 13/20 | 2/20 | 5/20 | True |
| bc_low_lr | 2000 | 6/20 | 14/20 | 0/20 | False |
| bc_low_lr | 4000 | 7/20 | 13/20 | 0/20 | False |
| bc_low_lr | 6000 | 12/20 | 8/20 | 0/20 | False |
| bc_low_lr | 8000 | 4/20 | 16/20 | 0/20 | False |
| bc_low_lr | 10000 | 1/20 | 19/20 | 0/20 | False |
| ppo_stall | 0 | 13/20 | 2/20 | 5/20 | True |
| ppo_stall | 5000 | 13/20 | 2/20 | 5/20 | True |
| ppo_stall | 10000 | 14/20 | 3/20 | 3/20 | True |

## 限制和文件
- 当前仅为单个静态障碍物；没有自动进入动态阶段，也没有增加预算。
- 这是两种成套改进的实践比较，不能单独归因于算法、学习率、奖励或数据。只有一个训练随机种子。
- 新测试从种子 990000 起，和此前已见的 980000 测试分开；本轮之后这批场景也已不再是未见测试。
- `summary.json`：全部指标；`independent_*.json`：独立测试逐回合；`checkpoint_summary.csv`、`overview.png`：曲线。
- `manifest.json`：初始化、代码和配置；`bc_low_lr.json`、`ppo_selection.json`：选模细节；`correction_replay.json`：示范回放。
- 模型目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/stall_recovery_v3_seed61`；数据目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/stall_recovery_v3_seed61`。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/stall_recovery_v3_seed61.log`。
