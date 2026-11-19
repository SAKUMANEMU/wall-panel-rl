# v3 第一阶段结果：单静态障碍物、无噪声

BC 放行未通过，按预设规则停止；没有启动 PPO，也未进入动态障碍物阶段。

- 开始：2026-10-27T15:08:16.008781+08:00；结束：2026-10-27T15:13:06.544481+08:00；总耗时约 4.84 分钟。
- 专家示范：20 个对准成功回合、80 个避障成功回合；
  100 次尝试，4794 条状态—动作数据，50 个不同场景种子。
- BC 更新：2000 次；检查集选模目标：0.00801880。
- PPO 新增环境步：0。

| 最新验证场景 | 成功/回合 | 碰撞 | 超时 | 关节越界 |
|---|---:|---:|---:|---:|
| 单静态障碍物 | 0/20 | 20 | 0 | 0 |
| 远距离对准 | 10/10 | 0 | 0 | 0 |
| 近距离对准 | 10/10 | 0 | 0 | 0 |

BC 放行：**未通过**。
最低要求：当前阶段至少 20 例、成功率 ≥50%、碰撞率 ≤25%、无关节越界；
远/近对准各至少 10 例、成功率各 ≥90%、无碰撞/越界；目标处最大动作 ≤0.005。
最终 PPO 阶段验收仍为原来的成功率 ≥80%、碰撞率 ≤10% 等条件。
这些是重复使用的工程验证场景，不是独立测试成绩。

## 查看结果

- `overview.png`：BC 误差曲线、最新回合结果、目标动作检查。
- `summary.json` / `validation_summary.csv`：结构化结果。
- `bc_gate.json` / `bc_validation_episodes.json`：放行检查及逐回合结果。
- `teacher_metadata.json` / `bc_metadata.json`：示范统计、训练设置和数据来源。
- `status.json`：流水线结束状态；`prepare.log`、`teacher.log`、`bc.log` 为原始日志。
- 工作副本模型目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/staged_avoidance_v3_seed61/one_static`。
- 示范目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/staged_avoidance_v3_seed61/one_static/teacher`。

旧版本模型和结果保留。未降低安装精度（位置 <1 cm、姿态 <2°），未自动追加训练预算。
