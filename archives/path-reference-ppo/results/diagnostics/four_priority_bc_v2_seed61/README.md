# 四障碍优先BC结果

从上一轮BC1000候选继续，新增4000次更新，选中累计1000次更新的检查点。PPO新增步数0。

| 32个验证场景 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 续训前BC1000 | 17 | 15 | 0 |
| 选中模型 | 17 | 15 | 0 |
| 最后候选 | 17 | 15 | 0 |

未超过续训前验证成绩，因此64个预留测试场景未评估。

- 四障碍示范采样占100%；双障碍不参与训练或检查点筛选。
- 按四障碍成功数优先、碰撞/超时为同分次序选择；碰撞不高于起点、无关节越界。原物理碰撞、安装精度和240步上限保持原样。
- 双障碍仅作附加诊断：`{'episodes': 20, 'success': 8, 'collision': 6, 'timeout': 6, 'joint_limit': 0, 'planning_failure': 0}`，没有据此否决候选。
- 仅训练multi_adapter；原网络、价值网络和动作标准差保持不变；零/单障碍函数检查一致。
- 每次验证均使用纯策略，无在线RRT或碰撞动作拒绝。评估器ppo字段表示策略分支，本轮优化为BC。
- 检查点仅由验证集选定，配对测试若执行则发生在选择冻结之后；测试不用于重新选型。
- curves.png / checkpoints.csv：曲线、每个检查点成功/碰撞/超时。episodes.csv / validation/：逐场景详情。
- 模型及BC优化器状态：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/four_priority_bc_v2_seed61/bc`。selected_model.zip为本报告的选中候选。
- 原默认演示模型与旧报告均保留。
