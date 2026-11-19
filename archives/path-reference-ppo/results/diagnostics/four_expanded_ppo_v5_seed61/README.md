# 四静态障碍：扩展场景PPO续训

本轮继续20000步，选中新增10000步检查点（本四障碍阶段累计20000步PPO）。BC更新0。

| 全新100例配对测试 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 续训前PPO10k | 74 | 25 | 1 |
| 选中候选 | 84 | 16 | 0 |

| 同100例的最终候选对比 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 确定性、干净观测 | 84 | 16 | 0 |
| 目标位姿观测噪声5mm/0.5deg | 84 | 16 | 0 |
| 干净观测、随机动作 | 57 | 35 | 8 |

- 新64验证：续训前`{'episodes': 64, 'success': 51, 'collision': 13, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}`；选中`{'episodes': 64, 'success': 56, 'collision': 8, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}`。
- 已使用的旧64回归：续训前`{'episodes': 64, 'success': 51, 'collision': 13, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}`；选中`{'episodes': 64, 'success': 55, 'collision': 9, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}`。旧64从未加入训练或检查点选择，不能再称为新独立测试。
- 192训练（原64+新128）、新64验证、新100测试种子互斥；原32验证只作选择同分时的次级比较。
- 剩余13碰撞全部为墙板：后段slot4共11次，前段slot3共2次。诊断数据在相邻four_expanded_v5_seed61目录。
- 源策略扫描训练集发现57个失败布局，采样75%全训练集均匀+25%这些训练失败布局，测试场景不进入采样池。
- Adam状态与价值网络从源PPO完整保留，未重置；正比例训练奖励缩放仍为0.01。基础actor和动作标准差保持冻结，更新多障碍修正和critic。
- 四静态障碍尺寸/布置规则、物理碰撞、安装精度、240步上限不变。双障碍不参与训练或选型，无在线避障规划器或RRT代执行。
- 新测试和噪声条件在训练前声明，模型哈希在测试前冻结。测试结果不重新决定检查点。三个噪声工况共享同100布局，不能当作300个独立新布局。
- 随机动作按每场景固定Torch种子采样；传感噪声只作用目标位姿观测，真实安装判定不放宽。噪声对比无模型更新。
- 训练完成episode：`{'collision': 172, 'success': 124, 'timeout': 19}`，完成部分转移19760，剩余转移属于尚未结束episode，总新增PPO步数恰为20000。
- 新测试配对变化：`{'success -> success': 74, 'collision -> collision': 16, 'collision -> success': 9, 'timeout -> success': 1}`。
- curves.png / checkpoints.csv：新旧验证曲线、各检查点计数。evaluations.csv：逐场景的成功/碰撞/超时和最终误差。
- test_source.json / test_selected.json：新100例配对结果。test_sensor_noise.json / test_action_noise.json：两种噪声各自结果。
- training/progress.csv位于模型目录，保存PPO损失、KL和训练奖励；原始timesteps含继承计数，新增步数需减protocol.continuation.source_counter。
- 模型与优化器检查点：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/four_expanded_ppo_v5_seed61/checkpoints`。


- 新100配对测试保留全部74个原成功场景，9个碰撞和1个超时转为成功。传感噪声条件下，84个成功与16个碰撞的场景归属也保持相同；只代表本噪声级别和本场景集。
- 完整回归182项测试通过。固定验证种子1032001和一次随机种子1302077125的无界面演示安装成功，不把这两个演示加入成功率统计。
- 当前新测试84%尚未达到90%目标；随机动作57%显示动作采样下仍有鲁棒性差距。
- 已完成的训练epoch不代表越多越好：新增10000步是选中检查点，20000步末候选的新64验证为46成功/11碰撞/7超时，未采用。
- 本轮新100例现已使用，后续调参后应另建新测试集再声称独立测试。

播放最新扩展模型（随机布局，0.6倍速度；旧演示入口保留）：

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_expanded.py
```
