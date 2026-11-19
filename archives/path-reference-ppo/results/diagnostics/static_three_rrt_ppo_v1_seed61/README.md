# 每回合 RRT/PPO 50% 混合训练结果

**本轮没有通过最低能力门槛的模型。所选文件仅是诊断候选，不用于替换当前默认模型。**

已完成 **10,000 个 PPO 采样步**，另执行 **5,080 个 RRT 步**；合计 15,080 个控制步。
当前课程为 3 个静态障碍物、零传感器噪声。RRT/PPO每回合独立抽签，整回合保持所选控制器。

| 追加PPO步数 | 控制方式 | 成功 | 碰撞 | 超时 |
|---:|---|---:|---:|---:|
| 0 | ppo | 0/20 | 1 | 17 |
| 0 | mixture | 8/20 | 1 | 10 |
| 5,000 | ppo | 0/20 | 14 | 6 |
| 5,000 | mixture | 8/20 | 9 | 3 |
| 10,000 | ppo | 0/20 | 20 | 0 |
| 10,000 | mixture | 8/20 | 12 | 0 |

**最终选择追加 0 步检查点**。纯PPO验证：0/20成功、1碰撞、17超时。
同权重的混合执行验证：8/20成功、1碰撞、10超时。
选择标准先保留最低成功率、对准与终点收敛能力，再优先减少碰撞；混合执行成绩不参与纯PPO选模。
上述是已见验证场景，混合成功包含在线RRT的贡献，不是纯PPO能力或新独立测试。
纯RRT及两组远近对准的结果见 `final_evaluation/summary.json`。

## 训练样本与更新
- 实际开始的回合：`{'rrt': 101, 'ppo': 97}`；完成的回合及终止原因：`{'rrt': {'success': 99, 'collision': 2}, 'ppo': {'timeout': 25, 'success': 23, 'collision': 43, 'joint_limit': 2}}`。
- 抽签概率50%不等于有限样本回合数各半，也不等于两种控制器的执行步数各半。规划失败保持RRT标签并计入分母。
- 成功RRT回合提供 5,055 个监督标签；最终监督池 5,055 条。
- 独立BC更新接受 10 次、KL超限回滚 0 次。BC仅更新残差/门控网络，PPO缓冲区不含RRT动作。
- 所选检查点实际已经历 0 次BC更新；对应的优化器、采样RNG和监督池另存为 `selected_aux_state.pt`，其步数已与所选PPO模型核对。
- PPO完整回合以外还包含 190 个未完成回合步；训练回合结果含探索随机性，不能替代确定性验证。
- PPO采样中的恢复/奖励事件：`{'observed_transitions': 10000, 'recovery_attempt_events': 0, 'recovery_success_events': 0, 'stall_penalty_events': 0, 'stall_penalty_sum': 0.0, 'wall_penalty_sum': -84.25694139977972}`。
- 原模型文件、冻结先验参数、运行时代码、步数、监督标签来源和导出哈希均已核验。

## 文件
- `validation_curve.csv`：各检查点纯PPO及混合执行结果；`overview.png`：曲线与采样诊断。
- `summary.json`、`training_episodes.csv`：训练汇总与完整回合；`verification.json`：完整性核验。
- `selected_model.zip` 配合 `manifest.json` / `controller_config.json` 使用；`final_candidate.zip` 为预算结束时模型。
- `final_evaluation/`：所选模型的纯PPO、纯RRT和混合执行逐回合结果。
- 本轮尚无新的独立测试或匹配的纯PPO训练对照，不能把已见验证变化单独归因于BC/RRT。
- 原始训练、BC优化器/监督池快照与检查点：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/static_three_rrt_ppo_v1_seed61`。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/episode_rrt_ppo_v1_seed61.log`。

## 终端演示所选模型

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_episode_mixture.py \
  --model /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/results/diagnostics/static_three_rrt_ppo_v1_seed61/selected_model.zip \
  --manifest /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/results/diagnostics/static_three_rrt_ppo_v1_seed61/manifest.json --playback-speed 0.6
```

追加 `--controller ppo` 可只查看PPO；不指定时每次演示随机选择一个控制器。
