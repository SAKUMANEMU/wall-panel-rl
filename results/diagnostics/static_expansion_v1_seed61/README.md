# 单障碍新场景验证与三障碍静态扩展

先冻结100个新的完整场景，各自投影成单障碍和三障碍版本；同一场景的起点、目标、参考路径和中间障碍一致。
保持目标精度、零噪声、77维观测、7维动作、奖励和条件恢复规则；增加静态障碍数。
场景种子从 1006000 开始，均与既有场景及训练/验证数据分开。几何生成失败 0 次，单独记录；未按策略或RRT成败筛选场景。

| 场景与模型 | 成功 | 碰撞 | 超时 | 越限 | 规划失败 |
|---|---:|---:|---:|---:|---:|
| 单障碍新100（训练前） | 95/100 | 2 | 3 | 0 | 0 |
| 三障碍验证（训练前） | 0/20 | 1 | 17 | 2 | 0 |
| 三障碍验证（所选模型） | 0/20 | 1 | 17 | 2 | 0 |
| 三障碍留出100（训练前模型） | 0/100 | 4 | 90 | 6 | 0 |
| 三障碍留出100（所选模型） | 0/100 | 4 | 90 | 6 | 0 |
| 单障碍同100回归（三障碍所选模型） | 95/100 | 2 | 3 | 0 | 0 |

单障碍新100是当前固定模型的前瞻检查；三障碍100在训练和选模结束后评估，未传入训练或选模。
后续用三障碍所选模型重放单障碍同100属于回归检查，不是第二批新独立测试。

## 三障碍短预算训练

- 追加 10,000 个PPO步及 5,080 个RRT步；选择追加 0 步的检查点。
- 从原模型继续PPO权重和PPO优化器。新阶段重新建立三障碍RRT示范池，并初始化本阶段独立BC优化器；辅助学习率/权重等数值保持原配置。
- 实际开始的回合：`{'rrt': 101, 'ppo': 97}`；完成的回合：`{'rrt': {'success': 99, 'collision': 2}, 'ppo': {'timeout': 25, 'success': 23, 'collision': 43, 'joint_limit': 2}}`。
- BC更新接受 10 次、KL超限回滚 0 次，监督池 5,055 条。
- **PPO训练回合中，三障碍静态成功 0 次，无障碍对准成功 23 次。对准成功不能当作三障碍成功。** 详细分类见 `training_by_task.json`。
- 三障碍纯RRT验证：`{'episodes': 20, 'success': 20, 'collision': 0, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}`；混合执行：`{'episodes': 20, 'success': 8, 'collision': 1, 'timeout': 10, 'joint_limit': 1, 'planning_failure': 0}`。混合成绩包含RRT贡献。

## 扩展门槛与当前默认模型

- 两个不同更新检查点连续达标：**False**。
- 所选三障碍模型验证达标：**False**；三障碍100留出检查达标：**False**。
- 单障碍回归的碰撞优先评分不下降：**True**。
- 三障碍阶段整体准备就绪：**False**。
- 默认演示与模型继续保留此前已验证的单障碍版本。未自动进入缓慢移动障碍物阶段。
- 未达到门槛的三障碍“所选模型”仅是本轮诊断候选，不能等同于已通过部署/课程升级检查。
- 选模仍采用既定最低能力及碰撞优先规则，没有依据留出100的成绩重新选模型或调整超参数。

## 三障碍演示

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_episode_mixture.py \
  --model /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/results/diagnostics/static_three_rrt_ppo_v1_seed61/selected_model.zip \
  --manifest /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/results/diagnostics/static_three_rrt_ppo_v1_seed61/manifest.json \
  --scene-bank /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/static_expansion_v1_seed61/three_validation.json --seed 900000 --playback-speed 0.6
```

追加 `--controller ppo` 可单独查看PPO，追加 `--controller rrt` 可单独查看专家。

## 固定权重的输入排列诊断

在相同20例物理场景中，仅将障碍输入顺序从[1,2,3]改为[2,1,3]，固定源模型结果为 0/20成功、20碰撞、0超时。
原始顺序为0/20成功、1碰撞、17超时、2越限。两种顺序均未恢复成功，且行为差异明显。
该诊断提示模型对障碍槽位排列敏感，但不能单独证明失败的唯一原因；诊断顺序未用于训练、选模或新100场景。

## 文件

- `summary.json` / `comparison.csv` / `overview.png`：整体结果。
- `one_before.json` / `one_after.json`：单障碍前瞻检查和回归逐回合结果（若执行训练）。
- `three_before_validation.json`、`three_holdout_before.json`、`three_holdout_after.json`：三障碍结果（若执行训练）。
- `protocol.json`、`frozen_banks.json`、`verification.json`：预定规则、冻结数据和完整性核验。
- 三障碍训练报告：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/results/diagnostics/static_three_rrt_ppo_v1_seed61`。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/static_expansion_v1_seed61.log`。
