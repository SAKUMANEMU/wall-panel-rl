# 停滞触发的受限路点恢复：固定权重对照

两组使用同一个当前保留 PPO 的全部网络权重。普通恢复半径仍为 0.30rad，终点伺服仍为 0.20rad，回合上限 240 步。
仅条件组会在确认停滞且卡于非终点参考时，临时尝试 0.45rad 前向恢复；每回合最多成功一次、最多尝试三次，尝试间隔至少 20 步。
触发前至少运行 60 步，并观察连续 30 步的位置、姿态、关节净位移与路点进展；不对无障碍对准任务、终止状态或动态场景启用该机制。
改变的是局部参考观测规则，不是 PPO 重新训练或神经网络权重提升；无动作替换或在线专家避障。

| 场景组 | 设置 | 成功 | 碰撞 | 超时 | 关节越限 |
|---|---|---:|---:|---:|---:|
| 原验证 | baseline | 16/20 | 2 | 2 | 0 |
| 原验证 | conditional | 18/20 | 2 | 0 | 0 |
| 已见 200 回归 | baseline | 182/200 | 10 | 8 | 0 |
| 已见 200 回归 | conditional | 188/200 | 10 | 2 | 0 |
| 新 100 场景 | baseline | 93/100 | 4 | 3 | 0 |
| 新 100 场景 | conditional | 94/100 | 4 | 2 | 0 |

预定回归无退化检查：**True**；细项：`{'validation': True, 'combined_200': True, 'alignment_retained': True}`。

## 结论
- 各批配对检查合计有 9 个超时转为成功，没有新碰撞或原成功转失败；391 个未触发恢复的回合保持与原设置一致。
- 已见 200 场景成功 182→188、碰撞仍为 10、超时 8→2；原验证成功 16→18、碰撞仍为 2、超时 2→0。
- 新 100 场景成功 93→94、碰撞仍为 4、超时 3→2。新增场景上的改善只有一个回合，不能夸大为统计显著或训练后提升。
- 结果支持把“停滞后受限恢复”作为下一步接入/微调的候选；本轮没有自动替换默认控制器或训练权重。

## 触发与逐场景检查

- validation：`{'identical_untriggered_episodes': 38, 'triggered_episodes': 2}`；配对结果变化：`{'success -> success': 16, 'timeout -> success': 2, 'collision -> collision': 2}`。
- previous_100：`{'identical_untriggered_episodes': 118, 'triggered_episodes': 2}`；配对结果变化：`{'success -> success': 92, 'timeout -> success': 2, 'timeout -> timeout': 2, 'collision -> collision': 4}`。
- recent_100：`{'identical_untriggered_episodes': 116, 'triggered_episodes': 4}`；配对结果变化：`{'success -> success': 90, 'timeout -> success': 4, 'collision -> collision': 6}`。
- fresh_test：`{'identical_untriggered_episodes': 119, 'triggered_episodes': 1}`；配对结果变化：`{'success -> success': 93, 'collision -> collision': 4, 'timeout -> timeout': 2, 'timeout -> success': 1}`。

## 四个失败案例

| 种子 | 原设置 | 条件恢复 | 实际恢复次数 | 尝试事件 |
|---|---|---|---:|---|
| 900002 | timeout（240步） | success（236步） | 1 | [{'step': 229, 'recovered': True, 'old_index': 6, 'new_index': 49, 'reference_distance_after_rad': 0.3948504750118595, 'position_m': 0.28548351304681346, 'orientation_deg': 16.792898614928014}] |
| 900003 | timeout（240步） | success（205步） | 1 | [{'step': 194, 'recovered': True, 'old_index': 36, 'new_index': 44, 'reference_distance_after_rad': 0.4206529122392661, 'position_m': 0.1977863638697638, 'orientation_deg': 12.29337887255518}] |
| 900005 | collision（59步） | collision（59步） | 0 | [] |
| 900015 | collision（41步） | collision（41步） | 0 | [] |

## 结果范围
- 原验证及 200 回归场景已被查看，属于回归/调参证据，不是新独立测试。
- 仅当回归不退化时才按预定规则运行新 100 场景；没有事后搜索参数。
- 新场景从种子 1004000 起，两组在同一批冻结场景上评估；该批结果不改变本轮已固定的恢复条件。
- 没有修改默认配置、覆盖模型文件、运行训练或自动启动动态阶段。
- 没有触发恢复的回合必须与原设置的终止原因、步数、误差完全匹配，检查结果已记录。

## 文件
- `summary.json`、`comparison.csv`、`overview.png`：完整结果；`non_regression.json`：预定门槛。
- 每组的 `validation.json` 包含四例完整动作/路点轨迹；两个回归 JSON 包含逐回合尝试与触发事件。
- 若运行新测试，每组 `fresh_test.json` 为逐回合新测试记录。
- `manifest.json` / `verification.json`：模型、代码、场景与终点参数核对；未导出伪装成新训练模型的 zip。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/conditional_recovery_v3_seed61.log`。
