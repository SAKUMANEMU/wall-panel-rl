# 当前保留 PPO 与最终候选：超时诊断和扩大评估

本轮只做推理评估，没有训练、改变权重或自动切换模型。retained 为当前保留模型，candidate 为本轮最终零碰撞候选。
原验证“零碰撞”仅指该候选在原 20 个验证场景中的 0 次碰撞，不是全场景保证。

## 原验证集变化

逐场景结果变化：`{'success -> success': 16, 'timeout -> timeout': 2, 'collision -> timeout': 2}`。

| 种子 | 原模型结果 | 候选位置误差 m | 姿态误差 ° | 末 40 步类型 | 路点已到最后 | 动作抵消比 | 延长至 480 步 |
|---|---|---:|---:|---|---|---:|---|
| 900002 | timeout | 0.3165 | 20.03 | moving_without_task_progress | False | 0.0132 | timeout @ 480 |
| 900003 | timeout | 0.2325 | 11.72 | moving_without_task_progress | False | 0.0066 | timeout @ 480 |
| 900005 | collision | 0.0895 | 7.49 | nearly_stationary | False | 0.0010 | timeout @ 480 |
| 900015 | collision | 0.3467 | 96.19 | slow_task_progress | False | 0.1126 | timeout @ 480 |

抵消比 = 动作均值范数 /（门控基础动作范数＋修正动作范数）；接近 0 表示两者相互抵消。
停滞分类按最后 40 步运动与误差变化进行。路点不前进可能与偏离原参考路径有关，不能仅凭索引不变断言游标存在代码错误。
延长回合只针对原超时案例作诊断，原 240 步正式评估标准和结果保持不变。

## 扩大配对评估

| 场景组 | 模型 | 成功 | 碰撞 | 超时 | 关节越限 |
|---|---|---:|---:|---:|---:|
| 已见 100 场景 | retained | 92/100 | 4 | 4 | 0 |
| 已见 100 场景 | candidate | 86/100 | 6 | 8 | 0 |
| 新 100 场景 | retained | 90/100 | 6 | 4 | 0 |
| 新 100 场景 | candidate | 88/100 | 3 | 9 | 0 |
| 合计 200 场景（描述性） | retained | 182/200 | 10 | 8 | 0 |
| 合计 200 场景（描述性） | candidate | 174/200 | 9 | 17 | 0 |

## 结论与下一步
- 保留模型合计成功 182/200，候选 174/200；候选碰撞少 1 次，但超时多 9 次。本轮证据支持继续保留原模型。
- 原验证 4 个超时中 3 个出现强动作抵消；3 个同时未到最后路点、偏离原参考超过恢复半径 0.3 rad，且未进入终点伺服范围（最大单关节误差 0.2 rad 以内）。这提示参考恢复与终点伺服之间可能存在衔接问题。
- 4 个案例延长到 480 步仍未成功，单纯延长回合不是本轮证据支持的主要改进方向。
- 建议下一步先做参考恢复与终点伺服衔接的单项对照；本轮没有修改控制逻辑、参数或训练模型。

新场景从种子 1002000 起生成，和训练、原验证及此前各批测试分开。模型在生成场景前已固定。
合计 200 场景含 100 个已查看过的场景，不能将整个合计结果称为新的独立测试。

## 逐场景配对变化

- 已见 100 场景：`{'success -> success': 86, 'success -> collision': 2, 'timeout -> timeout': 4, 'success -> timeout': 4, 'collision -> collision': 4}`。
- 新 100 场景：`{'success -> success': 86, 'timeout -> timeout': 4, 'collision -> collision': 3, 'success -> timeout': 4, 'collision -> timeout': 1, 'collision -> success': 2}`。

## 播放动画

已验证的当前保留模型成功案例：

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/demo_autonomous_ppo.py --seed 1000000 --playback-speed 0.6 --hold-seconds 30
```

查看候选时追加：`--model /home/johnsonchen/wall-panel-rl/results/diagnostics/static_stability_v3_seed61/final_candidate_unselected.zip`。
动画使用与评估相同的自主策略环境和已保存的无障碍参考，不会换成专家避障路线。

## 文件
- `summary.json`、`comparison.csv`、`overview.png`：全部比较结果。
- `timeout_comparison.json`、`*_validation_traces.json`、`extended_timeout_*.json`：路点、动作分解、间距及完整轨迹。
- `known_*.json`、`fresh_*.json`：逐回合结果和配对变化；模型及原状态不被覆盖。
- `manifest.json`：模型、代码和配置；`status.json`：完成状态。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/static_candidate_comparison_v3_seed61.log`。
