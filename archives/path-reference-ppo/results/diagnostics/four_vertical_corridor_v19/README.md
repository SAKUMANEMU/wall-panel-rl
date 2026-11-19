# 更大高低错落与实测上下绕行（v19）

基础相对路线高度偏移由8–14cm增至14–22cm，沿路线正负交替。保留O2额外下移5cm，并在下移之后重新要求：锚点总高度跨度至少35cm、相邻锚点高度差至少14cm。动态位移/速度上限仍为18cm/0.12m/s，四个箱体大小不一，目标、奖励、动作接口及240步上限保持。

为避免仅将障碍移远，新布局沿一个预备回合的真实PPO＋预测安全＋恢复轨迹布置，而非只沿原无障碍名义路线布置。预备回合结果不是成功筛选条件；其轨迹只作几何布置依据，原观测参考关节路径仍不改写。最后的物理回合照常判断成功、碰撞与超时，不能保证每个随机场景都会交替上下通过。

`scripts/demo_vertical_weave.py` 默认采用新配置 `configs/four_vertical_corridor.yaml`。旧布局保留在 `scripts/demo_vertical_weave_v18.py` 与 `configs/four_vertical_weave.yaml`。其他明确引用旧配置的入口和已保存旧录像继续反映旧布局；旧v18验证的63/64成功率不适用于新布局。

## 直接查看已验证的上下绕行片段

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/replay_episode.py results/replays/vertical_corridor_1166026_motion_excerpt.json --fps 60 --playback-speed 0.6
```

这是完整物理回合第155–239步的连续真实状态，只跳过开头等待，约7秒播放。没有更改动作、终态或完整回合239步的结果；文件内 `excerpt` 字段明确标记源记录和时间范围。完整回合的动作与等待全部保留在下方文件。

完整回放：

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/replay_episode.py results/replays/vertical_corridor_1166026.json --fps 60 --playback-speed 0.6
```

## 新随机场景与固定种子重新计算

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_vertical_weave.py --controller ppo_progress --fps 60 --playback-speed 0.6
```

每次仍随机起点和布局/相位。先构造预备场景并执行预备回合，再生成更大错落布局、执行正式回合，最后开窗播放。几何不合法时重试；有障碍执行的失败不被替换为成功。生成与仿真更耗时，重复查看推荐直接回放。

增加 `--seed 1166026` 可从头复现选定原始相位场景；增加 `--headless --trajectory-output PATH` 只录制。默认不加 `--controller` 时保持每回合50% RRT /50%增强PPO选择。为确保查看本示例中的控制器，上述命令明确选择 `ppo_progress`。

## 实测上下通过

选定种子1166026，PPO＋预测安全＋局部恢复，不使用全局RRT回退、没有重训。完整回合239步成功、位置误差约1.17mm、姿态0.094°；新入口逐帧复现相同真实状态。最终锚点高度跨度66.95cm，最小相邻高度差32.81cm。

按各障碍的最近XY通过时刻检查，完整墙板的竖直包围盒满足：

| 障碍 | 相对位置 | 仿真时刻 | 墙板中心高度 | 完整墙板竖直净空 |
|---|---|---:|---:|---:|
| O1 | 从下方通过 | 8.45s | 1.111m | 3.04cm |
| O2 | 从上方通过 | 9.55s | 1.211m | 12.68cm |
| O4 | 从下方通过 | 11.85s | 0.558m | 37.93cm |

对应实际升高约10cm、随后降低约65cm；最近XY距离约0.8cm、13.6cm、14.4cm。判据要求时间有序、距箱体XY中心不超过24cm、完整墙板竖直净空至少1cm、相邻高度变化至少7cm且方向交替。不能仅用墙板中心高度认定完整通过。物理碰撞仍由原环境逐子步判断；这些通过事件只是对实际轨迹的后验说明，不是新加的任务硬约束。

## 开发检查范围

12个候选来自之前用过的验证场景，作为示例开发用途：7个新几何构造失败；5个正式回合中2成功、3超时，无碰撞。其中1个成功案例满足上述3个相邻障碍的升降通过判据。完整尝试记录在 `results/diagnostics/four_vertical_corridor_v19/outcomes.json` 和 `probe_*.json`，没有只保存成功案例。

为了减少等待，又保持同一几何分别前移运动相位6/7.5/9秒：1个成功、2个超时，均未同时满足该上下通过判据，因此没有替换原相位示例。结果全部保存在 `phase_variants.json`。这属于明确披露的演示搜索，不能将选中的成功录像解释成成功率。新布局明显更难，尚未重训或开展新的独立泛化评估；原98.4%仅对应旧布局。

三个语义测试验证了时间有序的真实升降、完整墙板净空，以及“中心高但整体未通过”“恒定高度但障碍高低交替”均不能误判。完整和片段录像均通过无窗口回放检查；模型哈希、布局复现和最终高度差见报告的 `verification.json`。


本目录：`trajectory.png` 为实测三维路线与完整墙板高度包围盒；`verified_passages.json` 记录通过事件；`selected_case.json`、`fresh_generated_case.json` 保存布局；`verification.json` 和 `code_archive/` 保存模型哈希与代码；`playback_stats.json`、`excerpt_playback_stats.json` 保存回放验证。
