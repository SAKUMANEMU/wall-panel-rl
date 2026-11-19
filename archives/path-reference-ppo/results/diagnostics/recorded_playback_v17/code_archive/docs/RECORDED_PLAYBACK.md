# 先仿真记录，再固定帧率回放

所有使用 `demo_autonomous_ppo.main` 的动画入口默认先通过 DIRECT 模式完成整段真实仿真，再打开 GUI 回放。预测避障和恢复搜索在第一阶段完成，不阻塞第二阶段的画面。源模型、物理步长、奖励、成功/碰撞判定和 episode 上限没有改变。

## 直接播放已录制的案例

已经录制了 O2 下移 5 cm 后的种子 1144004：118 步成功。此命令不加载模型、不规划、不再次推进物理仿真：

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/replay_episode.py results/replays/vertical_1144004.json --fps 60 --playback-speed 0.6
```

## 重新仿真并自动播放

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_vertical_weave.py --controller ppo_progress --scene-bank results/diagnostics/four_vertical_weave_v15/passage_examples.json --seed 1144004 --trajectory-output results/replays/vertical_1144004.json --fps 60 --playback-speed 0.6
```

先显示 `Stage 1` 和仿真进度，结束并保存后显示 `Stage 2`，随后打开动画窗口。第一阶段仍可能需要几十秒，第二阶段不再等待模型预测。成功、碰撞、超时等真实终止结果原样记录，没有筛选、替换或重新判定成功。

删除 `--scene-bank ... --seed ...` 可生成新的随机场景；删除 `--controller ppo_progress` 恢复每回合 50% RRT / 50% PPO＋预测安全＋局部恢复的默认选择。改变模型或场景后须重新录制，旧回放文件始终反映录制时的场景。

## 参数与文件

- `--playback-mode recorded`：默认，先录制后播放。
- `--playback-mode live`：边计算边显示，保留原模式，可能因预测计算卡顿。
- `--headless`：仅录制，不打开窗口；结合 `--trajectory-output` 选择文件位置。
- `--fps 60`：目标显示帧率，不改变控制频率或物理步长。
- `--playback-speed 0.6`：仿真时间相对播放墙钟时间的比例；0.6 倍更慢。
- `--hold-seconds 30`、`--start-hold-seconds 2`：回放前后停留时间；可设为 0。
- 未指定 `--trajectory-output` 时自动生成 `results/replays/episode_种子_时间戳.json`，终端打印完整路径。

回放文件包含配置、场景、原始结果和逐控制步的关节角、墙板位置/四元数、障碍物中心及仿真时间。原先只含位置的旧 JSON 不能直接回放，需要用更新后的演示脚本重新录制。`scripts/replay_episode.py` 不会应用当前高度偏移配置再次移动障碍物，它精确使用录制时布局。

播放以墙钟帧截止时间调度，关节角和位置做线性插值，墙板姿态做最短弧 SLERP。插值仅用于显示，不构成额外物理仿真或新的碰撞检测结论。真实避障等待、停滞与回退会保留，这些静止属于算法行为，不是预测计算阻塞。若显示器或系统负载跟不上目标帧率，会跳过迟到帧并保留最终状态，避免快速补播造成新的卡顿。

## 验证

录制后所有 119 个真实状态与 O2 调整后的旧执行记录一致，结果仍为 118 步成功。四个障碍物、关节和墙板姿态均核验；回放期间不执行控制器、不推进物理，独立回放进程无需导入 Torch 或 Stable Baselines3。五个新增回放测试和两个历史演示回归用例通过。

图形回放在 Xvfb 的独立窗口中按 60 FPS /0.6x 实测，目标时长约 9.833 s，591 个显示帧；详细计时在 `results/diagnostics/recorded_playback_v17/gui_playback_stats.json`。真实桌面的显卡/窗口负载仍会影响显示时序。
