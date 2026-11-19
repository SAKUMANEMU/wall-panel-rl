# 固定帧率录制回放（v17，完成）

动画入口默认先在 DIRECT 模式执行完整 episode，保存配置/场景、逐真实控制步的关节角、墙板位置与四元数、四个动态障碍位置和原始终止结果，然后关闭计算连接、打开 GUI 按目标 60 FPS / 0.6 倍速度回放。预测、局部恢复等计算全部在第一阶段完成。未训练或修改模型，原物理和任务成功标准保持。

已录制文件：`/home/johnsonchen/wall-panel-rl/results/replays/vertical_1144004.json`，也可在工作树的 `results/replays/vertical_1144004.json` 找到。该例保留 O2 下移五厘米的设置。119 个真实记录状态（初始状态＋118步），结果仍为成功；与 v16 的逐状态关节、墙板位置、四障碍位置、仿真时间及结果逐项相同。

新增 5 个回放测试通过：记录端点和四元数插值、损坏记录拒绝、显示不推进物理且保留失败结果、固定帧间隔、低速渲染跳帧逻辑。另 2 个历史动画回归用例通过。真实四动态场景所有记录时刻的关节、墙板位置/方向及障碍位置也逐项核验。独立回放进程不导入 Torch 或 Stable Baselines3；模型路径只是结果元数据，不会加载。

Xvfb 虚拟 GUI 图形实测：目标显示 591 帧，实际 591 帧，跳过 0 帧，目标时长约 9.833 秒、实际约 9.842 秒。平均每帧姿态更新约 4.94 ms，回放期间模型/控制器调用和物理步数均为零。该时序是虚拟图形环境的功能检查，不代表所有桌面负载下都不会丢帧。

直接回放，不重新计算：

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/replay_episode.py results/replays/vertical_1144004.json --fps 60 --playback-speed 0.6
```

原有 `demo_vertical_weave.py` 等命令也可直接使用，现在自动先记录再打开窗口。录制中终端显示 Stage 1 与进度，播放前显示 Stage 2。默认新文件名包含种子及时间戳；设置 `--trajectory-output` 可选择固定路径。`--headless` 只录制，`--playback-mode live` 返回原来的边计算边显示模式。

显示在相邻真实控制状态间插值，未更改源记录或新增物理/碰撞结论；实际等待、停滞及回退仍保留。播放只消除预测计算造成的画面阻塞，不消除算法本身的静止动作。若渲染负载超过目标帧率会跳过迟到帧，不快速补播，并保持最后状态。碰撞或超时的 episode 同样按其实际结果播放。

完整参数说明见 `docs/RECORDED_PLAYBACK.md`。`gui_playback_stats.json` 为图形播放计时，`playback_stats.json` 为无窗口快速姿态验证，`verification.json` 为源结果/姿态/权重及代码哈希记录，`replay_preview.png` 为中途真实记录状态的渲染预览。
