# 更大的上下错落与实测上下绕行示例（v15）

四个大小不一的动态障碍物，高度偏移从相对原路线正负 3–7 cm 增加到正负 8–14 cm，沿路线交替上下摆放。障碍物锚点的总高度跨度至少 24 cm，相邻锚点高度差至少 10 cm。相对路线偏移和障碍物绝对高度差是不同指标，路线本身也有高度变化。

运动幅度上限仍为 18 cm，速度上限仍为 0.12 m/s，四个障碍全动态。随机场景保留随机初始机械臂位置、障碍锚点和相位。现有 `demo_progress_dynamic.py`、`demo_predictive_dynamic.py` 的默认高度配置已切到 `configs/four_vertical_weave.yaml`；原版入口另存为对应的 `_v14.py`，历史配置、场景与结果没有覆盖。未启动 PPO/BC 训练，源模型权重未变，真实回合仍为 240 步。

## 验证范围

1. 先对旧场景重新采样更大的高度偏移，仅按几何约束选择 8 个开发场景；8 成功、0 碰撞、0 超时。结果在 `episodes.json`。使用过的源参考路线，不能作为新的独立成功率估计；此前 96/100 的结论也不能直接外推到本高度配置。
2. 上述案例多数从整体上方绕行。因此又对其中 4 个先前的实测执行路线重新布置高低障碍，原始观测参考关节路径未改；1 个无法满足完整运动边界与间距约束，3 个几何可行且全部成功。尝试结果（包括拒绝项）全部在 `passage_probe.json`。
3. 从 3 个成功案例中按实测轨迹相对原观测参考路线的双侧高度偏离量选择示例，而不是把筛选后的样本当成功率测试。固定种子 1144002，PPO＋预测安全＋局部恢复控制器，无全局 RRT 回退。正式演示入口复现：141 步成功、位置误差约 0.75 mm、姿态误差约 0.052°，36 次预测干预、16 个恢复动作。实测轨迹先升高、下降，再升高调整，全部真实动作按原碰撞定义检查。

该例展示上下调整高度后的动态避障成功执行。最近 XY 位置的高度记录并不能证明机械臂在每个障碍处严格交替从整个箱体上方/下方穿过；没有添加“必须交替穿越”的硬约束。若需要强制交替穿行的研究任务，还需定义中间通道或路点约束并单独验证。

## 播放选中的上下绕行成功案例

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_vertical_weave.py --controller ppo_progress --scene-bank results/diagnostics/four_vertical_weave_v15/selected_case.json --seed 1144002
```

青色线是障碍布置依据的先前执行路线，蓝色线是本回合真实执行轨迹，橙色文字标注障碍锚点高度。预测阶段暂停物理时间并隐藏画面刷新。0.6 倍为目标播放速度，预测计算可能使墙钟播放更慢。

## 每次新的随机场景（50% RRT / 50% PPO＋预测安全＋恢复）

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_vertical_weave.py
```

随机场景沿原名义路线摆放，不保证复现固定案例的形态；固定案例则沿实测执行路线重新摆放。随机入口保持两种分支的标注，PPO 权重本身没有新增“上下穿行”训练。

## 文件

- `selected_case.json`：固定演示场景及原观测参考路径。
- `selected_trajectory.json`：固定示例真实轨迹和高度指标。
- `demo_reproduction.json`：通过正式演示入口复现的逐步轨迹与最终指标。
- `trajectory.png`：三维路线及高度随仿真时间的变化。灰色是原观测参考，紫色点线是布置障碍所依据的先前路线，蓝色为实际执行。
- `initial_upper_detour/`：最初主要整体从上方绕行的示例，保留供对照。
- `verification.json`：高度约束、源模型未变、复现结果和代码哈希。

生成/复现开发检查：先执行 `scripts/check_vertical_weave.py --run`，再执行 `scripts/probe_vertical_passage.py` 与 `scripts/plot_vertical_weave.py`。生成脚本拒绝覆盖已有结果目录；如需重做，使用隔离的副本或新报告目录。
