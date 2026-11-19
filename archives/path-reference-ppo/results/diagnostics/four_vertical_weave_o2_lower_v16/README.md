# O2 下移五厘米（v16）

GUI 中第二个障碍物 O2（数组索引 1）的锚点、初始中心及整条正弦运动轨迹下移 0.05 m。O1/O3/O4、所有箱体尺寸、运动幅度 0.18 m、速度上限 0.12 m/s、相位及机械臂初末位置均保持。配置：`configs/four_vertical_weave.yaml` 中 `demo_obstacle_height_offsets_m: [0.0, -0.05, 0.0, 0.0]`。

当前使用该配置的动画入口自动应用，旧的演示命令可直接重跑。随机场景完成高度调整后会重新检查起终点、墙面及障碍间完整运动边界；无效的随机场景会重试。固定种子若无效则报错。历史场景文件未改写，源模型权重未变，没有训练。针对已标记调整的场景重复应用不会重复下降。

三个既有演示场景均通过几何校验及真实执行：

| 种子 | 结果 | 步数 |
|---|---|---|
| 1144002 | 成功 | 134 |
| 1144004 | 成功 | 118 |
| 1144008 | 成功 | 201 |

这些是特选示例的回归检查，不能当作新的独立成功率。历史 v15 图表和结果仍记录调整前版本。调整后的原路线碰撞见证被标为历史数据，未将旧见证当作新布局的证明。

播放 O2 下移后的示例：

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_vertical_weave.py --controller ppo_progress --scene-bank results/diagnostics/four_vertical_weave_v15/passage_examples.json --seed 1144004
```

`seed_*.json` 是本次正式动画入口以 headless 模式记录的真实轨迹与结果。`geometry_verification.json` 记录平移、其他变量不变及完整运动边界校验。`adjusted_cases.json` 仅保存本次校验后的布局供检查，重复载入不会重复应用偏移。
