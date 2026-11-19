# 四个静态障碍物：放在初始化模型的运动路线上

**当前位置由初始化模型在无障碍环境中的真实仿真运动轨迹决定，不再仅靠近参考路线随机摆放。初始化权重未改，未训练。**

- 沿整段轨迹约20%、40%、60%、80%的位置分布，带少量随机偏移；观测槽位先保留两个中段障碍。
- 每个障碍都保存独立的关节位形见证：在该位形下，与机械臂或墙板存在实际几何相交。
- 起点与终点保持有效，障碍之间至少0.1米间距；大小沿用原四障配置；障碍静止。
- 青色线是无障碍时的轨迹，蓝色线是本次带障碍执行轨迹。无障碍推理和几何放置在播放前完成，不是训练。
- 每次默认随机初始位置和布局，--seed可复现。仅在无完整无障碍路线/几何无法放置时重试，不按带障碍执行是否成功筛选。

布置尝试100个固定场景，合法99个，放置失败1个。只在合法的同一批场景中作配对对比；失败详见placement_failures.json。

未完成布置的原因：`[{'seed': 1018091, 'reason': 'Cannot build a complete nominal route: timeout'}]`。这是无障碍路线构建/几何放置失败，不是剔除带障碍策略失败案例。

| 四障碍位置 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 原位置 | 81/99 | 11 | 7 |
| 实际无障碍运动路线上 | 24/99 | 75 | 0 |

**旧版82%属于旧位置布置，不能用于本版。当前模型未重新学习四障碍避障，不能保证绕开新位置。**

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py
```

复现固定布局可在末尾加 --seed 20261103；headless检查可加 --headless。

- comparison.csv / comparison.png：配对结果；on_path.json：逐回合原因。
- placement_preview.png：第一场景的真实无障碍轨迹与四障碍几何图。
- verification.json：逐障碍相交见证与模型校验。
- manifest.json / selected_model.zip：当前演示默认配置和原初始化模型。
- 场景和每障碍位形见证：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/four_static_on_path_v1_seed61/cases.json`。
- 模型SHA256：`9fcf8ab4249cc7feb1d5257501bb33f47dfee5345fc66343344e15f9e8372257`。
