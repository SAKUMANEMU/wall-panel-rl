# 初始化模型：四个静态障碍物

**已切换当前默认演示至四个静态障碍，使用保留的初始化权重。没有BC/PPO再训练。**

| 障碍数 | 成功 | 碰撞 | 超时 |
|---:|---:|---:|---:|
| 2 | 87/100 | 6 | 7 |
| 4 | 82/100 | 11 | 7 |

逐场景变化：`{'success -> success': 82, 'collision -> collision': 6, 'success -> collision': 5, 'timeout -> timeout': 7}`。复用已见的v4底层100场景作为障碍数量变化的配对回归，不是新独立测试。

## 场景与模型

- 四个障碍均为静态，大小不一。保留原双障槽位顺序，再增加前段与后段障碍，原索引顺序[2,3,1,4]。
- 观测仍为77维、动作7维；保留5个障碍信息槽位，最后一个槽位为空，以兼容初始化模型。
- 初始位置、障碍位置默认每次随机生成；--seed固定场景。只因几何放置无效而重试，不按策略能否成功筛选。
- 目标位置、机器人、墙板尺寸、奖励和240步上限沿用源配置；播放速度0.6倍，无感知噪声。
- 新场景的无障碍参考在播放前准备；控制执行期间纯策略没有在线避障规划器。固定场景库则直接复用已有参考。
- 初始化权重对新增障碍槽位的连接及独立修正输出仍为零，因此该模型尚未学会主动利用其余三个障碍的信息。82%仅是当前场景下的表现，不是四障碍训练成果。

原模型：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/consistent_two_static_v4_seed61/initialized_model.zip`；SHA256：`9fcf8ab4249cc7feb1d5257501bb33f47dfee5345fc66343344e15f9e8372257`。复制模型与原模型逐字节一致。

## 终端演示

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py
```

指定种子复现：

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py --seed 20261103
```

原demo_autonomous_ppo.py入口也默认使用本配置。需要历史双障演示可运行demo_consistent_two_static.py；需复用固定四障场景可加：

```bash
--scene-bank /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/four_static_initial_v1_seed61/four_cases.json --seed 1018000
```

## 文件

- comparison.csv / overview.png：配对计数与图；four_static.json：逐回合结果。
- manifest.json + selected_model.zip：四障环境配置与初始化模型。
- placement_audit.json / verification.json：全部四障实体、初末状态及模型来源核验。
