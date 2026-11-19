# 本地复现与 PPO 训练接口检查

基于提交 3c0cd19，在 codex/path-reference-ppo 工作副本运行。未训练，未更新权重。

| 方法 | 静态成功 | 碰撞 | 越界 | 规划失败 | 无障碍对准 | 成功静态回合平均步数 |
|---|---:|---:|---:|---:|---:|---:|
| ordinary | 89/90 | 0 | 0 | 1 | 20/20 | 60.0225 |
| bc | 89/90 | 0 | 0 | 1 | 20/20 | 60.0225 |

110 个固定场景，每场规划一次，普通跟踪与 BC 共享路径、物理初态及噪声流，共 220 回合。seed 121019 的规划失败已复现。结果与发布报告一致，但不是独立泛化测试。全部场景使用静态障碍物和零观测噪声。

## 接口核查

- 输入 77 维、输出 7 维，模型可加载，推理及 evaluate_actions 输出有限值。
- 教师数据 100 条轨迹、5945 个状态动作对；重新计算 MSE=1.36879e-13，与发布值一致。
- actor 实际使用 q、dq 和下一关节路点。修改第 14–69 维的目标/障碍物特征不改变 actor 输出；critic 可使用全部特征。当前学习任务是伺服跟踪，不是端到端规划。
- BC 训练时仅 14 个增益/偏置系数参与更新；加载 ZIP 后，21 个伺服系数（包括速度项）以及 critic 默认可训练。requires_grad 的临时冻结设置未随普通 state_dict 自动恢复，需要在新入口显式设置。
- forward() 的动作没有 predict() 的 0.01 rad 限位裕度处理。构造合法近限位状态，确定性动作最大差约 0.14324。PPO 使用 forward 收集动作，因此新训练环境必须统一执行动作变换，并保持原始采样动作/log_prob 的正确对应，不能仅替换动作后沿用错误概率。
- 已加载模型的学习率为 0.003，batch_size=50，n_epochs=10，target_kl=None；并非发布报告中 PPO 短试验的 1e-5、100、3、0.005。微调入口需显式覆盖配置并记录实际生效值。
- 发布包提供离线 BC 的 OfflineSpaces，step 会主动报错；未包含可运行的 77 维 PPO Gym 包装器或历史 PPO 试验训练脚本。现有 70 维混合课程入口不能直接用于此模型。
- 路点进展奖励和游标更新次序仅在历史试验说明中描述；缺少对应训练代码，无法核实历史实现。新包装器应以同一个路点计算动作前后进展，再更新游标。
- 历史 PPO 试验的验证时点无法从已发布训练源码确认。新入口需要参数更新后验证、最终保存模型重载验证，以及同场景同路径的普通跟踪/BC 对照。

## 后续接入顺序

1. 增加静态 77 维 PPO 环境包装器，包含实时规划、路点观测、动作限幅及规划失败终止。
2. 明确 14 或 21 个 actor 系数的训练范围，启用 critic 学习并保持原 BC 权重副本。
3. 设置保守微调参数，加入更新后验证、成功/碰撞/超时误差/对准保留指标。
4. 接口测试通过后，再授权短预算训练；不自动启动长训练或独立 100 场景测试。
5. 静态微调验证后，再单独设计动态障碍物接入；本次静态结果不能证明动态安全。

## 验证与复现命令

现有测试：28 passed in 47.73s。接口审计仅做前向计算，无 backward、optimizer.step 或 learn；模型 SHA256 保持不变。

从工作副本根目录执行：
```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/audit_waypoint_interface.py
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python releases/path_reference_bc_20261025/evaluate.py --output /tmp/path_reference_recheck.json
```

数据文件：reproduction.json、interface_audit.json、tests.log。评估输出需使用不存在的路径。
