# 新四障碍 RRT 示范数据（未训练）

- 独立场景：训练64、验证32、测试64，种子互斥且不含旧99例。
- 保存64个成功episode、3973条状态动作转移；每个训练布局最多一条路线。
- 全部尝试结果（包含重试）：`{'success': 64}`。这不是PPO成功率，也不是独立测试成绩。
- 未解决训练种子：`[]`；数据完整性标志ready_for_bc=True。
- 标签同时满足：执行前路线复检通过、真实物理执行成功、执行后路线复检通过。失败尝试不加入标签。
- 新示范使用10毫米规划间距、0.005弧度采样；原99例回归仍为5毫米。物理碰撞阈值、执行器、奖励未变。
- 有限采样不等于连续安全证明；规划时仿真时钟暂停。
- 验证和测试只生成并冻结场景，没有执行含障碍专家episode，也没有用于示范筛选或训练。
- 场景沿冻结初始化策略的无障碍运动布置；生成时的无障碍轨迹仅用于几何布置。生成拒绝见generation_rejections.json。
- 生成阶段另有3个候选因无障碍名义轨迹未完整结束而拒绝，已保留原因；这些不属于含障碍RRT执行的失败尝试。
- 策略参考路径保持原值。独立保存raw_actions和执行后的actions，观测77维、动作7维；前三个episode物理重放误差均不超过1e-6。
- 本轮BC/PPO优化步数为0，初始化权重和默认演示策略均未替换。

## 文件

- 数据根目录：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/four_safe_teacher_v1_seed61`
- train_cases.json / validation_cases.json / test_cases.json：冻结场景。
- teacher/teacher.npz：obs、actions、raw_actions、offsets、seeds、tasks、terminal_obs、terminal_actions。
- teacher/attempts.json / metadata.json：全部尝试、接收规则和数据哈希。
- demonstrations.csv / overview.png：逐episode数据概况，不是学习曲线。
- protocol.json / code_archive/ / verification.json：配置、代码、划分和质量验证。

后续接入BC前需明确四障碍任务分组和旧技能保留比例；此独立数据集不能直接套用旧混合任务分组。

中断续采：原52个episode、3222条转移的所有数组前缀逐项完全一致，原文件备份在interrupted_checkpoint/。
续采规则与原配置相同；新增段前三个episode另外进行了物理重放。详见resume_protocol.json及resumed_segment_replay.json。
