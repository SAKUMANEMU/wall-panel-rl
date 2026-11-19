# 扩大失败场景覆盖与RRT路线一致性实验

**本轮未采纳学习更新。候选仍为初始化模型，不能将其测试成绩归因于新训练。**

新100场景配对结果：初始化87次成功、6次碰撞、7次超时；候选87次成功、6次碰撞、7次超时。

## 数据和专家路线

- 新增128个训练布局，与原54个非留出训练布局合并为182个。新64个布局用于选模；另100个新布局只做最终测试。
- 原20例双障验证作为回归保护；单障验证和回归均保留。以前的测试场景未混入这轮新增训练/验证/测试。
- 审计182个训练布局，初始化策略或上一轮失败候选在其中47个失败。
- 规划器报告成功的候选共546条；166个布局出现相同初始状态、不同RRT种子下的初始动作方向冲突。
- 恢复后提前边复检拒绝7条危险候选；实际采纳的完整/纠正示范均经过边检查与真实执行验证。
- 每布局最多比较3条RRT路线，用关节路径长度、相对无障碍参考的偏离和初始动作差异的固定代价选一条；不混合不同路线标签。
- 纠正示范从真实策略失败前交接，通过碰撞检查的直线连接接入同一保存路线的尾部。无安全连接时记录失败，不生成标签。
- 成功保存182条完整路线示范和99条纠正示范，后者覆盖44个布局。
- 删除旧双障多路线标签，保留单障/对准锚点与原10布局示范留出集。最终419条示范、17,805个监督状态。
- 失败前的策略动作仅用于重放，不作为专家标签；原无障碍参考始终不替换为专家路径。
- 路线一致性限定在同一布局内部，并未证明不同相近布局具有相同绕行拓扑；方向冲突也不是所有失败的已证实唯一原因。

## BC短预算验证

| BC更新 | 新验证成功 | 碰撞 | 超时 | 原验证成功 | 碰撞 | 超时 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 58/64 | 3 | 3 | 15/20 | 4 | 1 |
| 100 | 1/64 | 63 | 0 | 1/20 | 19 | 0 |
| 250 | 1/64 | 62 | 1 | 0/20 | 20 | 0 |
| 500 | 3/64 | 61 | 0 | 4/20 | 16 | 0 |
| 1,000 | 2/64 | 62 | 0 | 2/20 | 18 | 0 |
| 2,000 | 8/64 | 56 | 0 | 2/20 | 18 | 0 |
| 3,000 | 21/64 | 43 | 0 | 6/20 | 14 | 0 |

执行3,000次BC更新；选择0次；PPO学习启动门槛通过：False。
选择模型要求新64例与原20例均不降低成功数、不增加碰撞；满足最低能力后优先减少碰撞。单障与对准能力必须保留。
PPO只在非零BC更新对新验证集带来改进、对原验证不退化且响应第二障碍时启动。

## PPO适配

是否启动：**False**；实际PPO追加步数：0；所选追加步数：None。
每回合50%RRT/50%PPO；PPO只接收自身策略采样的转换。辅助BC监督池仅使用冻结的一致路线训练标签，运行中新生成的RRT路线不加入池。

## 最终测试与单障回归

| 数据/模型 | 成功 | 碰撞 | 超时 |
|---|---:|---:|---:|
| 新100/初始化 | 87/100 | 6 | 7 |
| 新100/候选 | 87/100 | 6 | 7 |
| 单障100/原模型 | 95/100 | 2 | 3 |
| 单障100/候选 | 95/100 | 2 | 3 |

第二障碍15厘米位置扰动的动作敏感性：`{'mean_action_change_15cm': 0.0, 'max_action_change_15cm': 0.0, 'adapter_mean_abs_correction': 0.0}`；敏感性不等于避障保证。
完整放行检查：`{'selected_bc_update': 0, 'selected_ppo_additional_steps': None, 'bc_learning_gate_passed': False, 'candidate_validation_eligible': False, 'new_test_gate_passed': True, 'ppo_stability_passed': False, 'new_test_not_worse_than_initialization': True, 'default_model_preserved': True, 'automatic_dynamic_stage': False, 'two_task_ready': False}`。
- 原单障完整轨迹与冻结动作参数保持一致。新100场景只在候选锁定后评估，不用于选模。
- 本轮新测试在报告生成后已成为已见数据，后续不能再称为独立新测试。

## 结果文件

- 原始数据和路线审计：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/consistent_two_static_v4_seed61`
- 模型/优化器：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/consistent_two_static_v4_seed61`
- comparison.csv / overview.png：全部检查点计数和曲线。
- bc_validation/ / ppo_run/：各检查点逐场景结果；test_candidate.json：新100例全部终止原因。
- teacher_replay.json：真实专家标签物理重放；verification.json：数据、代码、选路线和冻结参数核验。
- candidate_model.zip + candidate_manifest.json：候选及匹配环境；assessment.json：放行状态。
- 当前恢复运行日志：`/home/johnsonchen/wall-panel-rl/logs/consistent_two_static_v4_seed61_resume1.log`。
- 初次中断的代码、日志和80布局数据检查点保留在interrupted_collection_1，未重新生成冻结数据。
