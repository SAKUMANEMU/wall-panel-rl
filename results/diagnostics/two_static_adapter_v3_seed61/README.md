# 双静态障碍：独立修正网络、针对性RRT示范与短预算适配

**本轮未放行，不替换现有模型。新100场景：初始化87%成功/8次碰撞，学习后候选76%成功/14次碰撞。固定验证中的改进未推广到新场景。**

原单障碍动作网络被冻结；独立修正网络只在至少两个障碍存在时激活。没有在线避障规划器参与纯PPO评估。
初始化时新增槽位输入列归零、修正网络输出归零。训练保留障碍的固定顺序，不再交换原槽位含义。

## 数据与接口

- 原示范256条；新增针对性RRT成功示范37条；合计14,162个监督状态。
- 审计54个训练布局，其中29个布局的第二障碍挡住仅考虑第一障碍的RRT路线。
- 新采集共37次尝试，失败数为{'collision': 0, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}；失败不作为标签，交接前策略动作不作为专家标签。
- 只从训练布局采集；10个示范留出布局不进入梯度；固定20个验证布局用于选模；新100场景预先冻结，最后评估。
- 新示范采样权重为原示范4倍，双障/单障/对准批次比例80%/10%/10%。

## BC验证

| 更新 | 成功 | 碰撞 | 超时 | 单障成功 | 单障碰撞 |
|---:|---:|---:|---:|---:|---:|
| 0 | 15/20 | 4 | 1 | 18/20 | 0 |
| 100 | 8/20 | 10 | 2 | 18/20 | 0 |
| 250 | 15/20 | 2 | 3 | 18/20 | 0 |
| 500 | 12/20 | 8 | 0 | 18/20 | 0 |
| 1,000 | 15/20 | 5 | 0 | 18/20 | 0 |
| 2,000 | 12/20 | 8 | 0 | 18/20 | 0 |
| 3,000 | 14/20 | 6 | 0 | 18/20 | 0 |

执行3,000次BC更新；所选更新250；学习改进门槛通过：**True**。
启动PPO须同时满足：有非零更新、对第二障碍有动作响应、固定验证较初始化有成功/碰撞改进、保留单障和对准能力。

## PPO适配

启动：**True**；实际追加PPO步数：10,000；所选追加步数：0。
训练按回合50%RRT/50%PPO，RRT标签与PPO在策略采样缓冲区分开；辅助BC只更新独立修正网络。
额外RRT物理步数：6,563；训练完成回合按控制器统计：`{'ppo': {'success': 55, 'collision': 51, 'timeout': 18, 'joint_limit': 0, 'planning_failure': 0}, 'rrt': {'success': 131, 'collision': 0, 'timeout': 0, 'joint_limit': 0, 'planning_failure': 0}}`。

| PPO追加步 | 成功 | 碰撞 | 超时 |
|---:|---:|---:|---:|
| 0 | 15/20 | 2 | 3 |
| 5,000 | 14/20 | 3 | 3 |
| 10,000 | 14/20 | 4 | 2 |

## 最终评估

| 数据 | 模型 | 成功 | 碰撞 | 超时 |
|---|---|---:|---:|---:|
| 双障新100 | 初始化 | 87/100 | 8 | 5 |
| 双障新100 | 候选 | 76/100 | 14 | 10 |
| 单障回归100 | 原模型 | 95/100 | 2 | 3 |
| 单障回归100 | 候选 | 95/100 | 2 | 3 |

第二障碍移动15厘米时的动作响应：`{'mean_action_change_15cm': 0.004142080899327993, 'max_action_change_15cm': 0.00892561674118042, 'adapter_mean_abs_correction': 0.03967202082276344}`。
原单障完整轨迹保持一致：**True**。
放行检查：`{'bc_learning_gate_passed': True, 'selected_bc_updates': 250, 'selected_ppo_additional_steps': 0, 'selected_candidate_validation_eligible': False, 'new_test_gate_passed': False, 'ppo_stability_passed': False, 'learned_multi_obstacle_actions': True, 'default_model_preserved': True, 'automatic_dynamic_stage': False, 'new_test_not_worse_than_initialization': False, 'two_task_ready': False}`。原默认单障模型保留，没有自动进入动态阶段。

第二障碍敏感性只证明使用了输入，不等于避障保证。新100场景为本轮独立测试，旧100单障场景仅作回归。
纯PPO结果与selected_mixture_validation.json中的50/50混合结果分开，混合结果包含在线RRT贡献。

## 文件位置

- 原始数据与示范：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/two_static_adapter_v3_seed61`
- 模型与优化器：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/models/two_static_adapter_v3_seed61`
- comparison.csv、overview.png：各阶段数据与曲线。
- interface_diagnosis.json：训练前真实输入和梯度检查。
- bc_validation/、ppo_run/：各检查点逐回合评估；two_test_after.json：新测试的全部结果。
- teacher_replay.json：新专家示范重放核验；verification.json：参数与数据隔离核验。
- candidate_model.zip + candidate_manifest.json：匹配的候选与环境配置。
- training_accounting.json：PPO与RRT完成回合及分任务计数，未结束的PPO回合单独保留在步数中。
- 日志：`/home/johnsonchen/wall-panel-rl/logs/two_static_adapter_v3_seed61.log`。
