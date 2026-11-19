# 四障碍策略纠正示范（仅训练布局）

- 原64条示范不变，新增107个成功纠正episode，覆盖38个训练布局。
- 合并数据：171个episode、8406条状态动作转移；原所有数组前缀逐项完全一致。
- 全部尝试（失败不作标签）：`{'success': 107, 'no_verified_connector': 4, 'timeout': 2}`。
- 两个挖掘策略在训练布局的终止统计：`{'bc1000': {'success': 34, 'collision': 30}, 'bc5000': {'success': 36, 'collision': 26, 'timeout': 2}}`。
- 可验证的保存专家路线：64/64。实际路线或保存命令重建，采用原10毫米规划间距、0.005弧度细采样。
- 重放策略到接管点，连接到同场景原专家路线的字面后缀；不另选随机绕行路线，不更换策略观测中的原参考路径。
- 只有连接/执行前复检、实际物理安装成功及执行后复检全部通过的轨迹入库。
- 前3条新增示范含策略前缀进行物理重放，误差不超过1e-6。
- 训练布局仅64个；验证32和测试64未参与纠正采集。布局数与episode数不同，同一布局可有多个安全接管点。
- 合并数据：`/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/data/four_policy_corrections_v3_seed61/teacher.npz`。
- attempts.json / canonical_routes.json / verification.json：全部尝试、原路线与质量证明。
