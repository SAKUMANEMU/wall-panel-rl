# Large-motion PPO adaptation

All counts are success/collision/timeout/planning-failure. Each validation suite has 64 episodes. The target is fixed at 15 cm Euclidean anchor offset and 0.10 m/s Euclidean speed bounds; realized per-axis values are smaller. Curriculum changes only on new episodes.

| Added PPO steps | Training amplitude/speed | Target PPO | Frozen | Original static | RRT/PPO mixture |
|---:|---|---|---|---|---|
| 0 | 0.100m / 0.050m/s | 54/10/0/0 | 52/11/1/0 | 57/7/0/0 | 56/6/1/1 |
| 1000 | 0.100m / 0.050m/s | 54/10/0/0 | 52/11/1/0 | 57/7/0/0 | 56/6/1/1 |
| 2000 | 0.125m / 0.075m/s | 55/9/0/0 | 53/10/1/0 | 57/7/0/0 | 56/6/1/1 |
| 3000 | 0.150m / 0.100m/s | 55/9/0/0 | 53/10/1/0 | 57/7/0/0 | 56/6/1/1 |
| 4000 | 0.150m / 0.100m/s | 55/9/0/0 | 53/10/1/0 | 57/7/0/0 | 56/6/1/1 |
| 5000 | 0.150m / 0.100m/s | 55/9/0/0 | 53/10/1/0 | 57/7/0/0 | 56/6/1/1 |

Selected: +2000 PPO steps. Selected weights equal the source: False. Source and previous demo defaults are preserved. This is one training seed.

Validation paired outcomes: {'success -> success': 54, 'collision -> collision': 9, 'collision -> success': 1}. Rescued scene seeds: [1102062]; lost prior successes: [].

Independent paired test: source 76/19/5/0; selected 76/20/4/0; outcomes {'collision -> collision': 19, 'success -> success': 75, 'timeout -> timeout': 4, 'timeout -> success': 1, 'success -> collision': 1}. Selection was locked before this test; no subsequent tuning.

This experiment does not demonstrate an independent-test success-rate improvement. In this run one timeout becomes a success, while one previous success becomes a collision. Keep the source as the default; the validation-selected checkpoint remains an experimental result.

Training PPO transitions: 5000; additional RRT execution: 4771; verification replay: 4291. Completed branch outcomes: {'rrt': {'success': 58, 'timeout': 2, 'planning_failure': 3}, 'ppo': {'success': 34, 'collision': 20, 'timeout': 6}}. Scene modes: {'dynamic': 113, 'static': 10}. New accepted BC updates: 5.

Banks are fixed before training, with geometric/nominal-reference construction rejection logs. On-path anchors may be resampled within the previous path intervals to satisfy the full motion envelope. No outcome with obstacles is used to choose cases. This changes the scene distribution relative to earlier demo preflights, so those small preflight rates are not the baseline here. Evaluation labels never enter BC or PPO. RRT teachers enter auxiliary BC only after time checks and independent physical replay; their actions never enter the PPO rollout buffer.

See `protocol.json`, `verification.json`, `checkpoints.csv`, `curves.png` and `validation/`.
