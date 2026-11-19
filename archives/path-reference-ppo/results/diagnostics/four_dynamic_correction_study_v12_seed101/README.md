# Dynamic corrective BC study

The fixed validation target remains four moving staggered obstacles,18cm offset and0.12m/s speed. Source model, scene bank, reward and success criteria are unchanged.

Data: 32 verified corrective episodes from 32 training layouts; 507 first16-step labels, 2675 successful PPO states. Prefixes replay from the original reset, preserving nonzero time, velocity and reference state.

| Arm LR | Updates | Success | Collision | Timeout |
|---|---:|---:|---:|---:|
| 1e-06 | 0 | 37 | 26 | 1 |
| 1e-06 | 1 | 37 | 26 | 1 |
| 1e-06 | 5 | 37 | 26 | 1 |
| 1e-06 | 10 | 37 | 26 | 1 |
| 1e-06 | 25 | 36 | 27 | 1 |
| 1e-06 | 50 | 35 | 28 | 1 |
| 1e-06 | 100 | 34 | 28 | 2 |
| 1e-06 | 200 | 34 | 28 | 2 |
| 3e-06 | 0 | 37 | 26 | 1 |
| 3e-06 | 1 | 37 | 26 | 1 |
| 3e-06 | 5 | 37 | 26 | 1 |
| 3e-06 | 10 | 35 | 28 | 1 |
| 3e-06 | 25 | 35 | 28 | 1 |
| 3e-06 | 50 | 35 | 28 | 1 |
| 3e-06 | 100 | 35 | 28 | 1 |
| 3e-06 | 200 | 35 | 28 | 1 |

Selected checkpoint: {'arm': 0, 'learning_rate': 1e-06, 'bc_updates': 0, 'metrics': {'episodes': 64, 'success': 37, 'collision': 26, 'timeout': 1, 'joint_limit': 0, 'planning_failure': 0}, 'score': [1, 37, -26, -1], 'accepted_updates': 0, 'rejected_updates': 0}. PPO started: False. The predeclared gate prevents PPO after unsuccessful BC. No independent test is evaluated during BC; all labels come from training cases. Source-policy retention targets also use only successful training trajectories.
