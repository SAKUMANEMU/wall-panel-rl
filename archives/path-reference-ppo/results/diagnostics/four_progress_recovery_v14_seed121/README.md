# Reducing waits with progress-aware local recovery

This remains a model-based execution-controller improvement, not a new PPO model. The previous12-step predictive safety controller is kept. After step100, low progress or predicted budget exhaustion can trigger recovery. A model simulation first protects continuations which would succeed with the old controller. For other cases, collision-free goal-directed joint actions or brief single-joint biases are compared locally. Actual reference paths are never overwritten and no global RRT is used in this branch.

Scene: four unequal moving obstacles,18cm offset,0.12m/s speed, alternating3-7cm anchor offsets. Robot/panel geometry, success tolerances and240 real steps remain unchanged. All trial states and virtual time are restored before real execution.

Counts are success/collision/timeout/planning-failure.

| Suite | Previous safety | Progress recovery |
|---|---|---|
| Validation | 56/0/8/0 | 62/0/2/0 |
| Validation50/50 RRT mixture | 57/0/5/2 | 61/0/1/2 |
| New independent test | 76/3/21/0 | 96/1/3/0 |

Validation paired changes: {'success -> success': 56, 'timeout -> success': 6, 'timeout -> timeout': 2}. New-test changes: {'success -> success': 76, 'timeout -> success': 18, 'collision -> collision': 1, 'collision -> success': 2, 'timeout -> timeout': 3}. Old successes preserve outcome, length and final errors with no recovery action. The previous100-case v13 bank is used only for historical development diagnosis; this test bank starts at1144000 and has no overlapping scene ID. Controller parameters are frozen before the new test.

Computational tradeoff: {'baseline_mean_wall_s': 9.454646045620011, 'recovery_mean_wall_s': 14.152297443040078, 'baseline_zero_waits': 2876, 'recovery_zero_waits': 1599, 'local_recovery_actions': 559, 'forecast_cache_hits': 7}. These wall times include concurrent CPU work; predictive computation pauses simulation time. Recovery reduces simulated waiting/timeout but adds model-prediction work. There is no real-time guarantee.

See `paired_validation.csv`, `paired_independent_test.csv`, `verification.json`, `comparison.png`, `validation_gate.json` and `independent_test.json`. Earlier pure-PPO and predictive-only demos remain available. New explicit demo: `scripts/demo_progress_dynamic.py`, default50/50 RRT and predictive-PPO-progress branches; use `--controller ppo_progress` for this branch alone.
