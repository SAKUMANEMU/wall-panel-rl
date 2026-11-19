# Four-dynamic conservative PPO adaptation

Counts below are success/collision/timeout, with64 episodes per suite. Target dynamic speed is always0.02m/s. The original static suite is a separate64-case regression bank. Mixture rows reuse the same physically executed RRT branch and fixed coin assignments; mixed-system success is not pure PPO success.

| Added PPO steps | Training speed | Dynamic target | Paired frozen | Original static | Mixture |
|---|---:|---|---|---|---|
| 0 | 0.005 | 41/20/3 | 44/17/3 | 56/8/0 | 51/11/2 |
| 1000 | 0.005 | 42/19/3 | 44/16/4 | 57/7/0 | 51/11/2 |
| 2000 | 0.010 | 43/18/3 | 44/16/4 | 57/7/0 | 51/11/2 |
| 3000 | 0.020 | 43/18/3 | 43/17/4 | 56/7/1 | 51/11/2 |
| 4000 | 0.020 | 42/19/3 | 43/17/4 | 55/8/1 | 50/12/2 |
| 5000 | 0.020 | 39/21/4 | 44/17/3 | 51/10/3 | 50/12/2 |

Selected checkpoint: **+2000 PPO steps**. Selected weights equal the source: **False**. The default v5 source is not overwritten. This is one training seed; no general stability claim follows from it.

Training budget: 5,000 PPO transitions. Additional expert execution: 4,958; verification-replay transitions: 4,958. Total training-plus-replay physics: 14,916. Completed branch outcomes: {'rrt': {'success': 66}, 'ppo': {'success': 23, 'collision': 36, 'timeout': 8}}. Completed scene modes: {'dynamic': 92, 'static': 41}. Accepted BC updates: 5; rejected: 0. Verified expert episodes: 65.

## Independent reserved test

Source: 77/22/1; selected: 79/20/1. Paired outcomes: {'success -> success': 75, 'collision -> collision': 18, 'timeout -> timeout': 1, 'collision -> success': 4, 'success -> collision': 2}. No tuning follows this test.

The independent dynamic result is a modest77% to79% change: four source collisions become successes, while two source successes become collisions. This is not a statistical stability claim from multiple training seeds. The earlier v5 static84/100 result is a different condition and bank. The final5k candidate regresses; use the selected2k checkpoint for this experiment.

## Interpretation and evidence

Use `dynamic_vs_frozen_*.csv` to distinguish motion-specific failures from failures also present when boxes are frozen. Reference staleness during a detour is diagnostic, not proof of a cursor bug. Recovery thresholds were not changed in this experiment. `verification.json` audits source/optimizer initialization, data isolation, cached expert rows, and timestamped physical replay. `protocol.json` predeclares curriculum gates, selection, and the conditional independent test; `code_archive/` preserves execution.

See `checkpoints.csv`, `curves.png`, per-checkpoint `validation/` files, and the model directory's `episodes_worker_*.jsonl`, `training/progress.csv` and `auxiliary_bc/` states.
