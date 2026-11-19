# Online-BC ablation at a matched 5,000 PPO transitions

Only the number of online auxiliary BC updates per rollout changes (1 to 0). Both experiments start from the original PPOv5 weights, optimizer and value network, with the same seed 61, training bank, hard-case sampling, PPO hyperparameters, reward scale, and 50% RRT / 50% PPO episode routing. The no-BC run still executes and verifies RRT episodes; it does not learn from their labels.

| Experiment | Added PPO steps | Evaluation | Success / collision / timeout |
|---|---:|---|---|
| BC_on | 0 | ppo | 56 / 8 / 0 |
| BC_on | 0 | mixture | 60 / 4 / 0 |
| BC_on | 5000 | ppo | 55 / 8 / 1 |
| BC_on | 5000 | mixture | 60 / 4 / 0 |
| BC_off | 0 | ppo | 56 / 8 / 0 |
| BC_off | 0 | mixture | 60 / 4 / 0 |
| BC_off | 5000 | ppo | 50 / 11 / 3 |
| BC_off | 5000 | mixture | 55 / 7 / 2 |

At 5k, disabling online BC changes pure-PPO success by -5/64, collisions by +3/64, and timeouts by +2/64 versus BC-on. This is a single-seed validation ablation. It does not establish a general causal conclusion; the original 10k endpoint is not a budget-matched comparison.

Both runs have exactly the same 5,438 verified expert observation/action pairs at 5k (array equality), with 5 versus 0 accepted BC updates by that checkpoint. This run does not support online BC as the sole cause of regression: removing it did not improve the endpoint. Further work should examine PPO drift and the sensitivity of reference-cursor advancement while preserving the original source as a control.

The no-BC run selects +0 steps using full pure-PPO validation. Selected policy equals the source: **True**. This diagnostic run does not promote a new default model. The current independent-test result remains the earlier PPOv5 84/100; no fresh independent test is run here.

Training: 5,000 PPO and 5,438 RRT transitions, 10,438 total. Episode starts: {'rrt': 82, 'ppo': 83}. Completed outcomes: {'rrt': {'success': 82}, 'ppo': {'success': 30, 'collision': 44, 'timeout': 5}}. BC accepted/rejected updates: 0/0; saved verified expert pairs: 5438.

`verification.json` records protocol, source, archive and data-isolation checks. `checkpoints.csv`, `curves.png`, `paired_5k.json` and `paired_cases.csv` give the comparison. The detailed prior-regression diagnosis is in `../four_mix_regression_v7_seed61/README.md`; experiment-specific checkpoints and episode journals remain in their original model/report directories.
