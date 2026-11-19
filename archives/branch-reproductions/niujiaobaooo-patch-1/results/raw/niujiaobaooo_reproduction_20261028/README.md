# PPO branch reproduction

Branch: niujiaobaooo-patch-1; commit: 933ce7171c4009f71d395f3c32fa7b5475bb25f4

100 deterministic clean static-obstacle episodes, seeds 10000–10099, using the submitted best_model.zip and configs/eval_clean.yaml. No training.

Success: 57/100 (57%). Collision: 14/100 (14%). Joint-limit violations: 0. Remaining failures: 29, all at 160 steps.

Every seed, outcome and episode length matches the submitted CSV. Numeric differences and local dependency versions are recorded in reproduction_summary.json. Runtime is machine-dependent and excluded from equality checks.

Command (run from this branch directory):
```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/evaluate.py --config configs/eval_clean.yaml --clean-model experiment_results/hou_ppo_262144/best_model.zip --methods ppo_clean --episodes 100 --run-id YOUR_NEW_RUN_ID
```

Use a new run ID for each evaluation. This reproduces the submitted static benchmark, not the five-moving-obstacle task.
