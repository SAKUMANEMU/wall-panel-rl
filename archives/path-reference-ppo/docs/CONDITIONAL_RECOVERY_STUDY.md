# Stagnation-triggered reference recovery, fixed retained PPO

Retain the original .30rad recovery and .20rad terminal servo. After at least 60
steps, a full 30-step window must show <1mm target-position improvement, <.1deg
orientation improvement, <=.02rad net joint displacement, unchanged non-final
reference index and reference distance >=.20rad. Real obstacle presence is
required; alignment, terminal states and dynamic tasks do not trigger recovery.

Only then temporarily allow .45rad nearest-forward-reference recovery. Immediately
restore .30rad. At most three attempts per episode, >=20 steps apart, at most one
successful extension. No new path, obstacle-aware action filter, expert fallback,
direct goal jump, action overwrite, terminal-radius change or weight update.
The seven-dimensional local reference observation changes only after an actual
forward cursor recovery. Goal-progress gating does not count progress toward a
stale reference as proof that the task is progressing.

Two fixed-weight inference arms: original StagedAvoidanceEnv versus the separate
ConditionalRecoveryEnv. Same frozen model, 240-step horizon and collision/success
criteria. Original validation 20 targets + 20 alignment, then two known regression
banks (200 targets + 40 alignment combined). Trace four known failures. Require
every untriggered episode to reproduce original reason, length and final error.
Non-regression requires success not lower, collisions/timeouts not higher in both
original validation and pooled regression, no limits and full alignment retention.

If this predeclared gate passes, freeze the rule and score original/conditional on
100 new scenes from seed 1004000 onward plus 20 alignment cases. No parameter
search using that test. If the gate fails, do not generate extra tests. This
prospective test has a strict maximum of 100 targets. No PPO/BC training or default
controller adoption occurs automatically, regardless of results.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/evaluate_conditional_recovery.py
# Explicit inference-only run; refuses existing report/data output:
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/evaluate_conditional_recovery.py --run
```

Config: `configs/conditional_recovery_study.yaml`. Visible report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_recovery_v3_seed61`.
