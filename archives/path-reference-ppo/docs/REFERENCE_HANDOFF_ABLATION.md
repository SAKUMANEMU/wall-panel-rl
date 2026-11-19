# Fixed-weight reference/terminal handoff ablation

Three explicitly separate controller settings on the CURRENT RETAINED PPO:

| Variant | Recovery radius (joint-space L2 rad) | Terminal radius (max joint error rad) |
|---|---:|---:|
| baseline | .30 | .20 |
| recovery_wider | .45 | .20 |
| terminal_earlier | .30 | .40 |

Only one scalar changes in each treatment. These two radii use different joint
space metrics. Changing recovery affects local reference observations after a
detour. Changing terminal radius changes BOTH endpoint-servo blending and the
existing quadratic residual taper. These are controller behavior changes, not
new learned weights or a PPO-training improvement. No combined-factor trial.

Every variant loads the same checkpoint afresh, logs a tensor fingerprint and
retains the original on-disk file hash. Goal, collision definition, 240-step
horizon, action projection, seven actions and 77 features are preserved. No online
expert route, RRT/IK fallback, optimizer update or default-configuration mutation.

Inference evaluation: original 20 target + 20 alignment validation cases, detailed
traces on 900002/900003/900005/900015, and the same two already-seen regression
banks (100 target + 20 alignment each). The combined 200 target cases are tuning/
regression evidence, not a new independent test. Baseline must reproduce the
previous 182 successes / 10 collisions / 8 timeouts on the 200-case pool.

Assess non-regression in validation and combined regression success/collisions/
timeouts, no joint-limit failures, all alignment retained and goal action <=.005.
Results identify candidates for a future prospective test; they do not silently
switch the active model/controller or authorize a training budget.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/evaluate_reference_handoff.py
# Explicit --run executes one inference-only study, refusing existing outputs.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/evaluate_reference_handoff.py --run
```

Config: `configs/reference_handoff_ablation.yaml`. Visible report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/reference_handoff_ablation_v3_seed61`.
