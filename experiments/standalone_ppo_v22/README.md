# Standalone PPO v22

Objective: improve direct neural-policy installation success. Prediction and RRT
are never used to choose actions during validation, regression or final test.
The shared environment retains its original state-based reference cursor,
action projection, collision checks, 240-step limit and success tolerances.

## Changes from v21

- Initialize from the v21 selected 24,576-step checkpoint, rather than the older
  submission model. Preserve the separate v21 experiment and submission files.
- Train the existing avoidance gate, residual branch and multi-obstacle adapter.
  Keep the tracking servo and action standard deviation frozen. The initial
  deterministic actions are identical to the transferred model.
- Use a pre-clamp supervised output so oversaturated corrections can learn.
  The actual policy distribution/forward/predict retain the same clamped mean.
- Collect successful enhanced-teacher rollouts on the 100 training cases only.
  Also query a 12-step predictor on states reached by the autonomous student;
  physically execute student actions in this second stream. Safe-horizon labels
  are distinguished from complete successful demonstrations. Failed student
  episodes are journalled, and states with no safe candidate provide no label.
- Fit corrective labels, aggregate a second round of student-visited states
  (DAgger-style), and compare checkpoints using direct-policy success only.
- Run two independent 32,768-step PPO arms (learning rates 3e-6 and 1e-5), with
  fixed action standard deviation 0.05. Small KL-guarded actor-only imitation
  updates occur before each fresh on-policy rollout; expert transitions never
  enter PPO's rollout buffer. Reset PPO optimizer moments at each arm start.
- Validate every 8,192 PPO steps. Select by direct success, then joint-limit and
  collision counts and final pose cost. Enhanced results never select a model.
- Freeze 32 fresh geometry-valid test scenes before adaptation. Evaluate them
  only after checkpoint selection is frozen. Retain the old64 as regression,
  and report failures without replacement. No selection depends on test results.

## Run

Use the existing project Python environment:

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_standalone.py \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/standalone_ppo_v22_seed20261112
```

A new run requires a fresh output directory. `status.json` tracks the active
stage and batch. Checkpoints, direct-policy evaluation JSON, collection traces,
supervised loss, PPO episode CSV and auxiliary update records are saved there.
The defaults plan exactly 65,536 PPO transitions across both arms, plus supervised
updates; no training improvement is assumed in advance. The original source
remains eligible and can win if every candidate is worse. Interrupted runs keep
partial artifacts but do not silently resume.

## Checks

```bash
python -m pytest -q
python scripts/check_standalone.py
```

The second command checks one real training case, a small supervised update,
an auxiliary update before a 256-step PPO rollout, and checkpoint serialization.
It uses a separate smoke directory and does not evaluate or train on test data.

## Resume after completed collection

After an explicit resume instruction, use the saved initial/DAgger label files:

```bash
python scripts/resume_standalone.py \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/standalone_ppo_v22_seed20261112 \
  --check-only
```

Remove `--check-only` to perform the remaining 1,500 aggregate-data supervised
updates and both PPO arms. The 3,000 earlier supervised updates are not repeated.
The resume entry checks the source checkpoint hash, label arrays and split
isolation, and reuses the frozen test bank without new generation/collection.
It refuses a second launch or an already-started training stage.
