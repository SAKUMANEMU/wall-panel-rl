# Dynamic adaptation v21

Isolated development snapshot of the Learning Agent C&R code. The submitted
folder, archive and selected source checkpoint are kept as frozen inputs.

## Study

1. Generate 100 training and 16 new selection-validation scenes, in seed order.
   All scenes have four dynamic obstacles and the current larger height stagger.
   Existing development/validation/demo seeds are excluded. Geometry rejections
   are logged; final rollout success is not a scene-selection criterion.
2. Collect one RRT attempt per training scene. If it fails, run the original
   enhanced controller as a second, separately logged teacher attempt. Only
   complete physical successes provide BC labels, using the original reference
   observations and the environment's executed seven-joint actions.
3. Validate the unchanged source, then train only the obstacle adapter for 2,000
   BC updates. Sample successful scenes uniformly and retain valid waiting labels.
4. Validate BC, then perform 32,768 additional on-policy PPO steps in the same
   dynamic task; validate every 8,192 steps. A fresh PPO optimizer avoids carrying
   obsolete Adam moments across supervised training. The actor prior stays frozen.
5. Compare earlier local recovery on the selected checkpoint using the same new
   validation bank. This is a separate execution-controller ablation.
6. Report the selected model on the old 64 scenes as regression data, not as a
   new independent final test. The source remains a selection candidate.

Selection prioritizes enhanced collision/joint-limit count, then enhanced success,
direct PPO success, direct failures and waiting. All outcomes remain in the
validation denominator. No target tolerance, reward or episode budget is relaxed.
If all initial teacher attempts fail, or fewer than 10 training scenes succeed,
the pipeline stops with an explicit diagnostic instead of training on empty data.

## Run

Use the existing project environment, with one thread per worker:

```bash
env -u PYTHONPATH OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/run_adaptation.py \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/dynamic_adaptation_v21_seed20261112
```

Choose a fresh output folder for a new run. The protocol, atomic `status.json`,
case banks, generation attempts, teacher attempts/diagnostics, BC data/loss,
PPO episode log, checkpoints and separate evaluations are saved there. A running
experiment must not be edited. At this version, interrupted runs retain partial
data but require an explicitly implemented resume/restart rather than silently
reusing partial results.

## Checks

```bash
python -m pytest -q
python scripts/check_adaptation.py
```

The interface check uses a development case to execute RRT, update BC, then
perform a 256-step PPO update in a separate smoke directory. It is not a training
or evaluation success-rate experiment. Formal data are generated separately.
