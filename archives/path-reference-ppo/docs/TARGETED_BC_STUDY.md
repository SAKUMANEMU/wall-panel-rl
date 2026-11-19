# Targeted BC continuation study

Continue the selected 50% BC actor from `bc_budget_v3_seed61/same_data_10k`,
not the unrelated downloaded PPO branch. The checkpoint and its separate BC
Adam/RNG state at update 10000 are both restored. PPO counters remain unchanged.

1. Reproduce the old 20-scene target result and diagnose contacts, joint-space
   reference deviation and clearance across full policy trajectories.
2. Diagnose only the original training bank minus the original supervised held-out
   seeds for correction collection. Generate and freeze 100 independent target
   scenes from seeds 980000 onward plus 20 alignment retention cases.
3. Attempt at most 400 offline expert recoveries at trajectory-dependent handoffs.
   Collect 100 successful corrective episodes. Prefixes are physically replayed,
   including velocities and elapsed time, and are never expert action labels.
   Failed expert attempts remain logged but contribute no successful labels.
4. Replay 10 spread-out corrections before fitting. Merge successful expert suffixes
   with the original 100 demonstrations for the treatment arm. Held-out scene
   membership stays fixed; no corrective labels come from held-out/test layouts.
5. Run two separate 10000-update continuations, `original_only` and `targeted_mix`,
   from identical weights, BC Adam moments and RNG. Keep the existing batch size,
   learning rate and loss. Half of each batch is alignment, half avoidance;
   episodes within a task are sampled uniformly. Adding corrections changes the
   within-avoidance episode distribution and is part of this treatment.
6. Evaluate at 0/2000/4000/6000/8000/10000 additional BC updates using the original
   validation set. The initial model is an eligible checkpoint, so regressions
   cannot silently replace it. Success/collision/retention select checkpoints,
   with held-out loss as the final tie-breaker. BC updates and PPO environment
   transitions are recorded separately.
7. Lock both selections, then score baseline and selected models on the frozen
   independent set once. Independent results do not select the winner. No PPO,
   threshold relaxation, automatic stage promotion or budget extension occurs.

The independent set is unused at the start of this study. Once results are seen,
future tuning needs a new test set for another independent final claim.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_targeted_bc.py
# Explicit execution, refuses existing study output directories:
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/run_targeted_bc.py --run
```

Configuration: `configs/targeted_bc_study.yaml`. Visible report directory:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/targeted_bc_v3_seed61`.
