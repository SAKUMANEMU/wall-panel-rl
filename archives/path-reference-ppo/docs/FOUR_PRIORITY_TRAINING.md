# Four-obstacle priority training (2026-11-04)

The user prioritizes four-obstacle success and relaxes earlier two-obstacle
retention. Remove two-obstacle examples from training and two-obstacle scores
from selection. Keep collision detection, installation precision, actuator,
reference observation and240-step horizon unchanged. All scenes are static,
with four differently sized boxes along the nominal motion and no sensor noise.

Start from the earlier BC1,000 candidate, not the BC25 retention-selected model.
Its four-obstacle validation score is17 success/15 collision on32 scenes.
Run three separate branches from those same source weights:

1. Four-only BC4,000 additional updates: no improvement over17/32. Preserve
   the complete history rather than replacing the source.
2. Mine only the64 TRAIN layouts with BC1,000 and BC5,000 policies. Recover each
   layout's original expert route, verify at10mm clearance and0.005rad samples,
   replay policy prefixes, and collect safe connections to literal route tails.
   Of113 attempts,107 succeed,4 lack a verified connector and2 time out. Accept
   successful pre/post-verified physical executions only. These107 episodes
   cover38 layouts; original64 episodes are byte-for-byte array prefixes in the
   merged171-episode/8,406-transition dataset. Three physical replays have error0.
   Four-only BC with takeover-state oversampling still regresses; keep BC1,000.
3. Pure PPO10,000 transitions from BC1,000, with no expert-controller episodes
   or auxiliary BC updates. Four vector environments sample only the64 TRAIN
   layouts. PPO updates multi_adapter and critic; original base actor/log_std
   stay frozen. Scale TRAIN rewards by0.01 and reset the value output for those
   units. Evaluate using original rewards and unchanged physical verdicts.

PPO LR1e-5, rollout250 per worker, batch100,3 epochs, clip0.05, target KL0.005,
value coefficient0.1. Checkpoints at0/2k/4k/6k/8k/10k use the same32 validation
scenes. Rank four success first, with collisions no higher than the source and
no joint-limit violations; ties favor fewer collisions/timeouts, then earlier
checkpoints. Two-obstacle performance cannot reject the model.

Select PPO10,000 before reading the reserved test outcomes. Freeze model hash
in selection_before_test.json, then compare both fixed models on64 reserved
scenes. No checkpoint reselection occurs afterward.

| Four-static policy | Validation success/32 | Test success/64 | Test collisions | Test timeouts |
|---|---:|---:|---:|---:|
| BC1,000 source |17|37|27|0|
| BC + PPO10,000 |22|51|13|0|

Test success rises57.8% to79.7%, and every previously successful test stays
successful.14 previous collisions become successes. These are deterministic
policy results without sensing noise. Stochastic training episodes differ:
37 success/132 collision/14 timeout,183 completed episodes and9,662 completed
transitions. The remaining338 transitions belong to unfinished episodes. This
exposes sensitivity to sampled action noise; do not equate stochastic training
success with deterministic testing. The64-scene test is now used, so future
hyperparameter tuning requires a fresh bank for another independent test.

Code/config entry points:

- configs/four_priority_bc.yaml and scripts/run_four_priority_bc.py: first continuation.
- scripts/collect_four_policy_corrections.py: train-only safe correction labels.
- configs/four_corrective_bc.yaml: correction-data BC continuation.
- configs/four_priority_ppo.yaml and scripts/train_four_priority_ppo.py: pure PPO.
- src/wall_panel_rl/rl/four_priority_ppo.py: frozen training-bank wrapper.
- scripts/demo_four_trained.py: explicit selected-PPO animation.

Show the current trained model (random start/layout,0.6x playback):

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_trained.py
```

The earlier demo_four_static.py default remains the original initialization.
The dedicated trained entry uses candidate_manifest.json and selected_model.zip
from results/diagnostics/four_priority_ppo_v4_seed61. Existing --seed,--scene-bank
and --headless options remain available. Reports/curves are mirrored into the
main visible results/diagnostics folders; no old report/model is overwritten.

Full178 tests pass. The explicit trained demo is checked on validation seed
1022018 (success in62 steps) and a fresh-random seed422877867 (success in74 steps),
with four static blockers, no online avoidance planner, and the same wrapper as
evaluation. Saved demo evidence and full test log are in the final PPO report.
