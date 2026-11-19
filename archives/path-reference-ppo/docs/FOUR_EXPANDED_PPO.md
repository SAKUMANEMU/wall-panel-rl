# Expanded four-static PPO continuation

Diagnose the13 remaining collisions of the PPO10k policy. All reproduce the
original reason and control-step count. All involve the panel:11 hit slot4 near
the80% nominal-motion segment and2 hit slot3 near20%; control steps7–50. These
previously tested layouts remain diagnostic/regression only, not new training.

Freeze192 train layouts (64 original+128 fresh),64 fresh validation and100 fresh
test. Preserve the original32 validation cases and64 used test cases separately.
All roles have disjoint scene IDs. Fresh seeds start1030000/1032000/1034000;
one validation generation candidate times out in the obstacle-free nominal route
and is recorded, with no with-obstacle success filtering. Four blocker sizes,
gap/endpoint checks, target, sensing and actuator settings remain unchanged.

Scan source performance only on the192 train layouts:57 fail. Set training
sampling to75% uniform over all192 and25% uniform over these57. This never samples
old/new validation/test layouts. Continue PPO20,000 transitions with4 workers,
rollout250,batch100,3 epochs,LR1e-5,clip0.05,KL0.005,vf0.1. Keep reward scale0.01.
Load original Adam moments, value network and weights exactly; verify a stable
optimizer digest and tensor equality. Neither critic nor optimizer is reset.
Freeze base actor/log_std; update only multi_adapter and critic. No BC or online
RRT-controller transitions enter PPO.

Select every5k by fresh64 validation success, with collisions no worse than its
baseline and no joint-limit events. Old32 validation is a tie-breaker only;
old64 used-test performance cannot select the checkpoint. Freeze model hash
before final tests. Test source and candidate on the same fresh100 scenes, then
evaluate the candidate on the same100 with target-observation noise5mm/0.5deg
and with stochastic actions separately. Each stochastic scene has a fixed Torch
seed. These are three conditions on100 layouts, not300 fresh layouts.

| Additional PPO steps | New64 success/collision/timeout |
|---:|---|
|0|51/13/0|
|5,000|52/11/1|
|10,000 (selected)|56/8/0|
|15,000|49/11/4|
|20,000|46/11/7|

| Fresh100 paired test | Success | Collision | Timeout |
|---|---:|---:|---:|
| Source PPO10k |74|25|1|
| Selected continuation |84|16|0|
| Selected, noisy target observation |84|16|0|
| Selected, sampled actions |57|35|8|

All74 source test successes are retained;9 collisions and1 timeout become
successes. Sensor-noise outcome categories are unchanged on these100 scenes,
without implying immunity to other noise levels. Old64 regression improves
51/13/0 to55/9/0. Selected state has20k total PPO in this four-obstacle stage
(10k inherited+10k selected), although the experiment executes20k additional
transitions and preserves the weaker30k-total final candidate separately.
Training completes315 episodes:124 success/172 collision/19 timeout,19,760
completed transitions plus240 unfinished. Training sampling/noise and selected
deterministic test outcomes are separate statistics.

Current84% is below the proposed90% objective. More updates alone introduced
timeouts; keep the validation-selected model. Action-noise performance remains
a material robustness gap. The fresh100 test is now used and must not be
described as a new independent test after future tuning.

Full182 tests pass; checks include real continuation preserving Adam/critic,
hard-layout pool isolation and per-scene stochastic reproducibility. Fixed
validation seed1032001 and fresh random1302077125 headless demos succeed with
four static obstacles and no online planner. These are demonstrations only.

Files:

- configs/four_expanded_ppo.yaml: bank sizes/seeds, sampling, PPO and noise levels.
- scripts/prepare_expanded_four.py: collision replay and scene freezing.
- scripts/train_expanded_four_ppo.py: scan, continuation, selection and tests.
- src/wall_panel_rl/rl/four_ppo_continuation.py: preserved state/hard sampling/noise.
- results/diagnostics/four_expanded_v5_seed61:13-collision contacts and data audit.
- results/diagnostics/four_expanded_ppo_v5_seed61: selected_model.zip,
  candidate_manifest.json,curves.png,checkpoints.csv,evaluations.csv and JSONs.

Play the selected expanded-scene model, random start/layout and0.6x speed:

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_expanded.py
```

Older demo_four_trained.py still reproduces PPOv4; demo_four_static.py retains
the original initialization. No previous model/report or GitHub branch is
overwritten. Reports are mirrored into the main visible results/diagnostics.
