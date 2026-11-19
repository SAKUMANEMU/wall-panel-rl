# Conservative four-obstacle dynamic adaptation

This separate experiment starts from the retained PPOv5 weights, Adam state and
critic. It uses `configs/four_dynamic_ppo.yaml`, not the earlier static trainer.
PPO receives5,000 transitions at learning rate3e-6; four workers times250 ticks
give1,000 transitions per rollout. Other PPO settings and0.01 reward scale are
preserved. The original v5/default demo is not overwritten.

## Data and motion

Freeze192 training layouts,64 fresh validation layouts and100 reserved test
layouts in `data/four_dynamic_v8_seed61`. Training reuses162 source layouts
whose full5cm motion bounds pass geometry checks and adds30 fresh layouts.
The source hard-TRAIN identities that remain valid retain25% hard-case sampling.
Validation, test, old evaluation banks and prior dynamic preflight cases are
excluded from training. All input bank hashes are frozen before optimization.

Scene construction may reject endpoint/wall/spacing geometry or an unavailable
complete obstacle-free reference rollout. These exclusions are logged by stage.
No dynamic-policy success/collision/timeout result filters a scene. Constructing
a reference for the reserved test uses only the obstacle-free source rollout;
the dynamic test task is not evaluated before model selection.

Every training episode independently samples a case, then a static/dynamic mode
(30%/70%), then random motion phases. The controller is a separate50% RRT /50%
PPO episode coin. These two probabilities must not be confused. Four unequal
boxes keep the original on-path anchors,5cm vector displacement bound, zero
sensor noise, fixed goal and77-input/seven-action interface.

Dynamic speed levels are0.005,0.01 and0.02m/s. A new level affects only NEW
episodes; running PPO episodes retain their original trajectories and clock.
Before advancing one level, require no success/collision regression at the
current speed and target0.02m/s, and on both static validation suites. A failed
gate holds the current level; completing5k does not force advancement.

## RRT teacher and auxiliary BC

RRT targets the same immutable reference endpoint. Dynamic planning uses
`TimedRRT`, known obstacle trajectories and arrival-time checks; planning pauses
simulation time. A full swept-region detour is not substituted for timing.

Successful execution alone does not admit BC labels. Dynamic teachers require:

1. The planned timed schedule passes intermediate collision checks.
2. Actual physics executes successfully with the original observation route.
3. Actual joint transitions pass timestamp-aware checks.
4. A separate physics environment replays the executed normalized actions from
   the same initial scene, with matching observations, joints and time, and
   succeeds. The replay tolerance is1e-6.

Static teacher episodes retain the existing static path guard and also require
the physical replay. Failures stay on their assigned branch and are logged;
they provide no BC labels. Evaluation/test episodes never supply training labels.

Only PPO transitions/actions/log probabilities enter SB3's rollout buffer.
Verified RRT pairs enter a separate actor-only BC Adam at the NEXT rollout
start, before sampling. Keep weight0.05, LR1e-6, one update per rollout and
sampled KL cap0.003. BC never updates the critic or alters a collected rollout.

RRT execution and verification replays cost extra physics transitions. Logs
separately count PPO transitions, RRT transitions, verification-replay steps,
rollout totals, and total training-plus-replay physics. Audit replays are not new
training episodes and their labels are not duplicated.

## Validation and selection

Every1k PPO transitions, deterministically evaluate fixed target-speed0.02m/s,
current curriculum speed, frozen-at-initial-phase versions of the64 new cases,
and the original64 static v5 validation cases. At step0 evaluate all three
speeds to establish stage-specific source baselines. Record success, collision,
timeout, errors, reference index/staleness and conditional recovery attempts.
Reference recovery thresholds are monitored, not simultaneously changed.

Select using target-speed PURE-PPO results, subject to no extra collisions and
no regression of either static suite. Equal scores retain the earlier model.
Mixture scores are separate. A common coin seed chooses the same RRT/PPO owners
on the64 target-speed cases. Execute the independent RRT branch once and cache
those actual outcomes; combine them with each checkpoint's matching pure-PPO
rows. Cached expert failures are retained. This paired composition avoids
spending the budget rerunning an expert whose weights never change.

If a new checkpoint is selected, lock it before a single source-versus-selected
100-case pure-PPO dynamic test. Do not tune after this test. If initialization
remains selected, the100 test episodes stay unused. This is one training seed;
further seed repetitions are needed before a general stability claim.

## Commands and paths

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
# Configuration checks only.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_dynamic_ppo.py

# Already launched experiment; do not rerun into its existing output folders.
tail -f /home/johnsonchen/wall-panel-rl/logs/four_dynamic_ppo_v8_seed61.log
cat /home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_ppo_v8_seed61/status.json
```

Preparation report: `results/diagnostics/four_dynamic_preparation_v8_seed61`.
Training report: `results/diagnostics/four_dynamic_ppo_v8_seed61`.
Checkpoints/journals: `models/four_dynamic_ppo_v8_seed61`.
The report directory is also linked from the main project's visible results.

Before launch,28 targeted checks passed. A real training-wrapper probe runs a
successful69-step dynamic RRT episode, matches its fresh replay to below1e-15,
provides69 verified BC pairs, then enters a PPO episode. The probe makes no
optimizer update and retains the PPO collision outcome. Formal training starts
only after these checks.

## Completed result

Run finishes with all5k PPO transitions. Select+2k, not the final candidate:
target-speed validation41/20/3 ->43/18/3; frozen paired44/17/3 ->44/16/4;
original-static56/8/0 ->57/7/0. Cached common-branch mixture stays51/11/2.
Training speed reaches0.02m/s after the+2k gate, but later candidates do not
improve selection; the final5k target result is39/21/4.

The once-reserved100-case dynamic test gives77/22/1 for v5 and79/20/1 for the
selected+2k candidate. It rescues four source collisions and loses two source
successes. This is a small improvement from one training seed. The100 test bank
is now used; future tuning needs another reserved bank for an independent test.

Actual training:66 RRT successes,67 completed PPO episodes(23S/36C/8T), plus
four unfinished PPO episodes. Of66 successful teachers,65 pass the time/replay
label gate; one physically successful replay fails the conservative post-time
check and supplies no BC labels. Save4,832 verified pairs. Five auxiliary BC
updates pass their KL guards. Physics accounting:5,000 PPO +4,958 RRT +4,958
verification-replay transitions=14,916, excluding evaluation trajectories.

To inspect the new selected model in a dynamic demo:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py --controller ppo --model results/diagnostics/four_dynamic_ppo_v8_seed61/selected_model.zip --manifest results/diagnostics/four_dynamic_ppo_v8_seed61/candidate_manifest.json
```

The original default demo still uses v5. Use explicit new model/manifest paths
for this experiment's selected weights.
