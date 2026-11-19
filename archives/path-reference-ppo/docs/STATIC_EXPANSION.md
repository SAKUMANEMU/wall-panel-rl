# Static obstacle expansion — 2026-11-02

Execute the first two steps of the planned curriculum: prospective one-static
verification, then three sparse static obstacles with one bounded adaptation.
Dynamic motion is deferred until the static-stage checks pass.

## Frozen scenes and separation

Generate 100 new full source scenes, starting at seed 1006000, without looking at
policy/expert performance. All 100 satisfy the start/goal and obstacle-spacing
checks; none is rejected. Project each source into paired one- and three-obstacle
cases with identical start, goal and saved obstacle-free reference. Keep sensing
noise zero, 77 features, seven joint-increment actions, reward, recovery rules and
240-step horizon. Sparse training/validation box gaps exceed .22m/.23m.

The 100-case banks are not supplied to training/checkpoint selection. Training
continues from the selected one-static model on the original 54 training layouts;
the three-stage validation uses the original 20 distinct validation scene IDs.
PPO weights and optimizer continue. A fresh per-stage auxiliary Adam and a new
three-obstacle RRT demonstration pool are used with the same BC hyperparameters.

## Observed results

Single-obstacle prospective 100: **95 successes, two collisions, three timeouts**;
far/near alignment is 20/20. This passes the predeclared gate and starts the
three-static pilot. The old 0/20 collision observation was not a general guarantee.

Three-obstacle pure PPO validation:

| Additional PPO transitions | Success | Collision | Timeout | Limits |
|---:|---:|---:|---:|---:|
| 0 | 0/20 | 1 | 17 | 2 |
| 5,000 | 0/20 | 14 | 6 | 0 |
| 10,000 | 0/20 | 20 | 0 | 0 |

The RRT expert succeeds in all 20 validation targets. No PPO checkpoint meets
the minimum task-ability/safety criteria. The diagnostic selection keeps step 0,
with exactly the original source network tensors; it is not a newly usable
three-obstacle policy. The 100 held-out three-obstacle cases give 0 success,
four collisions, 90 timeouts and six joint-limit failures for both source and
this retained diagnostic model.

The adaptation samples exactly 10,000 PPO transitions plus 5,080 RRT transitions.
RRT completes 99 successes and two collisions; only the 5,055 successful teacher
transitions enter the BC pool. Ten auxiliary updates pass the sampled KL check.
PPO completes 23 successful **obstacle-free alignment** episodes, 25 three-static
timeouts, 43 three-static collisions and two three-static limit failures. No PPO
three-static episode succeeds. Four PPO episodes remain partial at the boundary.

Replaying the source-retaining diagnostic model on the one-static 100 naturally
reproduces 95/2/3. It does not demonstrate retention by the rejected 5k/10k updates.
The original source/default model remains unchanged and dynamic training is not
started. The new test banks, source file and held-out separation are verified.

## Input-order diagnosis

The original one-box projection puts source obstacle 2 in slot 0; the three-box
projection orders source obstacles [1,2,3]. With frozen source weights and the
same 20 physical validation scenes, reordering to [2,1,3] yields 0 success and
20 collisions. Thus preserving the original obstacle in slot 0 does not solve
the transfer failure. The differing failure behavior indicates order sensitivity;
it does not isolate a sole cause. The diagnostic order is not adopted or used
for training, checkpoint selection or the fresh test bank.

Before moving to dynamic obstacles, investigate a two-obstacle bridge and a
stronger multi-obstacle BC warm-up, while retaining the one-static regression
checks. No extra training beyond the predeclared 10k budget is run here.

## Files and reproduction

- Config/protocol: `configs/static_expansion.yaml`.
- Execution: `scripts/run_static_expansion.py`; no flag checks inputs only.
  `--run` refuses existing outputs.
- Three-stage training config: `configs/static_three_rrt_ppo.yaml`.
- Data: `data/static_expansion_v1_seed61/`. `fresh_full.json` preserves all five
  potential source obstacles; `one_fresh.json`/`three_fresh.json` are the actual
  static evaluation projections.
- Overall report: `results/diagnostics/static_expansion_v1_seed61/`.
- Three-stage training report: `results/diagnostics/static_three_rrt_ppo_v1_seed61/`.
- Raw training models/journals: `models/static_three_rrt_ppo_v1_seed61/`.

Reports are mirrored to the main checkout's visible `results/diagnostics/` folder.
The default demo continues to show the verified one-static model. To show the
three-obstacle environment with the successful RRT baseline:

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_episode_mixture.py \
  --controller rrt \
  --model results/diagnostics/static_three_rrt_ppo_v1_seed61/selected_model.zip \
  --manifest results/diagnostics/static_three_rrt_ppo_v1_seed61/manifest.json \
  --scene-bank data/static_expansion_v1_seed61/three_validation.json \
  --seed 900000 --playback-speed 0.6
```

`--controller ppo` replays the failed diagnostic policy; the report explicitly
marks that candidate ineligible. This distinction keeps environment feasibility,
expert performance and learned policy performance separate.
