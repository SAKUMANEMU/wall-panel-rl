# Independent two-static adapter (2026-11-03)

This study continues the existing one-static controller, rather than training
from scratch. It keeps the original one-static default and every earlier run.

## Why change the previous extension?

The real teacher observations contain a populated second obstacle slot, and
its input weights receive nonzero gradients. There is no missing-input or
disconnected-gradient fault. However, the selected previous candidate has zero
weights on that slot and therefore no response to moving the second obstacle.
Only allowing new input columns to change leaves the downstream actor frozen.
Slot swapping also changes the response of the frozen first-obstacle actor.
These are design limitations to test, not proof of a single failure cause.

## New learning interface

`MultiObstaclePolicy` freezes the original actor after zeroing unused obstacle
input columns. A separate two-layer, 128-unit correction network uses the same
77 observations, local waypoint error and frozen base action. Its output starts
at zero. The correction is active only when at least two obstacles are present,
and tapers to zero at the installation goal. BC and PPO likelihoods use the same
action mean. Physics applies the existing actuator projection in all modes.

Only this new correction network is trained by BC/auxiliary imitation. PPO can
also update the critic. Original one/zero-obstacle actions stay exactly equal.
There is no collision-query action filter or online RRT in pure PPO execution.

## Targeted data and bounded training

Audit only the 54 non-held-out training layouts. Compare a one-obstacle RRT
route with the same scene plus its second obstacle. Where that route is blocked,
collect a two-obstacle RRT demonstration. Also trace the initial policy, hand
over before failed actions, and collect successful RRT correction suffixes.
Replay saved prefixes and labels physically; prefixes are not expert labels.

Merge with the existing 256 demonstrations, retain the 10-layout supervised
holdout, and preserve canonical obstacle order. New demonstrations receive four
times the episode sampling weight. Task weights are 80% two-static, 10%
one-static and 10% alignment. Use 3,000 BC updates with checks at 0, 100, 250,
500, 1,000, 2,000 and 3,000.

PPO starts only if a nonzero BC checkpoint both responds to the second obstacle
and improves fixed validation success/collisions versus initialization while
preserving prior skills. The startup gate is distinct from the final release
gate (at least 80% success and at most 10% collisions). PPO is limited to 10,000
additional transitions, checked at 5,000/10,000, with 50/50 RRT/PPO episode
selection and separate supervised auxiliary updates. Expert transitions never
enter PPO's on-policy buffer. No automatic longer run or dynamic stage follows.

Freeze a new 100-scene test from seeds 1012000 onward before fitting. Select the
candidate using fixed validation only, then score initialization and candidate
on the new test. The old one-static 100-scene bank is a regression check.

## Results and commands

The bounded run is complete and **not adopted**. BC selects update250: fixed
validation15/20 success,2 collisions,3 timeouts versus initialization15/20,
4 collisions,1 timeout. PPO5k gives14/3/3; PPO10k gives14/4/2, so no PPO update
is retained. On the new paired100-scene test, initialization gives87/8/5 and
the learned candidate76/14/10. The validation collision reduction does not
generalize. One-static known regression stays95/2/3 with identical trajectories.
Full147 tests pass. Preserve the previous models and do not advance stages.

See the [complete report](../results/diagnostics/two_static_adapter_v3_seed61/README.md),
[checkpoint counts](../results/diagnostics/two_static_adapter_v3_seed61/comparison.csv)
and [curves](../results/diagnostics/two_static_adapter_v3_seed61/overview.png).
The report distinguishes initialization, BC and PPO contributions and records
whether the candidate meets all release checks.

Run from `/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo`:

```bash
# Read-only configuration check; does not train.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_two_static_adapter.py

# Explicit candidate demonstration, pure policy by default, 0.6x playback.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_two_static_adapter.py

# Verification without a GUI.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_two_static_adapter.py --headless
```

The selected successful demo is for visualization only. It does not estimate a
success rate. The default `demo_autonomous_ppo.py` still uses the one-static model.
Do not rerun `--run` into an existing experiment: the runner refuses overwrites.
