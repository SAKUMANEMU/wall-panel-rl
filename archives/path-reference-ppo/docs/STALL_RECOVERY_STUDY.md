# Stall recovery comparison

Start both variants from the selected 65% BC checkpoint (20000 cumulative BC
updates), preserving the 240-step task and original success/collision criteria.
Only training layouts minus the original supervised held-out seeds supply expert
recovery labels. Diagnosis on those 54 layouts yields 39 successes, 8 collisions,
7 timeouts; several timeouts have nearly equal/opposite prior and residual means.

- Collect 40 successful expert recoveries from timeouts. Handoffs cover stagnation
  onset, stationary states and preceding approaches, at most step 160, leaving at
  least 80 physical steps for the expert. At most 120 candidates, no quota retries.
  Save only expert suffix labels and audit 10 complete prefix+suffix replays.
- BC: parent 200-episode mixed data plus 40 recoveries; 10000 additional updates
  at LR 1e-4 (previously 1e-3), restoring the matching BC Adam/RNG state. Preserve
  the old held-out membership and evaluate every 2000 updates, including step 0.
- PPO: independently start from the SAME 65% BC, with 10000 actual additional
  environment transitions, four workers, 250 rollout steps, LR 1e-4, three epochs,
  clip .05, target KL .005, fixed exploration std .10. Original source PPO counter
  is retained separately; BC updates are not environment timesteps. PPO loads its
  own original optimizer state, not the separately saved BC Adam.
- PPO training adds a bounded -0.10 stagnation penalty when a full 20-step window
  has <1mm position improvement, <.1 degree orientation improvement, <.01 rad
  remaining-reference improvement AND <.02 rad net joint displacement. A moving
  detour or reference/pose progress exempts the action. No extra terminal penalty,
  hard action filter, expert execution fallback or waypoint rewrite.
- PPO worker Monitor logs contain shaped reward. All selection and reported
  evaluation rewards use the original environment/reward, permitting comparison.
  Metrics include success/collision/timeout/limits and far/near alignment. Both
  selectors can retain step 0 if all updates regress. PPO is gated by a fresh score
  of the starting 65% BC before creating workers/output.
- Freeze 100 new test scenes from seed 990000 onward, disjoint from training,
  validation and the already-seen 980000 bank. Lock selections using the original
  validation only, then score baseline, BC and PPO once on the new bank. Never
  choose a winner using test results or start another stage automatically.

This is a practical comparison of bundled interventions (new recovery data and
lower-LR BC versus PPO with shaped reward), not an isolation of algorithm effects.
One optimizer seed and 20 reused validation layouts do not establish significance.
Dynamic obstacles, stronger residual architecture changes and extra budgets are
outside this pilot.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_stall_recovery.py
# --run executes once, refusing existing model/data outputs.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/run_stall_recovery.py --run
```

Config: `configs/stall_recovery_study.yaml`; visible report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/stall_recovery_v3_seed61`.
