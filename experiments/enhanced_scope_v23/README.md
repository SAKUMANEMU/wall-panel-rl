# Enhanced control and expanded scene scope v23

This isolated study returns to PPO plus predictive control plus local recovery.
Model weights stay frozen; no BC/PPO optimizer is run in this study. The earlier
100% result applies to the old64 layout and is not assumed on the expanded bank.

## Scene scope

- Four unequal moving obstacles; amplitude bound 0.18 m and speed bound 0.12 m/s.
- Initial-joint sampling keeps its original 0.03 rad spread and uses new seeds.
  The start-distance requirement and fixed installation target stay unchanged.
- Equal high/low guide components. Both guide checkpoints are frozen before
  generation and have checksums in the protocol. Candidate controllers cannot
  change the scene generator or replace failed final outcomes.
- Common XY translation up to 0.12 m, plus individual XY jitter up to 0.02 m.
- Common vertical translation between -0.12 and +0.65 m, further constrained so
  the WHOLE box and its motion envelope remain between z=0.02 and z=1.65 m.
- Reject start/goal overlap, wall/box spacing violations and boxes whose swept
  envelopes miss the frozen physical guide/reference corridors. All four boxes
  require witnesses. Geometry checks do not select final task successes.
- Generate 8 development plus 16 test scenes per guide: 16 new development,
  32 new test. Add 8 known high and 8 known low cases for controller development.
  The low cases from v22 are already inspected diagnostics, not independent tests.

`configs/expanded_scope.yaml` contains the spatial limits. Recorded witness
metadata before translation remains historical; new scope metadata certifies the
current envelope/corridor checks, not actual successful avoidance.

## Controller comparison

Compare the v21 enhanced-selected actor and v22 pure-selected actor under:

1. Original v21 predictive horizon 12 and local recovery horizon 36.
2. Earlier recovery (step 60, window/cooldown 16, up to 5 recoveries).
3. Diverse recovery: predictive horizon 18, recovery horizon 48, plus local
   reference-following and paired-joint perturbation candidates.

All hypothetical trials restore physics and environment state. Actual execution
uses the same collision definition, joint bounds, 240-step budget and 1 cm / 2 deg
installation criteria. No global RRT fallback is introduced. Selection maximizes
development success among candidates with no increase in baseline collisions or
joint-limit failures; failures/waiting/steps break ties. Final test outcomes do not
select parameters. Evaluate old64 and all known low32 separately as regressions.

## Commands

```bash
python -m pytest -q
python scripts/check_enhanced_scope.py
python scripts/run_enhanced_scope.py \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/enhanced_scope_v23_seed20261113
```

Use the existing project Conda interpreter, and clear inherited PYTHONPATH if
needed. One thread per worker is recommended. The runner saves atomic status,
construction attempts, frozen case banks, all candidate outcomes and selected
runtime/model. New runs require a fresh output folder. Partial failed runs are
retained rather than silently overwritten.

After completion, use both the selected runtime and model for reproduction:

```bash
python scripts/demo_scope.py \
  --run /home/johnsonchen/wall-panel-rl/results/diagnostics/enhanced_scope_v23_seed20261113 \
  --headless
```

Omit `--headless` for playback after simulation. Final selected-test and low32
recordings are also saved under the run's `recordings/` folder.
