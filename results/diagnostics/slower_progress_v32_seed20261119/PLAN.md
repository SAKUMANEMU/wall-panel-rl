# V32 slower-speed training and verification

Source: V31 selected PPO checkpoint, SHA256
`b8650e7264b05923f73b704227afe19f0383ce66ff8c00016d63fa0fba359a10`.
All prior versions are preserved. The V31 selected controller is the paired
same-scene baseline. New scene seed blocks start at 6,100,000 and are disjoint
from the V31 expanded 80-scene test and the two retrospective timeout cases.
Those old test cases are not used as BC/PPO training data.

1. Verify physical speed bounds, planning rollback, fixed reference, site
   collision limits and the real PPO action/update interface.
2. Generate 32/16/40 disjoint train/validation/test scenes by geometry only.
3. Compare the source V31 and new recovery controller on validation scenes.
4. Collect successful training demonstrations (20 cases, up to two attempts
   each), log all failed attempts, then perform up to 500 BC updates.
5. Run 16,384 genuine on-policy PPO steps with 0.03 action standard deviation,
   LR 3e-6, checkpoints at 8,192 and 16,384 steps. The initialization is an
   eligible fallback; BC improvement is not required to start this trial.
6. Select only using validation success, collision count, then pose task error.
   Freeze the bundle before opening the 40 independent test scenes.
7. Evaluate selected and source bundles on the same slowed test cases; report
   success/collision/timeout, per-count metrics, Wilson intervals, paired
   outcomes, CSV, plots and full replay trajectories. Do not replace the model
   using test outcomes.

Physical speed range: 0.04--0.18 m/s, approximately 15.4% lower mean sampled
speed than the old 0.04--0.22 m/s distribution. The maximum falls by 18.2%.
No change to obstacle dimensions, 0.18 m motion extent, floor/ceiling, installed
pose, true success criterion or 240-step episode budget.

Timeout intervention: recovery begins at step 32, attempts physically safe
forward rejoining before detached detours, caps rejoin wait candidates at four
steps and guards against reference loss without requiring goal-distance
regression. Forecasts use rollback and never advance real episode time.

Measured success-rate improvement is an experimental question, not guaranteed
by reducing obstacle speed. Slower obstacles can also keep an opening blocked.
