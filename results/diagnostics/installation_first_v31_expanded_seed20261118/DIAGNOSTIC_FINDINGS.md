# Initial trajectory and geometry findings

These observations describe the two original failed episodes. Geometry checks
use saved physical poses and `getClosestPoints`; they perform no physics steps.
The frozen 2 x 2 counterfactual table is written to `DUAL_TIMEOUT_ANALYSIS.md`.
No settings or weights are adjusted from these findings in this verification run.

## Seed 4720000: early tracking delay, then whole-arm clearance bottleneck

- Reference advance pauses for 107 control steps (26--132), and later resumes.
  It reaches index 32 of 44 at timeout; nearest forward reference distance is
  only 0.086 rad at the final state.
- The episode contains 40 zero-command steps. The longest zero-command run is
  16 steps (190--205), so this is not 240 steps of passive waiting.
- The closest object is initially obstacle slot 0 against the held panel.
  At step 200 the closest pair is obstacle slot 1 and robot link 4, separated
  by about 1.215 mm. At step 240 it is slot 1 and robot link 6, about 1.218 mm.
- Final errors are 0.564 m and 41.0 degrees. The late trajectory still advances;
  this is insufficient end-to-end completion progress, with distinct early
  tracking and late arm-clearance bottlenecks, rather than final precision alone.

## Seed 4720100: persistent reference loss after an early blocked approach

- The reference stays unchanged for 110 steps (25--134), then another 105 steps
  (136--240). The final reference is index 5 of 47 while the robot keeps moving.
- Final current-reference distance is 3.268 rad. Even the nearest future path
  point is 0.966 rad away, above the controller's 0.65 rad rejoin bound; the
  normal/extended observation-recovery radii are 0.30/0.45 rad.
- There are 19 zero-command steps. Seven recovery candidates are rejected by
  the reference-escape safeguard, yet later recovery still does not restore
  forward route tracking. Final errors are 0.409 m and 56.8 degrees.
- At step 40, the floor-to-obstacle-bottom opening is about 0.162 m, whereas
  the panel's current world-Z extent is about 0.323 m. A direct underpass at
  that instant and orientation cannot fit. This is not a proof that a rotated,
  lateral or overhead full-arm route is feasible; those require full collision
  checking. At step 80 the panel-obstacle distance is about 0.639 mm.

## Limits and next questions

Both episodes have valid start/goal geometry and no executed collision, but this
does not certify a traversable dynamic route for these controllers. Both boxes
move below 0.08 m/s in these examples; simply reducing maximum speed is not a
demonstrated remedy. Obstacle count alone does not determine difficulty because
size, placement, phase and available opening differ.

Later interventions should be tested on separate training analogues: earlier
safe progress around the first box, intermediate time-aware rejoin targets before
joint deviation exceeds local recovery bounds, and full-arm clearance near the
second box. The two diagnosed test seeds are not added to the training set here.
The fresh 80-scene evaluation stays frozen and retains every outcome.
