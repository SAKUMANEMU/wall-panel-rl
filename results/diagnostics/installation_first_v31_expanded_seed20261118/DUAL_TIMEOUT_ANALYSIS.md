# Dual-obstacle timeout diagnosis

Retrospective analysis of two original v31 test failures. Model weights, controller settings,
motion, collision constraints and installation tolerances are frozen. These cases are not
new independent validation samples and are never used for training in this study.

| Seed | Final position error | Orientation error | Zero commands | Reference index | Reference distance |
|---|---:|---:|---:|---:|---:|
| 4720000 | 0.564 m | 41.0 deg | 40 | 32/44 | 0.086 rad |
| 4720100 | 0.409 m | 56.8 deg | 19 | 5/47 | 3.268 rad |

## Interpretation

- Seed 4720000: the reference still advances near the end, but the installation pose remains far away. Waiting and recovery consumed time; this is insufficient completion progress, not a millimetre-level terminal-precision failure.
- Seed 4720100: the recorded reference is stale while the robot continues moving. Recovery rejection limits some detours, but later accepted recovery does not restore the tracking interface. This is route-tracking/rejoin failure rather than simply standing still.
- Both terminate at the original 240-step budget without collision. Initial obstacle speeds are below 0.08 m/s, so excessive obstacle speed alone does not explain these failures.
- Valid start/goal placements do not prove that these controllers can execute a feasible dynamic path.
- Recovery windows in the JSON are observed trajectory intervals, not a claim that every predicted action was executed.

## Frozen 2 x 2 diagnostic counterfactuals

| Seed | Model / controller | Result | Steps | Position error | Orientation error |
|---|---|---|---:|---:|---:|
| 4720000 | v29_model__installation | timeout | 240 | 1.057 m | 105.8 deg |
| 4720000 | v29_model__old | timeout | 240 | 1.440 m | 148.5 deg |
| 4720000 | v31_model__installation | timeout | 240 | 0.564 m | 41.0 deg |
| 4720000 | v31_model__old | timeout | 240 | 0.564 m | 41.0 deg |
| 4720100 | v29_model__installation | timeout | 240 | 0.400 m | 31.3 deg |
| 4720100 | v29_model__old | timeout | 240 | 0.400 m | 31.3 deg |
| 4720100 | v31_model__installation | timeout | 240 | 0.409 m | 56.8 deg |
| 4720100 | v31_model__old | timeout | 240 | 0.669 m | 179.4 deg |

The expanded random-scene study is reported separately. It does not tune or select models.
