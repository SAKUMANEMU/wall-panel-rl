# Expanded frozen v31 verification

No model updates or model selection were performed. Both combinations ran the same
fresh scenes. Geometry, motion distributions, installation tolerances and budget are unchanged.

| Combination | Episodes | Success | Collision | Timeout | Success rate | Wilson 95% interval |
|---|---:|---:|---:|---:|---:|---|
| v31_selected | 80 | 35 | 8 | 37 | 43.8% | 33.4%--54.7% |
| v29_baseline | 80 | 29 | 8 | 43 | 36.2% | 26.6%--47.2% |

## Results by obstacle count

| Count | v31 success | v31 collision / timeout | v29 success | v29 collision / timeout |
|---|---:|---:|---:|---:|
| 2 | 12/20 | 1 / 7 | 13/20 | 2 / 5 |
| 3 | 9/20 | 0 / 11 | 5/20 | 1 / 14 |
| 4 | 8/20 | 3 / 9 | 4/20 | 4 / 12 |
| 5 | 6/20 | 4 / 10 | 7/20 | 1 / 12 |

## Paired outcomes

Both succeeded: 17; v31 only: 18; v29 only: 12; both failed: 33.
Paired success-rate difference: +7.5%.

The retrospective two-case diagnostic is separate and is not counted in these totals.
These are results for PPO plus predictive control and recovery, not pure PPO.
