# V32 slower dynamic-obstacle adaptation

Selected validation checkpoint: `ppo_8192`.

Both bundles execute exactly the same held-out scenes at 0.04--0.18 m/s.
These results measure PPO plus predictive control and local recovery.
Selection is frozen before opening the test bank. No failed episode is discarded.

| Bundle | Episodes | Success | Collision | Timeout | Success rate | 95% Wilson interval |
|---|---:|---:|---:|---:|---:|---|
| selected | 40 | 22 | 7 | 11 | 55.00% | 39.8%--69.3% |
| v31_source_same_slow_scenes | 40 | 20 | 4 | 16 | 50.00% | 35.2%--64.8% |

## Success by obstacle count

| Count | Selected | V31 source |
|---|---:|---:|
| 2 | 5/10 | 6/10 |
| 3 | 9/10 | 7/10 |
| 4 | 4/10 | 3/10 |
| 5 | 4/10 | 4/10 |

## Paired outcomes

{'both_success': 15, 'selected_only': 7, 'source_only': 5, 'both_failed': 13}

The earlier V31 43.75% result used a faster speed distribution and different scenes.
A direct numerical difference from that experiment does not isolate a learning effect.
