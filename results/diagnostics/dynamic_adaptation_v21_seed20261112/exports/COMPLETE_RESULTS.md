# Dynamic adaptation v21: complete scene results

Selected checkpoint: `ppo_24576`. The early-recovery alternative was not selected.
All figures below are measured completed rollouts. New16 is selection-validation; old64 is regression, not a fresh final test. Enhanced execution includes prediction and local recovery.

## Data accounting

- Training scenes: 100; new validation scenes: 16; old regression scenes: 64.
- Scene construction: 584 attempts, 118 geometrically valid, 466 rejected; 2 valid scenes unused.
- Successful training teacher episodes: 100/100; RRT labels are separate from PPO evaluation.
- Evaluation records: 352, across 80 distinct scene seeds; repeated stages are repeated rollouts, not new scenes.

## All stages

| Stage / dataset | Mode | Success | Collision | Timeout | Joint limit | Planning failure | Success rate | Mean waiting |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| source | ppo | 0 | 16 | 0 | 0 | 0 | 0.00% | 0.00% |
| source | enhanced | 9 | 2 | 5 | 0 | 0 | 56.25% | 45.37% |
| bc | ppo | 4 | 12 | 0 | 0 | 0 | 25.00% | 0.00% |
| bc | enhanced | 16 | 0 | 0 | 0 | 0 | 100.00% | 22.30% |
| ppo_8192 | ppo | 7 | 9 | 0 | 0 | 0 | 43.75% | 0.00% |
| ppo_8192 | enhanced | 16 | 0 | 0 | 0 | 0 | 100.00% | 8.02% |
| ppo_16384 | ppo | 9 | 7 | 0 | 0 | 0 | 56.25% | 0.00% |
| ppo_16384 | enhanced | 16 | 0 | 0 | 0 | 0 | 100.00% | 8.68% |
| ppo_24576 | ppo | 9 | 5 | 2 | 0 | 0 | 56.25% | 0.00% |
| ppo_24576 | enhanced | 16 | 0 | 0 | 0 | 0 | 100.00% | 2.06% |
| ppo_32768 | ppo | 6 | 6 | 4 | 0 | 0 | 37.50% | 0.00% |
| ppo_32768 | enhanced | 16 | 0 | 0 | 0 | 0 | 100.00% | 5.04% |
| early_recovery | ppo | 9 | 5 | 2 | 0 | 0 | 56.25% | 0.00% |
| early_recovery | enhanced | 16 | 0 | 0 | 0 | 0 | 100.00% | 2.06% |
| regression64 | ppo | 14 | 31 | 19 | 0 | 0 | 21.88% | 0.00% |
| regression64 | enhanced | 64 | 0 | 0 | 0 | 0 | 100.00% | 9.17% |

## Selected model: all 64 regression scenes

Errors are terminal values. Position is in millimetres and orientation in degrees. S = success, C = collision, T = timeout. Every row is one paired scene.

| Seed | PPO outcome | PPO steps | PPO position (mm) | PPO angle (deg) | Enhanced outcome | Enhanced steps | Enhanced position (mm) | Enhanced angle (deg) | Enhanced zero commands |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| 1188001 | C | 11 | 1710.195 | 130.372 | S | 141 | 0.233 | 0.006 | 9 |
| 1188005 | T | 240 | 324.108 | 25.249 | S | 161 | 1.468 | 0.136 | 0 |
| 1188006 | C | 7 | 1786.691 | 140.601 | S | 86 | 2.073 | 0.125 | 9 |
| 1188007 | C | 8 | 1762.786 | 137.548 | S | 147 | 0.569 | 0.038 | 59 |
| 1188008 | S | 69 | 1.834 | 0.136 | S | 69 | 1.834 | 0.136 | 0 |
| 1188026 | T | 240 | 435.998 | 32.904 | S | 118 | 0.772 | 0.071 | 0 |
| 1188028 | C | 62 | 224.803 | 53.711 | S | 114 | 1.393 | 0.973 | 0 |
| 1188035 | S | 81 | 9.121 | 0.622 | S | 81 | 9.121 | 0.622 | 0 |
| 1188045 | T | 240 | 411.353 | 30.858 | S | 105 | 0.596 | 0.055 | 0 |
| 1188049 | C | 8 | 1778.099 | 140.325 | S | 90 | 0.386 | 0.029 | 0 |
| 1188063 | C | 8 | 1776.341 | 140.607 | S | 158 | 4.733 | 0.449 | 64 |
| 1188068 | C | 3 | 1827.276 | 148.530 | S | 168 | 2.602 | 0.190 | 70 |
| 1188075 | S | 68 | 4.614 | 0.332 | S | 68 | 4.614 | 0.332 | 0 |
| 1188081 | S | 72 | 0.616 | 0.036 | S | 72 | 0.616 | 0.036 | 0 |
| 1188086 | T | 240 | 398.590 | 28.551 | S | 104 | 0.858 | 0.077 | 0 |
| 1188090 | T | 240 | 532.045 | 33.382 | S | 105 | 0.440 | 0.030 | 0 |
| 1188098 | C | 7 | 1786.041 | 141.108 | S | 114 | 0.284 | 0.016 | 15 |
| 1188099 | S | 72 | 3.358 | 0.231 | S | 72 | 3.358 | 0.231 | 0 |
| 1188106 | C | 3 | 1829.105 | 149.108 | S | 97 | 7.098 | 0.528 | 17 |
| 1188109 | C | 7 | 1775.683 | 139.797 | S | 180 | 0.064 | 0.044 | 53 |
| 1188112 | T | 240 | 234.843 | 19.344 | S | 121 | 0.248 | 0.023 | 0 |
| 1188114 | C | 5 | 1796.477 | 143.591 | S | 191 | 0.359 | 0.029 | 101 |
| 1188117 | C | 8 | 1750.292 | 134.903 | S | 108 | 9.233 | 0.592 | 0 |
| 1188120 | S | 84 | 0.213 | 0.014 | S | 84 | 0.213 | 0.014 | 0 |
| 1188121 | T | 240 | 653.125 | 48.873 | S | 109 | 0.549 | 0.047 | 0 |
| 1188122 | C | 11 | 1713.782 | 130.730 | S | 113 | 0.714 | 0.065 | 0 |
| 1188123 | T | 240 | 1019.958 | 64.983 | S | 107 | 6.084 | 1.578 | 0 |
| 1188125 | S | 66 | 0.204 | 0.013 | S | 66 | 0.204 | 0.013 | 0 |
| 1188133 | T | 240 | 418.989 | 28.345 | S | 105 | 4.716 | 1.300 | 0 |
| 1188142 | C | 6 | 1789.955 | 142.306 | S | 84 | 0.228 | 0.015 | 0 |
| 1188145 | T | 240 | 155.105 | 23.781 | S | 111 | 0.585 | 0.054 | 0 |
| 1188159 | C | 5 | 1815.160 | 146.863 | S | 189 | 0.546 | 0.039 | 90 |
| 1188161 | C | 7 | 1779.004 | 139.644 | S | 171 | 1.619 | 0.115 | 74 |
| 1188162 | C | 31 | 1106.726 | 77.395 | S | 145 | 0.442 | 0.041 | 10 |
| 1188172 | C | 10 | 1721.386 | 131.829 | S | 136 | 6.085 | 0.481 | 35 |
| 1188176 | C | 9 | 1719.680 | 131.366 | S | 76 | 5.800 | 0.403 | 0 |
| 1188182 | T | 240 | 393.357 | 29.185 | S | 141 | 0.592 | 0.022 | 0 |
| 1188185 | C | 37 | 995.196 | 69.255 | S | 120 | 0.070 | 0.049 | 0 |
| 1188189 | S | 72 | 6.318 | 0.419 | S | 72 | 6.318 | 0.419 | 0 |
| 1188194 | S | 73 | 2.263 | 0.193 | S | 73 | 2.263 | 0.193 | 0 |
| 1188199 | S | 59 | 7.985 | 0.559 | S | 59 | 7.985 | 0.559 | 0 |
| 1188201 | C | 43 | 328.612 | 25.689 | S | 61 | 1.058 | 0.060 | 0 |
| 1188208 | C | 7 | 1775.123 | 139.057 | S | 152 | 0.374 | 0.029 | 36 |
| 1188209 | C | 44 | 779.060 | 53.769 | S | 114 | 6.238 | 0.583 | 0 |
| 1188212 | T | 240 | 437.292 | 27.655 | S | 104 | 0.699 | 0.064 | 0 |
| 1188216 | T | 240 | 340.681 | 26.747 | S | 115 | 0.213 | 0.020 | 0 |
| 1188223 | T | 240 | 324.577 | 25.769 | S | 122 | 0.626 | 0.057 | 0 |
| 1188224 | T | 240 | 110.429 | 19.020 | S | 105 | 0.325 | 0.030 | 0 |
| 1188231 | T | 240 | 485.277 | 36.865 | S | 104 | 0.877 | 0.058 | 0 |
| 1188232 | S | 72 | 0.277 | 0.019 | S | 72 | 0.277 | 0.019 | 0 |
| 1188233 | T | 240 | 278.130 | 23.941 | S | 105 | 2.167 | 1.514 | 0 |
| 1188241 | S | 114 | 0.452 | 0.039 | S | 114 | 0.452 | 0.039 | 0 |
| 1188244 | C | 11 | 1714.011 | 130.852 | S | 142 | 4.768 | 1.349 | 7 |
| 1188249 | C | 7 | 1776.191 | 139.285 | S | 115 | 0.323 | 0.030 | 0 |
| 1188254 | C | 52 | 398.386 | 31.804 | S | 83 | 1.808 | 0.134 | 0 |
| 1188257 | T | 240 | 362.740 | 29.485 | S | 106 | 0.532 | 0.049 | 0 |
| 1188258 | C | 12 | 1674.205 | 125.753 | S | 195 | 7.268 | 0.496 | 109 |
| 1188261 | C | 6 | 1793.312 | 142.249 | S | 184 | 0.211 | 0.020 | 88 |
| 1188262 | C | 5 | 1791.465 | 142.504 | S | 161 | 4.160 | 0.328 | 78 |
| 1188271 | C | 5 | 1818.023 | 146.657 | S | 143 | 0.297 | 0.017 | 23 |
| 1188274 | C | 14 | 1633.031 | 121.494 | S | 141 | 0.658 | 0.060 | 20 |
| 1188277 | S | 47 | 7.078 | 0.653 | S | 47 | 7.078 | 0.653 | 0 |
| 1188280 | T | 240 | 90.038 | 13.165 | S | 105 | 0.074 | 0.051 | 0 |
| 1188283 | S | 83 | 0.520 | 0.031 | S | 83 | 0.520 | 0.031 | 0 |

## Selected model: all 16 new validation scenes

| Seed | PPO outcome | PPO steps | PPO position (mm) | PPO angle (deg) | Enhanced outcome | Enhanced steps | Enhanced position (mm) | Enhanced angle (deg) |
|---|---|---:|---:|---:|---|---:|---:|---:|
| 1212541 | S | 68 | 5.416 | 0.415 | S | 68 | 5.416 | 0.415 |
| 1212543 | S | 231 | 3.438 | 0.210 | S | 231 | 3.438 | 0.210 |
| 1212544 | S | 83 | 1.819 | 0.123 | S | 83 | 1.819 | 0.123 |
| 1212545 | S | 64 | 1.636 | 0.132 | S | 64 | 1.636 | 0.132 |
| 1212546 | C | 8 | 1754.819 | 136.706 | S | 203 | 2.073 | 0.130 |
| 1212547 | S | 72 | 2.528 | 0.190 | S | 72 | 2.528 | 0.190 |
| 1212554 | S | 83 | 7.698 | 0.518 | S | 83 | 7.698 | 0.518 |
| 1212558 | S | 75 | 8.193 | 0.789 | S | 75 | 8.193 | 0.789 |
| 1212561 | C | 55 | 294.201 | 22.821 | S | 111 | 1.231 | 0.123 |
| 1212566 | C | 57 | 314.907 | 26.802 | S | 71 | 1.099 | 0.090 |
| 1212570 | C | 51 | 410.832 | 31.410 | S | 86 | 2.362 | 0.164 |
| 1212572 | T | 240 | 140.697 | 7.645 | S | 106 | 0.064 | 0.044 |
| 1212573 | C | 142 | 247.162 | 17.962 | S | 105 | 0.651 | 0.060 |
| 1212575 | S | 80 | 0.662 | 0.057 | S | 80 | 0.662 | 0.057 |
| 1212577 | S | 57 | 0.272 | 0.017 | S | 57 | 0.272 | 0.017 |
| 1212578 | T | 240 | 368.790 | 30.257 | S | 126 | 2.829 | 0.861 |

## Files

- `all_evaluated_episodes.csv`: all stages and all original per-episode fields.
- `stage_summary.csv`: stage and execution-mode aggregate counts.
- `selected_model_regression64_paired.csv`: both controllers side by side for all64.
- `selected_model_new16_validation.csv`: selected-checkpoint validation episodes.
- `teacher_episode_outcomes.csv`: training teacher outcomes. Detailed diagnostics remain in `../teacher_attempts.json`.
- Full case geometry, motion phases and reference routes: `../train_cases.json`, `../validation_cases.json`, and the experiment snapshot `data/validation_cases.json` for old64.
- Complete final-run trajectory recordings were not saved by this study; evaluation JSON/CSV contains outcomes and diagnostics, not full replay trajectories. A chosen episode can be rerun and recorded separately.
