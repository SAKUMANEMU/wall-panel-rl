# Farther starts and three random obstacles: measured RRT validation

Seeds: 10000–10099; 100 episodes per stage. Initial arm states are paired across stages.

| Stage | Mean initial distance (m) | Success | Collision | Blocked direct edges | Successful detours |
|---|---:|---:|---:|---:|---:|
| far_start | 0.922 | 100/100 | 0 | 0 | 0 |
| far_three_obstacles | 0.922 | 100/100 | 0 | 84 | 84 |

Original starts averaged 0.486 m on the same seeds. Farther starts ranged from 0.804 to 1.032 m.

The three-obstacle stage uses exactly three randomly placed boxes, not fixed centers.
The direct-edge audit checks the actual IK goal selected by RRT. Physical execution
uses the same motor controller, collision definitions and strict 1 cm/2 degree
success threshold as the original project. No test failures were excluded.
Successful validation of these seeds is not a guarantee for every future random scene.

`episodes.csv` contains all 200 physical rollouts. `scenarios.json` records the actual
initial joint configurations, targets and obstacle geometry. `summary.json` contains
the unrounded aggregates; resolved YAMLs and representative trajectories are retained.

Tests: 18 passed. Original three-stage smoke test passed. A fresh-seed GUI demo
also succeeded (seed 330984964; see `gui_demo.txt`). Existing PPO policies were
not retrained for this extension.

Reproduce with a new output ID:

```bash
python scripts/verify_far_scenarios.py --run-id my_far_random_three
python scripts/plot_far_scenarios.py --run-id my_far_random_three
```
