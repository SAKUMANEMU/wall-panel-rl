# PPO with predictive execution safety

This is a model-based execution-controller improvement, not newly learned pure PPO. Original PPO weights and optimizers are unchanged. The extra layer runs short physical rollouts at candidate speeds, then restores physics, simulation time, velocities, observations, reference cursor and recovery history before actual execution. It uses known obstacle motion; planning time is paused. It has no global RRT fallback. Actual collisions/timeouts retain the original termination rules.

Fixed scene: four moving unequal staggered obstacles;18cm offset,0.12m/s speed. Selected horizon: 12 steps, chosen using only24 training scenes. No validation/test labels train a model.

Counts below are success/collision/timeout/planning-failure.

| Suite | Source PPO/system | Predictive PPO/system |
|---|---|---|
| Validation PPO branch | 37/26/1/0 | 56/0/8/0 |
| Validation50/50 mixture | 42/19/1/2 | 57/0/5/2 |
| Independent PPO-branch test | 57/41/2/0 | 78/0/22/0 |

Validation transitions: {'success -> success': 37, 'collision -> timeout': 7, 'collision -> success': 19, 'timeout -> timeout': 1}. All original successes reproduce their outcome, episode length and final errors without an intervention. Test transitions: {'success -> success': 57, 'collision -> success': 21, 'collision -> timeout': 20, 'timeout -> timeout': 2}. Parameters are locked before the independent test, with no subsequent tuning.

Virtual validation prediction steps discarded: 110514; interventions: 2233. These are computation, not training samples or real episode transitions. Finite-horizon checking does not guarantee safety beyond the horizon. Extra model information and computation distinguish this controller from pure PPO. Source/default PPO remains available.
