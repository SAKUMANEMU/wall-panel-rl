# Four moving obstacles: functional preflight

Unchanged model from `results/diagnostics/four_dynamic_speed_v9_fast02_seed71`. Four different-size boxes retain randomized on-path anchors. Known smooth three-axis motion has independent random phases/frequencies, a 0.15 m displacement norm bound and 0.1 m/s velocity norm bound. Planning does not advance the physics clock. RRT uses time-aware collision prediction and may traverse swept regions at safe times; PPO has no online avoidance planner. Noise is zero.

This small run checks functionality, not general success rate. Scene construction checks geometry and requires a completed obstacle-free nominal rollout to define the on-path anchors. Construction failures are logged separately; no dynamic policy/planning outcome is filtered. Both branches use the same 77-observation / 7-action interface and original reference endpoint. There is no training, checkpoint promotion, or fallback to the other branch.

| Seed | Controller | Outcome | Steps | Position (m) | Planning (s) |
|---|---|---|---:|---:|---:|
| 1080009 | rrt | success | 84 | 0.0013 | 2.27 |
| 1080012 | ppo | collision | 43 | 0.7323 | 0.00 |
| 1080031 | ppo | collision | 43 | 0.6454 | 0.00 |
| 1080055 | ppo | success | 63 | 0.0045 | 0.00 |
| 1080080 | rrt | success | 81 | 0.0007 | 1.57 |
| 1080113 | ppo | timeout | 240 | 0.2048 | 0.00 |
| 1080142 | ppo | collision | 7 | 1.7196 | 0.00 |
| 1080143 | ppo | success | 75 | 0.0012 | 0.00 |

```json
{
  "episodes": 8,
  "expert_probability": 0.5,
  "training_steps": 0,
  "source_unchanged": true,
  "model_weights_and_update_counter_unchanged": true,
  "independent_test": false,
  "by_controller": {
    "rrt": {
      "success": 2
    },
    "ppo": {
      "collision": 3,
      "success": 2,
      "timeout": 1
    }
  },
  "total_outcomes": {
    "success": 4,
    "collision": 3,
    "timeout": 1
  },
  "scene_construction_rejections": 136,
  "rejections_by_stage": {
    "geometry": 132,
    "nominal_reference": 4
  },
  "planning_paused_time_and_restored_scene": true,
  "original_reference_preserved": true,
  "motion_bounds_verified": true,
  "max_displacement_from_anchor_m": 0.1486995587993148,
  "max_observed_speed_m_s": 0.08974718407877691,
  "all_executed_cases_moved_in_xyz": true,
  "default_static_model_and_entries_preserved": true
}
```

See `cases.json` for exact phases/anchors, `episodes.json` for results, `traces/` for moving box/arm trajectories, and `protocol.json` for configuration and hashes.
