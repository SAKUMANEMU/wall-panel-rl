# Enhanced dynamic obstacles: functional preflight

Unchanged model from `results/diagnostics/four_dynamic_speed_v9_fast02_seed71`. Four different-size boxes move smoothly in three axes with random phases/frequencies. The configured Euclidean displacement bound is 0.1m and vector speed bound is 0.05m/s. Planning pauses simulation time; RRT uses time-aware collision checks. Sensor noise is zero.

This small run checks functionality, not general success rate. Scene construction checks geometry and requires a completed obstacle-free nominal rollout to define the on-path anchors. Construction failures are logged separately; no dynamic policy/planning outcome is filtered. Both branches use the same 77-observation / 7-action interface and original reference endpoint. There is no training, checkpoint promotion, or fallback to the other branch.

| Seed | Controller | Outcome | Steps | Position (m) | Planning (s) |
|---|---|---|---:|---:|---:|
| 1070001 | rrt | success | 77 | 0.0004 | 1.75 |
| 1070004 | ppo | success | 75 | 0.0025 | 0.00 |
| 1070005 | ppo | success | 80 | 0.0041 | 0.00 |
| 1070007 | ppo | collision | 43 | 0.6967 | 0.00 |
| 1070010 | rrt | success | 76 | 0.0004 | 1.69 |
| 1070014 | ppo | success | 65 | 0.0003 | 0.00 |
| 1070015 | ppo | success | 71 | 0.0002 | 0.00 |
| 1070016 | ppo | success | 73 | 0.0023 | 0.00 |

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
      "success": 5,
      "collision": 1
    }
  },
  "total_outcomes": {
    "success": 7,
    "collision": 1
  },
  "scene_construction_rejections": 9,
  "rejections_by_stage": {
    "geometry": 8,
    "nominal_reference": 1
  },
  "planning_paused_time_and_restored_scene": true,
  "original_reference_preserved": true,
  "motion_bounds_verified": true,
  "max_displacement_from_anchor_m": 0.09928433400997798,
  "max_observed_speed_m_s": 0.0418143831284482,
  "all_executed_cases_moved_in_xyz": true,
  "default_static_model_and_entries_preserved": true
}
```

See `cases.json` for exact phases/anchors, `episodes.json` for results, `traces/` for moving box/arm trajectories, and `protocol.json` for configuration and hashes.
