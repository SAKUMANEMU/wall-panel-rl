# Four slow dynamic obstacles: functional preflight

Unchanged v5 weights. Four different-size boxes retain the randomized on-path anchors. Known smooth three-axis motion has independent random phases/frequencies, a 0.05 m displacement norm bound and 0.02 m/s velocity norm bound. Planning does not advance the physics clock. RRT uses time-aware collision prediction and may traverse swept regions at safe times; PPO has no online avoidance planner. Noise is zero.

This small run checks functionality, not general success rate. Scene construction checks geometry and requires a completed obstacle-free nominal rollout to define the on-path anchors. Two candidates were rejected for wall-motion overlap and one for a nominal-rollout timeout. No dynamic execution outcome is filtered. Both branches use the same 77-observation / 7-action interface and original reference endpoint. There is no training, checkpoint promotion, or fallback to the other branch.

| Seed | Controller | Outcome | Steps | Position (m) | Planning (s) |
|---|---|---|---:|---:|---:|
| 1040000 | rrt | success | 96 | 0.0009 | 17.00 |
| 1040001 | ppo | success | 73 | 0.0001 | 0.00 |
| 1040003 | ppo | collision | 47 | 0.5112 | 0.00 |
| 1040005 | ppo | success | 66 | 0.0013 | 0.00 |
| 1040006 | rrt | success | 89 | 0.0014 | 2.15 |
| 1040008 | ppo | success | 73 | 0.0041 | 0.00 |
| 1040009 | ppo | success | 66 | 0.0094 | 0.00 |
| 1040010 | ppo | success | 65 | 0.0003 | 0.00 |

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
  "planning_paused_time_and_restored_scene": true,
  "original_reference_preserved": true,
  "motion_bounds_verified": true,
  "max_displacement_from_anchor_m": 0.04700732688988632,
  "max_observed_speed_m_s": 0.017557504534569272,
  "all_executed_cases_moved_in_xyz": true,
  "default_static_model_and_entries_preserved": true,
  "scene_construction_rejections": 3,
  "rejections_by_stage": {
    "geometry": 2,
    "nominal_reference": 1
  }
}
```

See `cases.json` for exact phases/anchors, `episodes.json` for results, `traces/` for moving box/arm trajectories, and `protocol.json` for configuration and hashes.

Preparation exclusions are detailed in `scene_construction_rejections.json`. The original `geometry_rejections.json` and execution archive are retained unchanged for provenance; its broad label included a nominal-reference failure and was clarified after execution.
