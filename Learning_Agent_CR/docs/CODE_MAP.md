# Code reading guide

## Recommended order

1. `src/wall_panel_rl/envs/wall_panel_env.py`: `reset`, `observation`, `step`, and `info` define the physical task. Rewards and terminal decisions come from this layer.
2. `src/wall_panel_rl/rl/waypoint_ppo.py`: adds the normalized seven-joint local reference and computes progress using the same waypoint before/after an action.
3. `src/wall_panel_rl/rl/autonomous_dynamic_ppo.py` and `avoidance_curriculum.py`: reference-preserving policy, residual/gated action components, obstacle slots, and clearance shaping.
4. `src/wall_panel_rl/rl/multi_obstacle_adapter.py`: `MultiObstaclePolicy` is the submitted checkpoint class. Its adapter input is 77 observation values + 7 reference errors + 7 base-action values. `configure_learning` specifies the exact trainable subset.
5. `src/wall_panel_rl/rl/conditional_recovery.py` and `slow_dynamic_four.py`: local-reference recovery and the four-moving-obstacle environment. They retain the original reference path and success definition.
6. `src/wall_panel_rl/submission.py`: portable `CaseBankEnv` and `run_episode` connect these project modules to the short public scripts.
7. `scripts/train_agent.py`: PPO rollout/update entry. It samples only the training bank; its actions come from the stochastic policy, not a teacher/controller.
8. `scripts/evaluate_agent.py`: deterministic agent evaluation, explicit controller ownership, complete outcome counting, and optional multiprocessing.
9. `src/wall_panel_rl/rl/predictive_safety.py`: `EpisodeSnapshot` saves/restores Bullet and Python episode state; candidate rollouts must not advance the real clock.
10. `src/wall_panel_rl/rl/progress_recovery.py`: stagnation detection, baseline-continuation protection, and bounded local recovery trials. This is an execution controller, not a neural-network update.
11. `src/wall_panel_rl/visualization/episode_replay.py`: independent pose playback. It never calls the controller or `stepSimulation` during playback.

## Data flow

```text
frozen case + runtime configuration
              |
              v
      Gymnasium reset -> observation (77)
              |                |
              |                v
              |       PPO policy -> action (7)
              |                |
              |      optional execution controller
              |                |
              +------ physical step / shared projection
                               |
                    reward + next state + termination
                               |
                    real-state recording / metrics
                               |
                     fixed-rate display-only replay
```

PPO training uses the direct policy branch only. The actor is a Gaussian policy; SB3 retains the sampled action and its original log probability for the clipped PPO objective. The environment's bound projection is part of the transition dynamics. Expert transitions are not silently placed in the on-policy buffer.

## Configuration and data

- `configs/runtime.json`: the actual physics configuration, wrapper settings, and enhanced-controller parameters used for the supplied results. Reward weights and target tolerances are here.
- `configs/training.yaml`: the portable short PPO continuation settings. These small rollout defaults are distinct from the historical checkpoint metadata.
- `models/model_card.json`: checkpoint policy class, hyperparameters, action standard deviation, trainable parameters, cumulative source counter, and checksum.
- `configs/scene.yaml` and `preliminary_scene.yaml`: larger-stagger and preliminary scene generation rules.
- `data/train_cases.json`: four development layouts for training-interface checks.
- `data/validation_cases.json`: the 64 frozen current-layout validation scenes.
- `data/demo_case.json`: the separately selected successful demonstration.
- `data/split.json`: split seeds and purpose. Training code checks validation/demo exclusion.
- `experiments/`: historical frozen study outputs; the public scripts write new runs only to `outputs/`.

Some underlying configuration fields are retained for compatibility with the original environment (including its five-slot obstacle capacity and historical PPO defaults). The active case bank contains **four** obstacles per case; `settings.active_stage` supplies the current four-obstacle stage. The public training schedule comes from `configs/training.yaml`, not `configuration.ppo` or `settings.additional_timesteps` in the historical runtime snapshot. Editing a generation profile affects newly generated scenes, not the geometry already frozen in the supplied case banks.

## Existing project modules

The original Python module paths are preserved because the SB3 checkpoint references its custom policy class by module name. The package includes the complete local import dependency closure of its public environment, policy, planner, controller, generator, and visualizer. Historical machine-specific orchestration scripts and large run directories are not needed to execute this submission.

Original BC/RRT integration utilities remain in `rl/episode_imitation.py`: teacher episodes and supervised updates are kept separate from PPO's on-policy rollout collection. The portable trainer exercises PPO continuation without auxiliary BC updates. IK/RRT implementations and the timed scheduler are in `planning/`; PyBullet supplies geometry/kinematic queries.

## Outcome semantics

Collision/limit failures terminate immediately; exhausting the fixed physical budget is a timeout. A planner failure in the explicitly selected RRT mode stays a planner failure. Replay does not reinterpret any recorded outcome. The vertical-passage calculation is postprocessing only and never modifies rewards, motion, or success criteria.
