# Known geometric routes with random episode speeds v28

The agent and predictive controller receive each obstacle's geometric route,
current route progress and current velocity. Each episode samples 2--5 differently
sized, large dynamic obstacles with equal probability. Each obstacle samples its
own travel speed uniformly from 0.03--0.18 m/s, held fixed during that episode.
Training resets also resample route harmonics, signs and initial progress.
Routes remain fixed within an episode. No future random speed schedule exists
or is exposed. With a fixed episode speed, future positions can legitimately be
computed from the known geometry and current progress.

## Motion and observations

The closed 3D route is `anchor + extent * sin(harmonics * u + phase)`.
Integer harmonics and phase signs define the shape. A 2048-segment arc-length
lookup separates geometric shape from timing, giving approximately constant
physical travel speed in simulation seconds. Current velocity is the geometric
tangent scaled to that episode's speed. The same motion function is used by
physics, observations and forecast rollouts. Prediction does not consume a
separate random motion stream.

The observation has **167 values**, with **7 joint actions**:

| Features | Dimensions |
|---|---:|
| Original robot, target, obstacle and reference inputs | 77 |
| Five route parameter slots | 5 x 18 = 90 |

Each route slot contains relative center (3), normalized extent (3), shape
harmonics (3), sine/cosine of geometric phase (6), sine/cosine of current route
progress (2), and current speed (1). Absent obstacle and route slots are zero.

`KnownRoutePolicy` migrates the v27 selected checkpoint. Its existing 77-input
actor is preserved; zero-initialized route correction and critic projection
outputs preserve the initial action distribution and value function. BC/PPO
can then learn the new features. Successful teacher actions belong only to a
separate supervised stream; PPO rollouts keep sampled actions and their original
log probabilities. The current initialization file is not overwritten.

## Scene generation and evaluation

New geometry-only seed blocks generate 48 training, 16 validation and 16
independent test scenes. All physical boxes must intersect a guide configuration
at their anchors; every route passes through its anchor. One generated obstacle
is timed to intersect the guide at the original simulation timestamp. Full motion
envelopes protect endpoints, wall, spacing and vertical workspace only; execution
uses real moving boxes. The 25% larger dimensions and installation tolerances
are retained. No candidate success filter is used in generation.

As requested, **all formal checkpoint and independent-test evaluations use PPO
plus predictive control plus local recovery**. There is no formal pure-PPO arm or
with/without-route ablation. Raw PPO training-episode counts are still logged for
optimization diagnostics. Enhanced results must not be described as pure-PPO
performance or as proof of the isolated benefit of route information.

The default pipeline performs 1,000 BC updates, then 65,536 additional PPO steps
with an 8,192-step pilot check and checkpoints every 8,192 steps. Enhanced physical
validation can take longer than the optimizer itself.

## Run and inspect

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/known_route_speed_v28
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -m pytest -q
```

The active run is under:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/known_route_speed_v28_seed20261116/
  pipeline_status.json
  scenes/status.json
  training/status.json
  generate_scenes.log
  train_and_evaluate.log
  training/models/
  training/evaluations/
  training/recordings/independent_test_enhanced/
```

To reproduce with fresh directories:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/generate_variable_cases.py --output outputs/route_scenes
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/run_variable_training.py --scenes outputs/route_scenes --output outputs/route_training
```

Use `scripts/replay_episode.py RECORDING.json --fps 30 --playback-speed 0.6
--timing smooth` for a saved episode. No video export is needed. Archived v27
tests require the original 77-input checkpoints; v28's active suite checks the
167-input interface, physical speed, migration, route sampling and rollback.

## Resume after losing a paused process

`scripts/resume_known_training.py --output RUN/training --checkpoint
RUN/training/models/ppo_8192.zip` restores a completed update checkpoint, including
the PPO optimizer. It verifies the original source/bank hashes, retains completed
evaluation rows, and continues the original total step budget. BC is not repeated.
Physical environments and RNG streams restart; continuation is not bitwise
identical to an interrupted in-memory episode. New resume logs are separate and
the original launch/pause records remain available. The active resumed launch is
recorded in the run's `training.launch.json`.
