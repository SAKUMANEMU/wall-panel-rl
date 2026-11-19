# Timeout-aware enhanced recovery v24

Isolated development study for PPO plus prediction plus local recovery. It does
not modify the running v23 scope experiment, PPO weights, success tolerances,
collision semantics, or the 240-step physical episode budget.

## Why another recovery variant

On the known low32 diagnostic bank, the v21 enhanced controller has 13 successes,
one collision and 18 timeouts. Timeouts include both long zero-command waits
with no accepted recovery and unsuccessful repeated recoveries. More waiting
penalties alone would not address both patterns.

The candidates detect stalls from step40 with a 16-step window and 12-step
cooldown, allowing six accepted recoveries. Recovery trials extend to 60 steps.
They include future waypoints on the immutable reference (offsets0/3/6/12) and
joint staging targets held for up to16 steps before continuing reference tracking.
The PPO observation reference and final installation target are not replaced.

- `strict`: requires predicted task-cost improvement, as in the original logic.
- `detour`: additionally permits a bounded temporary increase of at most0.12 in
  position-error +0.12*rotation-error cost. It must move at least0.06 rad and gain
  at least0.015m clearance or advance the reference by at least two points with
  a bounded predicted goal-cost change. Waiting alone does not satisfy this rule.

All candidate transitions use the existing physics model and rollback. No RRT
fallback is introduced. These finite-horizon checks are not an unconditional
collision-free guarantee. Additional predictions increase computation before
real steps; simulated time remains paused while planning.

## Development and evaluation

The development bank is the already-inspected low32 plus high16. Their measured
baseline is cached with model/checksum verification. Candidates must not increase
baseline collisions/limits or reduce high16 successes, then maximize total
success and reduce timeouts/waiting. The baseline remains eligible.

Selection is frozen before old64 regression and evaluation on the v23 expanded
test bank. The latter is generated independently by the fixed v23 scope rules.
This study's controller code and parameters are registered before those expanded
results are read; expanded outcomes never select a candidate. If the v23 bank is
not ready, status shows `waiting_for_frozen_expanded_test`.

```bash
python -m pytest -q
python scripts/check_timeout_episode.py
python scripts/run_timeout_study.py \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/timeout_recovery_v24_seed20261113 \
  --workers 2
```

Use the project Conda interpreter and clear inherited PYTHONPATH if needed.
A new run requires a fresh output folder. The runner records stage progress,
all outcomes, recovery events, predicted-step counts and full candidate recordings.
No BC/PPO optimizer is run. The initial pilot is a development check only and
must not be presented as a general success-rate estimate.

## Smooth interactive 3D replay

Replay a recorded successful episode in a PyBullet window:

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/timeout_recovery_v24
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/replay_episode.py \
  /home/johnsonchen/wall-panel-rl/results/diagnostics/timeout_recovery_v24_seed20261113/recordings/expanded_test_selected/1231415.json \
  --fps 30 --playback-speed 0.6 --timing smooth
```

Interpolation is prepared before playback, joint poses are updated in one batch,
and shadows and path drawing are disabled by default. Smooth timing retains every
sample and stretches playback if rendering falls behind. GUI startup calibration
may reduce the requested frame rate; `--fps 60` is also supported.
Recorded obstacle-avoidance waits remain visible. Playback does not run physics,
control, planning, or training, and does not change the recorded result.
Use `--show-path` for a static trajectory overlay or `--shadows` to enable shadows.
`--timing realtime` instead keeps clock speed by dropping late frames.

## On-path vertical weaving scene

`configs/on_path_weaving.yaml` refines the previous successful high/low corridor.
It moves a near-route box just enough to overlap the measured arm or panel guide
by 8 mm, with a maximum anchor translation of 12 cm. Guides include the actual
execution and its stored physical nominal runs. Every physical box must intersect
one guide configuration at its motion anchor. The generator separately reports
how many moving boxes intersect the previous actual execution at its original
timestamps. These checks use box dimensions, not only swept motion regions.
Full motion bounds protect the start, goal, wall, box spacing and the vertical
workspace. The policy reference remains unchanged.
O1's motion phase is aligned with its collision witness on the previous actual
execution, so simply replaying that old trajectory would now collide with O1.
The new execution must adapt to the changed scene.

Generate and physically evaluate the layout with the current PPO plus predictive
control and local recovery:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/adjust_weaving_scene.py --output outputs/on_path_weaving_demo

env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/replay_episode.py outputs/on_path_weaving_demo/recording.json \
  --fps 30 --playback-speed 0.6 --timing smooth
```

Use a new output directory on each run. `--layout-seed` instead enables complete
resampling along the actual execution, using alternating vertical anchor offsets
of 0.14--0.24 m and random motion phases while retaining the robot start and target.
Resampled layouts can fail or use a lateral bypass; their geometry does not
guarantee successful vertical weaving by the current controller.
`--geometry-only` builds and checks the scene without executing the controller.
The output includes the full scene, geometric evidence, actual termination
reason, and measured alternating above/below passages. A layout demonstration
selected for its visual behavior is not an independent success-rate benchmark.

The validated scene is also saved as `demo/on_path_weaving_v25.json`:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/replay_episode.py demo/on_path_weaving_v25.json \
  --fps 30 --playback-speed 0.6 --timing smooth
```
