# Variable dynamic obstacle weaving v26

This isolated experiment samples 2, 3, 4 or 5 dynamic obstacles with equal
probability at each training reset. The policy keeps its 77-dimensional
observation and 7-dimensional joint action; unused obstacle slots are zero-padded.
The previous experiments and initialization checkpoint remain unchanged.

Each differently sized box intersects a configuration of a frozen obstacle-free
physical arm/panel guide at its motion anchor. Anchors alternate above and below
the guide. One box per generated scene crosses its guide collision witness at the
same simulation timestamp. Three-axis motion has a vector amplitude bound of
0.18 m and speed bound of 0.12 m/s. Swept boxes protect the endpoints, wall,
spacing and workspace during generation; execution uses actual moving boxes.
Generation rejects geometry failures, never candidate task failures.

The scene bank has 48 training, 16 validation and 16 independent test cases,
balanced across obstacle counts. Training adds small random phase jitter within
the same protected swept bounds. No static scenes are sampled. A one-time bonus
rewards passing on the intended above/below side with whole-panel clearance;
original installation success and collision termination remain unchanged.
Actual alternating passages are measured separately from the reward events.

The pipeline collects four enhanced-controller teacher episodes, retaining only
successful episodes for separate BC warm-up. Teacher actions never enter PPO's
on-policy buffer. PPO then runs 65,536 additional steps, with an 8,192-step pilot
check and checkpoints every 8,192 steps. The tracking prior remains frozen;
avoidance adapter, gate, residual and critic are trainable. Checkpoint selection
uses validation safety, success and measured weaving. The independent test bank
is opened after selection. Episodes have at most 240 physical control steps;
the number of episodes depends on their termination lengths.

Use the project Conda interpreter from this directory:

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/variable_weaving_v26
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -m pytest -q
```

The already-started run is stored at:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/variable_weaving_v26_seed20261115/
  scenes/                    # Frozen scene banks and generation attempts
  training/status.json       # Current phase, steps and episode outcome counts
  training.log               # Background process log
  training/teacher_episodes.json
  training/training_episodes.jsonl
  training/checkpoint_history.json
  training/validation_curves.png
  training/models/
  training/evaluations/
  training/recordings/        # Final independent-test recordings
```

To reproduce with fresh output directories:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/generate_variable_cases.py --output outputs/variable_scenes
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/run_variable_training.py --scenes outputs/variable_scenes --output outputs/variable_training
```

Edit `configs/variable_weaving.yaml` for motion/layout/training parameters. Once
a recording exists, use `scripts/replay_episode.py RECORDING.json --fps 30
--playback-speed 0.6 --timing smooth` for interactive 3D playback. No video export
is required. A valid layout or a completed training budget does not guarantee
successful weaving; consult actual outcome counts and recorded trajectories.
