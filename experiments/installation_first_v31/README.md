# Installation-first adaptation v31

The default `configs/runtime.json` restores the v29 controller and selected
PPO-32768 checkpoint. The closer v30 geometry is retained: lateral variation
0.10 m, XY jitter 0.015 m, 2--5 dynamic obstacles. Dimensions, speed range,
physical floor/ceiling, 240-step budget and installation tolerances are unchanged.
Earlier experiments and checkpoints are preserved.

## Guarded timeout repairs

`configs/installation_runtime.json` is the explicit experimental profile, based
on the old controller. It adds no proactive vertical-passage objective.
The old recovery search is tried first. Nonterminal detours that leave a stale
reference substantially farther away and increase remaining joint goal distance
are rejected. If it returns no admissible plan, bounded waits of 0/4/8/12 steps and nearby forward reference rejoin candidates
considered. Every candidate is physically simulated and rolled back, with the
same moving obstacles, collision rule, limits and reference path.
Near the installation pose or deadline, a finish is accepted only after a full
physical forecast reaches actual installation success. Local IK may find another
joint configuration for the same observed panel target; success tolerances are
never relaxed. This remains PPO + prediction + local recovery, not pure PPO.

Real recordings include commands, reference indices/distances and pose errors.
`timeout_diagnostics.py` separates waiting, stale reference, terminal precision
and insufficient progress. Missing telemetry in older recordings is explicitly
reported; reference history is not invented.

## Demonstrations and training

The 24 training and 8 validation cases are the v30 closer scene banks. Eight
fresh independent test cases are generated geometrically from disjoint seed
blocks and opened for evaluation only after model/controller selection.
Teacher quotas are 2/4/4/2 cases for 2/3/4/5 obstacles. A failed training scene
may retry once with another known route/speed/phase sample. Every attempt is
saved; only physically successful episodes supply BC labels. Multi-obstacle and
terminal-state examples receive modest sampling weights without duplicating
transition counts. Failed and test trajectories never enter BC.

Vertical crossing, alternating-height and vertical-potential reward weights
are all zero. Installation and collision rewards retain their original values.
Checkpoint selection prioritizes **installation success, then collision/joint
failures, then task error**. The old controller/model bundle remains selectable.

The budget is up to **500 BC updates**, checked at 250/500. Only a BC validation
score better than the best initialization bundle unlocks **8,192 PPO steps**.
If this gate fails, PPO is skipped and the better bundle is independently tested.
PPO buffers always contain raw PPO-sampled actions and original log probabilities.

## Output

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/installation_first_v31_seed20261118/
  pipeline_status.json
  diagnosis/old_summary.json
  diagnosis/installation_training/
  scenes/provenance.json
  training/status.json
  training/teacher_episodes.json
  training/recordings/teachers/
  training/bc_gate.json
  training/selection.json
  training/selected_runtime.json
  training/models/selected_model.zip
  training/summary.json
```

The selected runtime and model form one bundle; use both when executing fresh
episodes. Saved recordings are self-contained and can be replayed with:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python /home/johnsonchen/wall-panel-rl/experiments/installation_first_v31/scripts/replay_episode.py /absolute/path/to/recording.json --fps 30 --playback-speed 0.6 --timing smooth
```

Checks use the existing conda environment:

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/installation_first_v31
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -m pytest -q
```

Playback uses recorded poses with no planning/physics and exports no video.
