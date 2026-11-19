# V32: slower dynamic-obstacle adaptation

This isolated experiment starts from the selected V31 PPO model. All scenes have
2--5 moving obstacles. Travel speed is sampled independently per obstacle and
episode in **0.04--0.18 m/s**, down from 0.04--0.22 m/s. Motion extent, obstacle
sizes, construction-site collisions, 240-step budget and installation tolerances
(1 cm, 2 degrees) are preserved.

## Integration and timeout changes

The 167-feature observation and seven joint-increment actions are unchanged.
At a stalled approach, predictive execution first tries a short, collision-checked
return to the immutable reference before unrestricted local detours. The wait
candidates are 0/2/4 physical steps. Recovery can start at step 32, rather than 40.
A strengthened guard rejects nonterminal detours that strand an unchanged
reference far away, even if Cartesian target distance happens to decrease.
The V31 controller is retained separately as `configs/source_controller.json`.

Only successful training demonstrations supply BC labels. BC and on-policy PPO
remain separate: PPO stores its sampled actions and original log probabilities;
expert actions are never substituted into its rollout buffer. PPO exploration
standard deviation is 0.03, learning rate 3e-6, with 16,384 adaptation steps and
validation every 8,192 steps. The original tracking prior remains frozen.
A PPO adaptation trial is allowed even if BC does not improve validation; the
initialization remains eligible for final selection.

## Protocol

- Training: 32 new scenes, eight per obstacle count.
- Validation: 16 new scenes, four per count, used for checkpoint selection.
- Independent test: 40 new scenes, ten per count, opened after selection freezes.
- Teachers: 4/4/6/6 training scenes for 2/3/4/5 obstacles, at most one motion retry.
- BC: 500 optimizer updates when at least 64 successful transitions are available.
- Final comparison: selected bundle and original V31 bundle on the same 40
  slower test scenes, 80 physical test episodes. Every outcome is retained.
- Results describe PPO plus predictive execution and recovery, not pure PPO.

The earlier V31 43.75% result used different, faster scenes. Comparing that
number directly with V32 does not isolate improvement caused by learning.

## Run

From this directory, using the existing project environment:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/run_site_pipeline.py \
  --config configs/variable_weaving.yaml \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/slower_progress_v32_manual
```

Choose a new output directory. No ROS/Gazebo server or GPU is required. For a
fresh machine, create the environment using `environment.yml` first. Use the
project's existing pinned environment for identical dependency versions.

## Outputs

The launched run is `results/diagnostics/slower_progress_v32_seed20261119` under
the main repository. `pipeline_status.json` tracks the whole pipeline;
`scenes/status.json` tracks generation; `training/status.json` tracks teacher
collection, BC, PPO and evaluation. Models, selection metadata, CSV outcomes,
Markdown results and plots are saved under `training/`. Successful independent
test replays are saved under `training/recordings/independent_test_enhanced/`.

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/replay_episode.py /absolute/path/to/a/saved/success.json \
  --fps 30 --playback-speed 0.6 --timing smooth
```

Replays use saved physical poses and do not rerun planning or training. The
`checks_site/` files and `configs/legacy_regression_runtime.json` are historical
physical regression fixtures; they are excluded from formal V32 statistics.
