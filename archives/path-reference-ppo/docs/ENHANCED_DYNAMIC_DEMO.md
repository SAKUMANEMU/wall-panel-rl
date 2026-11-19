# Larger and faster four-obstacle motion

`demo_four_dynamic_faster.py` loads the selected v9 fast02 candidate and
`configs/four_dynamic_faster.yaml` by default. The earlier slow demo/config and
all training banks remain available for comparison.

The current motion profile uses a15cm Euclidean offset bound around each anchor
and0.10m/s maximum vector speed. The previous enhanced setting was10cm/0.05m/s;
it is retained in `configs/four_dynamic_faster_10cm.yaml`. The original slow
setting remains5cm/0.02m/s. These are norm bounds, not independent
per-axis limits. Four different-size obstacles still move smoothly inXYZ, with
fresh random starts, anchor placements and initial phases by default. The goal,
reward/termination criteria, collision guards, zero noise and0.6x playback are
preserved. Larger motion envelopes can reject more layouts; the sampling limit
is100 attempts, with the same endpoint/wall/spacing guards. Valid layouts are
rarer with the larger envelopes, so fresh-scene generation can take longer.

This is an inference/animation change, not a new training run. The selected
model was trained with the previous motion settings. Old success-rate results
do not establish performance under these larger/faster trajectories.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_dynamic_faster.py --controller ppo
```

Use `--controller mixture` for50% RRT /50% PPO episode ownership, or omit the
controller option to use that default. RRT still predicts known trajectories
on the simulation clock and can traverse their swept regions at safe times.
Planning/setup rendering remains hidden. Planning does not advance simulation
time. `--seed` fixes a scene/motion sample; omission requests a fresh random one.

The previous10cm/0.05m/s eight-case physical functional check returns2 RRT successes and6 PPO
episodes(5 successes/1 collision). This small run is not a rate estimate.
Measured maximum anchor offset0.0992843m and maximum sampled speed0.0418144m/s
respect the0.10m/0.05m/s analytic limits. All boxes move in all three axes.
Eight construction candidates fail geometry and one fails the obstacle-free
nominal reference; these are logged before executing dynamic controllers.
No actual dynamic episode outcome is filtered, no optimizer update is made,
and source weights/update counters remain unchanged.

Previous report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_faster_preflight_v1_seed61/README.md`.
The report's initial fixed v5/slow-bound prose was corrected from its actual
protocol after execution; numeric outcomes and executed code archives remain
unchanged. Future report generation reads model/bounds from the profile.

Current15cm/0.10m/s functional report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_faster_preflight_v2_seed61/README.md`.

The current eight-case check yields4 successes/3 collisions/1timeout: RRT2/2
success, PPO2 successes/3 collisions/1timeout. Maximum measured offset is
0.14869956m and speed0.08974718m/s. All boxes move inXYZ; source weights and
update counters are unchanged. These small samples do not estimate general
success rate. The old model has not been retrained for the increased motion.
Constructing the eight valid cases rejects132 geometry candidates and4
unavailable obstacle-free references. Scene-generation waiting time is not
model training.

For a quicker replay of a random saved case (finite bank of eight scenes), add:
`--scene-bank results/diagnostics/four_dynamic_faster_preflight_v2_seed61/cases.json`.
Omitting that flag keeps fresh random scene generation.

To restore the previous10cm motion in the demo, add:
`--motion-config configs/four_dynamic_faster_10cm.yaml`.
