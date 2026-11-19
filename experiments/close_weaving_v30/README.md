# Closer vertical-passage pilot v30

This isolated version starts from the v29 selected PPO-32768 checkpoint. Earlier
models and experiments are preserved. Improved success or weaving is an objective,
not an established result; consult the live measurements below.

## Environment and execution

Anchor lateral variation is reduced from 0.15 to **0.10 m**, XY jitter from
0.025 to **0.015 m**, and alternating vertical offsets from 0.10--0.20 to
**0.10--0.18 m**. Sizes, speeds (0.04--0.22 m/s), motion amplitude, obstacle spacing
and protected endpoints are unchanged. Floor z=0.10 m and ceiling z=1.10 m remain
physical collision boundaries for every moving link and the whole panel.

`VerticalPassageController` proposes local joint targets using IK for an intended
above/below passage. IK is only a query: execution always uses the shared actuator
and physics. Candidates stage height, then pass the next nearby moving box. Plans
must predict a real collision-free whole-panel passage with forward movement,
and are rechecked while executing. The original reference path remains unchanged.
Bounded temporary recovery detours are enabled. Stall recovery retains the v29
minimum step (40) after development probes found earlier intervention unhelpful.
Late in an episode, physically successful full finish forecasts can be committed
without waiting for another stall. Vertical candidates reserve actuator time for
installation; the 240-step episode budget is unchanged.
No global RRT fallback is introduced. This is **PPO + predictive control + local
recovery**, not pure PPO. The same controller supplies successful BC demonstrations.
PPO buffers still use raw PPO actions and their original log probabilities.

## Pilot and measurement

Fresh seed blocks provide **24 training, 8 validation and 8 test scenes**, balanced
across 2--5 dynamic obstacles. Matched development probes compare old/new control
on the first two- and four-obstacle training scenes with identical weights and
geometry. The pilot then uses up to **500 BC updates and 8,192 PPO steps**.
An old-controller baseline is also measured on the same eight new validation
scenes, separately from the new controller's initialization baseline.
Only validation selects models; independent testing follows selection.

Training adds a bounded vertical potential via `0.99 * next - previous`, with
zero terminal potential. One actual intended passage pays 6 raw units; an ordered
opposite-side passage with at least 7 cm height change pays 4 extra. Each slot can
pay once. Collision penalties and successful-installation conditions are unchanged.

The stricter recorded trajectory metric is unchanged: the whole panel clears the
box by 1 cm within a 24 cm XY radius, in order, with alternating sides and at
least 7 cm of actual height change. Training crossing counts are separate.

## Settings and results

- `configs/variable_weaving.yaml`: layout, reward and pilot budget.
- `configs/runtime.json`: local vertical and timeout-recovery settings.
- `src/wall_panel_rl/rl/vertical_passage.py`: physical candidate prediction.
- `configs/v29_baseline_controller.json`: frozen development comparator.
- `environment.yml`: dependencies; the existing project conda environment works.

Live artifacts are under:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/close_weaving_v30_seed20261118/
  experiment_manifest.json
  scenes/status.json
  development_probes/status.json
  development_probes/results.json
  training/status.json
  training/models/
  training/evaluations/
  training/recordings/
```

Checks and playback (use an actual absolute recording path):

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/close_weaving_v30
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -m pytest -q
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/replay_episode.py /absolute/path/to/recording.json --fps 30 --playback-speed 0.6 --timing smooth
```

Replay uses saved poses, runs no planning and exports no video.
