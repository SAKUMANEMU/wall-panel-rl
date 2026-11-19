# Current complete project snapshot: V32

This release branch preserves the main project's history and adds the local
experimental workspaces, models, datasets, reports and saved demo trajectories.
The active experiment is `experiments/slower_progress_v32/`; the root `src/` and
older README sections describe earlier versions.

## Run the current version

From a fresh clone of this branch:

```bash
conda env create -f environment.yml
conda activate wall-panel-rl
unset PYTHONPATH
cd experiments/slower_progress_v32
python -m pytest -q tests/test_slower_progress.py tests/test_installation_first.py \
  tests/test_known_routes.py tests/test_construction_site.py \
  tests/test_expanded_verification.py tests/test_episode_replay.py \
  tests/test_resume_known_training.py
```

No Gazebo/ROS or CUDA setup is required. PyBullet GUI playback needs a graphical
desktop. CPU training works headlessly. Python and dependency versions are pinned
in `environment.yml`.

The active system has 167 observations, seven joint-increment actions, 2--5
moving obstacles, 0.04--0.18 m/s randomly sampled travel speeds, and predictive
control plus local reference recovery. Installation tolerances are 1 cm and
2 degrees, with a 240-step episode budget. It is not pure PPO execution.

## View a successful current 3D demo

From the repository root, select a saved successful independent test automatically:

```bash
DEMO_FILE=$(python - <<'PYTHON'
import json
from pathlib import Path
folder = Path('results/diagnostics/slower_progress_v32_seed20261119/training/recordings/independent_test_enhanced')
for path in sorted(folder.glob('*.json')):
    if json.loads(path.read_text())['result']['reason'] == 'success':
        print(path.resolve())
        break
else:
    raise SystemExit('No saved success recording found')
PYTHON
)
python experiments/slower_progress_v32/scripts/replay_episode.py "$DEMO_FILE" \
  --fps 30 --playback-speed 0.6 --timing smooth
```

Playback uses recorded physical poses, without running the model or planning.

## Train and validate a new run

From the repository root, choose a new output name:

```bash
python experiments/slower_progress_v32/scripts/run_site_pipeline.py \
  --config experiments/slower_progress_v32/configs/variable_weaving.yaml \
  --output results/diagnostics/slower_progress_v32_new_run
```

The pipeline generates disjoint train/validation/test scenes, collects successful
BC demonstrations, performs 500 BC updates and 16,384 on-policy PPO steps,
selects only on validation, then compares the selected bundle and source V31
bundle on the same held-out scenes. Existing output directories are not reused.
The starting source model is `experiments/slower_progress_v32/models/current_ppo.zip`
(the frozen V31 initialization). The selected trained V32 bundle is stored below.

## Actual completed results

Run directory: `results/diagnostics/slower_progress_v32_seed20261119/training/`.

| Bundle on the same 40 slowed test scenes | Success | Collision | Timeout | Success rate |
|---|---:|---:|---:|---:|
| V32 selected | 22 | 7 | 11 | 55% |
| V31 source | 20 | 4 | 16 | 50% |

V32 reduces timeouts but has three additional collisions. This small test does
not establish a reliable overall improvement. Raw stochastic PPO training had
0 successful episodes out of 198; the 55% figure belongs to enhanced execution.
The chosen checkpoint is `ppo_8192`, selected from validation before testing;
16,384 PPO adaptation steps were executed in total.

- Report: `training/RESULTS.md`
- Machine-readable outcomes: `training/paired_test.json`, `training/test_outcomes.csv`
- Curves: `training/validation_curves.png`, `training/test_outcomes.png`
- Model: `training/models/selected_model.zip`
- Required paired controller configuration: `training/selected_runtime.json`
- Replay trajectories: `training/recordings/independent_test_enhanced/`

The previous V31 expanded study is retained under
`results/diagnostics/installation_first_v31_expanded_seed20261118/`.
Its 43.75% success rate used different, faster scenes and is not a matched
learning-effect comparison with V32.

## Snapshot layout and scope

- `experiments/`: isolated experimental source trees and their assets/configs.
- `results/`: full local experiment outcomes, scenes, model snapshots and replays.
- `models/`: earlier models and checkpoints.
- `Learning_Agent_CR/`, `Learning_Agent_CR.zip`: course submission material.
- `archives/`: earlier linked worktree and reproduced branch, flattened to
  ordinary files; see `archives/README.md` for original path mappings.
- `UPLOAD_MANIFEST.json`: file hashes and snapshot coverage.

Conda/virtual environments, Git internal metadata, private tool settings and
Python caches are excluded. Dependency specifications are included instead.
Historical status/PID/log files and absolute paths are preserved as experiment
records; a status marked running in an old record does not mean it is live now.
The snapshot is sizeable because it includes historical datasets and checkpoints.
