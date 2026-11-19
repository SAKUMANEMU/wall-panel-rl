# Conditional recovery PPO snapshot — 2026-10-30

Branch: `release/conditional-recovery-ppo-20261030`.
This branch captures the current local implementation, training data, saved
models, tests and experiment reports. It is based on the earlier path-reference
BC branch; the repository's existing main and other branch tips are preserved.

## Install and demonstrate

```bash
git clone --branch release/conditional-recovery-ppo-20261030 --single-branch https://github.com/SAKUMANEMU/wall-panel-rl.git wall-panel-rl-conditional
cd wall-panel-rl-conditional
conda env create -f environment.yml
conda activate wall-panel-rl
unset PYTHONPATH
python scripts/demo_autonomous_ppo.py --headless --seed 1000000
python scripts/demo_autonomous_ppo.py --seed 1000000 --playback-speed 0.6
```

The GUI demo requires a local graphical display. The headless command verifies
the same policy/environment without opening a window. For a case that exercises
conditional recovery, run:

```bash
python scripts/demo_autonomous_ppo.py --seed 900002 \
  --scene-bank data/staged_avoidance_v3_seed61/one_static/validation_staged.json \
  --playback-speed 0.6
```

## Current selected system

- Seven-joint KUKA iiwa with a rigid small panel; 77 features and seven actions.
- BC tracking prior plus a learned PPO residual. The saved obstacle-free route
  supplies local references; runtime actions come from the policy.
- One static obstacle, zero measurement noise; five obstacle information slots
  remain in the observation. Dynamic-obstacle experiments are historical, not
  the task underlying the current success rates.
- Conditional recovery may advance a stale non-final reference after verified
  stagnation. It does not directly overwrite policy actions.
- Model: `results/diagnostics/conditional_ppo_adaptation_v3_seed61/selected_model.zip`.
  Keep it with `controller_config.json` / `manifest.json` in that directory.
  The model zip alone does not encode the conditional environment rule.

## Results and selection

| Checkpoint | Validation success | Collision | Timeout |
|---|---:|---:|---:|
| Conditional baseline | 18/20 | 2 | 0 |
| Additional 5k PPO | 16/20 | 1 | 3 |
| Additional 10k PPO | 17/20 | 0 | 3 |

Both updated checks meet the minimum stage thresholds, but neither passes strict
non-regression selection. The retained checkpoint therefore has **zero selected
additional steps**, despite completing the full 10k experiment. It preserves the
source weights with the integrated conditional controller. The final unselected
candidate is saved separately as `final_candidate.zip`.

The retained system reproduces 282 successes, 14 collisions and four timeouts on
300 already-seen regression scenes. This is not new independent evidence or proof
of PPO learning gains. During the 10k stochastic training transitions, only one
conditional recovery was attempted and none advanced the reference. Training
completed 135 episodes plus 264 transitions in partial episodes.

Latest report directory: `results/diagnostics/conditional_ppo_adaptation_v3_seed61/`.
Read `README.md`, `summary.json`, `validation_curve.csv`, `overview.png` and
`verification.json`. Other report directories preserve preceding experiments.

## Training and historical path provenance

Latest entry point: `scripts/train_conditional_ppo.py`.
Latest config: `configs/conditional_ppo_adaptation.yaml`.

```bash
# Validate inputs only; does not train:
python scripts/train_conditional_ppo.py
```

Training requires explicit `--run` and refuses existing output directories. For
a new experiment, create a new config with unique `output_root` and `report_root`;
preserve the archived reports. The latest config uses repository-relative paths.

Historical configurations, report manifests, tests and summarization scripts may
retain `/home/johnsonchen/wall-panel-rl` paths from the original workstation.
These records are preserved for provenance. To rerun an older study elsewhere,
make a separate configuration and adapt its paths; do not rewrite archived model
hashes or report data. The current demo uses repository-relative defaults.

## Included files

Before publication, all 115 project tests passed. The repository-local selected
model also reproduced the default headless demo's 65-step success.

Source, configs, documentation, tests, training data, exported models, Monitor
records and reports are included. Local conda/virtual environments, caches,
Jupyter checkpoint copies, ignored periodic model checkpoints, TensorBoard files
and logs are excluded by `.gitignore`. They are not needed for the included demo.
The abandoned separate branch reproduction is not part of this snapshot.
