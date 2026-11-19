"""Generate the requested 18-point handoff from real files and measurements."""
import _bootstrap
import json
from pathlib import Path
import pandas as pd
import yaml


def main():
    root = Path('.')
    config = yaml.safe_load((root / 'configs/base.yaml').read_text())
    table = pd.read_csv(root / 'results/summary/metrics.csv')
    runs = []
    for name in ['ppo_clean', 'ppo_noise']:
        p = root / 'models' / name
        training = json.loads((p / 'training.json').read_text())
        runs.append(f'- `{name}`: {training["actual_additional_timesteps"]:,} additional transitions, '
                    f'{training["total_model_timesteps"]:,} cumulative transitions; '
                    f'primary checkpoint `models/{name}/best/best_model.zip`; '
                    f'final checkpoint `models/{name}/final_model.zip`.')
    rows = []
    for _, row in table.iterrows():
        rows.append(f'| {row["method"]} | {row["evaluation_noise"]} | {int(row["success_count"])} | '
                    f'{100*row["success_mean"]:.1f}% | {100*row["collision_mean"]:.1f}% |')
    text = '''# Final development summary

## 1. Implemented functionality

Complete physical PyBullet simulation, Gymnasium environment, shared position
actuator, centralized collision/limit checks, rigid panel, quaternion pose errors,
noise injection, explicit RRT-Connect and multistart IK, configurable SB3 PPO,
curriculum, paired evaluation, CSV summaries, plots, GUI and headless demos, tests,
reproducible dependency files, documentation and report generation. No essential
placeholder implementations remain.

## 2. Repository structure

```text
assets/robot/kuka_iiwa/     Vendored URDF and meshes
configs/                  Base, clean/noisy train/eval and curriculum YAMLs
src/wall_panel_rl/
  envs/                   Environment, collisions, rotations and sensing noise
  control/                Shared Bullet position controller
  planning/               IK and RRT-Connect
  baselines/              Classical waypoint execution
  rl/                     Reward logging
  utils/                  Config and evaluation rollout/metrics
scripts/                  Train, evaluate, benchmark, plot, demo, diagnostics,
                          smoke test, report and summary generation
 tests/                   Core and physical regression tests
models/                   Actual training artifacts and configurations
results/raw/              Per-episode CSVs and scene/trajectory records
results/summary/          Aggregated real metrics and diagnostic evidence
results/figures/          Generated plots and simulator snapshots
```

## 3. Simulator and robot

PyBullet 3.2.7; seven-revolute-joint KUKA LBR iiwa from bundled pybullet_data,
vendored with its license/attribution. The allowed fallback replaces preferred
UR5 because no verified standalone asset was supplied. Panel dimensions are
0.30 × 0.20 × 0.03 m, mass 0.25 kg. Target is approximately
(0.846762, 0, 0.546459) m, quaternion xyzw (0, .706825, 0, .707388), defined by
configured reference FK. The vertical target has 25 mm wall clearance. Physics
runs at 240 Hz with up to 12 substeps per high-level action.

## 4. Observation space

41 float32 values: normalized q (7), scaled dq (7), relative target position (3),
relative rotation vector divided by pi (3), three obstacle slots (7 each:
relative center, box dimensions and presence). Unused slots are zero. No delay,
raw Euler subtraction, goal-joint oracle, or noisy ground-truth metrics.

## 5. Action space

Seven continuous values in [-1,1]. `delta_q = action * radians(4)`;
`q_des = clip(q_current + delta_q, lower_limits, upper_limits)`.
Attempted joint-limit violations receive a penalty and terminate. Both PPO and
RRT send desired joint positions to the same Bullet position actuator (position
gain .3, velocity gain 1, force 250). Only reset and collision/IK queries teleport.

## 6. Reward

`100*(previous_position_error-position_error)`
`+12*(previous_orientation_error-orientation_error)`
`-.015*sum(action**2)-.05`
`-50*collision-30*joint_limit_violation+100*success`.

Position is meters; orientation is radians. Strict success is <1 cm and <2°,
with no collision/limit event. Success/collision/limit terminate; 160 steps truncate.
Individual reward terms are included in info and TensorBoard.

## 7. PPO hyperparameters

SB3 MlpPolicy, actor/critic two 128-unit ReLU layers; learning rate .0003;
1,024-step rollouts; minibatch 64; 10 epochs; gamma .99; GAE .95; clip .2;
entropy coefficient 0. Fresh precision training uses log_std_init=-1; the first
retained diagnostic used default 0. No VecNormalize. Seed 42, CPU. Five callback
validation episodes every 10,000 additional steps (5,000 in alignment curriculum).
Final variants share one checkpoint and equal budgets, differing only in noise.

## 8. RRT parameters

Extension .22 rad; goal bias .15; max 3,000 iterations; search timeout 5 s;
edge spacing .035 rad; 40 shortcut attempts; 5 mm planning clearance;
IK up to 12 candidates × 200 iterations, 4 mm/1° FK acceptance;
waypoint tolerance .008 rad. Two trees actively route around obstacles.
The physical detour regression uses a blocked straight joint-space path and
checks successful motor-driven execution, not just a geometric plan.

## 9. Noise and fairness

Clean: zero noise. Noisy: independent Gaussian translation SD .005 m/axis and
rotation-vector SD .5°/axis, resampled per observation. RRT uses the first noisy
target for IK/planning once; PPO receives fresh target measurements each step.
Joint/obstacle state stays exact. Every method receives identical benchmark
scene seeds and initial noise samples. Metrics/reward/stopping use true poses.
The benchmark has 100 original-start scenarios per cell and no nearby-start mix.

## 10. Exact training commands

The completed successful curriculum and final branches used the following
commands. Existing run names refuse overwrite; rename them when reproducing.
Earlier diagnostic attempts are retained but are not needed for this pipeline.

```bash
python scripts/train_ppo.py --config configs/base.yaml --seed 42 \\
  --total-timesteps 50000 --run-name phase1_precision
python scripts/train_ppo.py --config configs/phase1_alignment.yaml --seed 42 \\
  --resume models/phase1_precision/final_model.zip \\
  --total-timesteps 30000 --run-name phase1_alignment
python scripts/train_ppo.py --config configs/base.yaml --seed 42 \\
  --resume models/phase1_alignment/final_model.zip \\
  --total-timesteps 50000 --run-name phase1_restored
python scripts/train_ppo.py --config configs/stage_b_curriculum.yaml --seed 42 \\
  --resume models/phase1_restored/final_model.zip --zero-obstacle-inputs \\
  --total-timesteps 30000 --run-name phase2_transfer
python scripts/train_ppo.py --config configs/train_clean.yaml --seed 42 \\
  --resume models/phase2_transfer/final_model.zip \\
  --total-timesteps 100000 --run-name ppo_clean
python scripts/train_ppo.py --config configs/train_noise.yaml --seed 42 \\
  --resume models/phase2_transfer/final_model.zip \\
  --total-timesteps 100000 --run-name ppo_noise
```

SB3 rounds budgets to full rollouts. Exact completed counts:

{RUNS}

## 11. Exact evaluation commands

```bash
python scripts/evaluate.py --config configs/eval_clean.yaml \\
  --clean-model models/ppo_clean/best/best_model.zip \\
  --noise-model models/ppo_noise/best/best_model.zip --run-id main_clean
python scripts/evaluate.py --config configs/eval_noise.yaml \\
  --clean-model models/ppo_clean/best/best_model.zip \\
  --noise-model models/ppo_noise/best/best_model.zip --run-id main_noise
python scripts/plot_results.py
python scripts/write_report.py
python scripts/write_development_summary.py
```

Checkpoint selection uses callback validation return before the benchmark.
Both primary run IDs already exist; use new names to reproduce without overwriting.

## 12. Exact demo commands

```bash
python scripts/demo.py --method rrt
python scripts/demo.py --method ppo --model models/ppo_clean/best/best_model.zip
python scripts/demo.py --method ppo --model models/ppo_noise/best/best_model.zip \\
  --config configs/eval_noise.yaml
python scripts/demo.py --method rrt --headless
```

The Linux host required `LIBGL_ALWAYS_SOFTWARE=1` and
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` for its Mesa/C++ runtime.
The graphical demo was actually executed successfully; see GUI screenshots.

## 13. Validation results

Created the pinned Conda environment under `.conda-env`; smoke test passed stages
A/B/C, Gymnasium/SB3 checks, IK/RRT and physical alignment. `pip check` passed.
{TEST_RESULT}
A ROS-sourced shell must unset PYTHONPATH to avoid unrelated ROS pytest plugins.
A physical blocked-path detour passes. Both noisy and clean primary evaluations
completed, with identical scenario records across methods/conditions.

| Method | Observation | Episodes | Success | Collision |
|---|---|---:|---:|---:|
{RESULT_ROWS}

## 14. Generated model paths

Primary: `models/ppo_clean/best/best_model.zip` and
`models/ppo_noise/best/best_model.zip`. Final policies are in their respective
`final_model.zip` files. Each run includes config.yaml, provenance.json,
training.json, train.monitor.csv and evaluations.npz. Periodic checkpoints and
TensorBoard event files are excluded from Git. Diagnostic models are retained.

## 15. Generated CSV paths

- `results/raw/main_clean/episodes.csv` (300 episodes)
- `results/raw/main_noise/episodes.csv` (300 episodes)
- `results/summary/metrics.csv` (continuous means, SDs, counts)
- `results/summary/rates_percent.csv` (binary percentages)

Each primary raw folder includes scenarios.json, config.yaml, provenance.json
and representative trajectory JSONs. All numbers derive from execution.

## 16. Generated figure paths

Under `results/figures/`: success.png, collision.png, position_error_m.png,
orientation_error_deg.png, steps.png, execution_time_s.png, planning_time_s.png,
joint_path_length_rad.png, training_reward.png, training_success.png,
trajectories.png, plus actual simulator demo images. Regenerate plots from CSV
with scripts/plot_results.py; learning plots read recorded Monitor/NPZ files.

## 17. Unresolved limitations

One training seed; finite training/evaluation budgets; no convergence guarantee;
simulator-only, simplified rigid panel and sparse boxes; fixed true target;
no sensor filtering, real hardware, wall-contact installation or delay.
A fixed target allows goal-configuration memorization, so noise robustness does
not prove transfer to new targets. RRT edge checks are discrete and motor tracking
can still collide. Timings depend on hardware/load. RRT uses one target sample,
whereas PPO receives a new sample each action. Failures stay in the results.

## 18. Differences from the specification and reasons

Allowed KUKA fallback uses seven actions rather than six. Target pose is generated
once by real FK from a YAML reference posture to guarantee reachability. Added a
near-target reset curriculum after documented learning diagnostics, then a 25%
near / 75% original mixture for obstacle training; evaluation never uses it.
Zeroed previously unused obstacle-input columns once during phase transfer to
preserve the learned function, retaining their trainability. Added 5 mm planning
clearance after transient physical contact exposed tracking risk. Optional
reject-only baseline, large-panel experiments and extra noise sweeps are omitted
in favor of the required six-cell experiment. Full rationale: DECISIONS.md.
'''
    test_output = (root/'results/summary/pytest.txt').read_text().strip().splitlines()[-1]
    selection = json.loads((root/'results/summary/model_selection.json').read_text())
    runs.append(f"- Evaluated best checkpoints: {selection['clean']['selected_model_timesteps']:,} timesteps for PPO-Clean and {selection['noise']['selected_model_timesteps']:,} for PPO-Noise, selected before benchmark evaluation.")
    text = text.replace('{RUNS}', '\n'.join(runs)).replace('{RESULT_ROWS}', '\n'.join(rows))
    text = text.replace('{TEST_RESULT}', f'Final pytest output: `{test_output}`.')
    text = text.replace('\n tests/', '\ntests/')
    (root/'DEVELOPMENT_SUMMARY.md').write_text(text)
    print('Wrote DEVELOPMENT_SUMMARY.md from recorded artifacts')


if __name__ == '__main__':
    main()
