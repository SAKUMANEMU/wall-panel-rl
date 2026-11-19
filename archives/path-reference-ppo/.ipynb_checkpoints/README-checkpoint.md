# Reinforcement Learning for Collision-Aware Robotic Wall Panel Installation

Can a PPO joint-space policy learn collision-aware panel manipulation and maintain
accurate alignment under target measurement noise, compared with sampling-based
planning and feedback control?

This repository contains a runnable PyBullet/Gymnasium environment, SB3 PPO
training, a project-implemented bidirectional RRT-Connect, fixed-seed evaluation,
CSV summaries, plots and graphical demos. The seven-joint KUKA iiwa holds a
0.30 × 0.20 × 0.03 m rigid panel. Success requires <1 cm position error and <2°
orientation error, without collision. There is **no observation delay**.

## Install

From this repository root:

```bash
conda env create -f environment.yml
conda activate wall-panel-rl
# If your shell previously sourced ROS, clear its unrelated Python plugins:
unset PYTHONPATH
python scripts/smoke_test.py
pytest -q
```

Python and direct library versions in `environment.yml` are the actual tested
versions. CPU PyTorch is pinned; no CUDA or GPU is required. A pip alternative
using Python 3.11.5 is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python -m pytest -q
```

Scripts add `src/` to their import path; an editable install is unnecessary.
Run commands from the repository root. The smoke test verifies stages A/B/C,
SB3's API checker, IK, RRT and physical installation. No full training occurs.

## Methods and configuration

The original static PPO uses a 41-element observation with normalized q (7), dq (7), relative target
position (3), relative rotation vector (3), and three obstacle slots (7 each:
relative center, dimensions, presence). Missing slots are zero. A two-layer
128-unit ReLU MLP outputs seven normalized increments. Each action produces
`q_des = clip(q + action * radians(4), joint_limits)`. Attempted limit violations
are penalized and terminate; clipping does not silently hide them.

The current dynamic retraining uses 70 observations with five obstacle velocity
slots; see [Dynamic PPO retraining](#dynamic-ppo-retraining-with-70-observations).
Models must be paired with a configuration having the same observation shape.

Both methods execute using the same Bullet joint position controller, 240 Hz
physics and 12 physics steps per high-level decision. The panel is attached by a
fixed constraint. Collisions include robot/panel versus obstacles and wall,
nonadjacent self-collision, and panel versus non-gripped links.

The reward is

```text
100 * (previous_position_error - position_error)
+ 12 * (previous_orientation_error - orientation_error)
- 0.015 * sum(action**2) - 0.05
- 50 * collision - 30 * joint_limit_violation + 100 * success
```

Orientation errors in the reward are radians. Terminal events use ground truth.
The controller observation alone has target measurement noise: independently
sampled Gaussian translation (5 mm per axis) and rotation-vector (0.5° per axis).
The clean condition uses zero noise. Joint state and obstacle geometry are exact.

RRT takes the **same first noisy target observation**, converts panel pose to
end-effector pose, solves multistart IK, then grows and connects two joint-space
trees. Edges are checked every 0.035 rad; configurations have a 5 mm planning
clearance. It follows the resulting waypoints through the common controller.
It plans once, rather than updating its target after every observation. PPO gets
fresh target observations each step. This experimental distinction is deliberate
and documented in `DECISIONS.md`; neither receives a secretly perfect target.
A noisy planned goal can lie outside the true success tolerance.

```mermaid
flowchart TD
    T[Observed target pose: clean or noisy] --> I[Panel-to-EE transform and IK]
    T --> O[Joint state, pose error, obstacle observation]
    I --> R[RRT-Connect: checked joint path]
    O --> P[PPO MLP: joint increments]
    R --> C[Shared joint position controller]
    P --> C
    C --> B[PyBullet robot and attached panel]
    B --> E[Ground-truth collision and pose evaluation]
```

All parameters are in `configs/base.yaml`; other YAMLs inherit and override it.
Stages: A = no boxes; B = one fixed sparse box; C = 1–3 randomized sparse boxes.
Start/goal overlap is rejected. Sparse sampling does not mathematically guarantee
a connecting path. Failures stay in the reported denominator.

## Training

Each run requires a unique name and saves resolved config, provenance, Monitor
CSV, TensorBoard logs, periodic checkpoints, best evaluation model and final model.
No VecNormalize is used. SB3 rounds timesteps up to complete rollout buffers.

```bash
python scripts/train_ppo.py --config configs/base.yaml \
  --total-timesteps 50000 --run-name my_phase1 --seed 42
python scripts/train_ppo.py --config configs/phase1_alignment.yaml \
  --resume models/my_phase1/final_model.zip \
  --total-timesteps 30000 --run-name my_alignment --seed 42
python scripts/train_ppo.py --config configs/base.yaml \
  --resume models/my_alignment/final_model.zip \
  --total-timesteps 50000 --run-name my_restored --seed 42
python scripts/train_ppo.py --config configs/stage_b_curriculum.yaml \
  --resume models/my_restored/final_model.zip --zero-obstacle-inputs \
  --total-timesteps 30000 --run-name my_phase2 --seed 42
python scripts/train_ppo.py --config configs/train_clean.yaml \
  --resume models/my_phase2/final_model.zip \
  --total-timesteps 100000 --run-name my_clean --seed 42
python scripts/train_ppo.py --config configs/train_noise.yaml \
  --resume models/my_phase2/final_model.zip \
  --total-timesteps 100000 --run-name my_noise --seed 42
```

The alignment run uses nearby initial states. Obstacle training mixes 25% nearby
and 75% original starts to retain precision learning; evaluation always disables
this mixture. The measured unsuccessful earlier attempts are retained separately. The one-time
`--zero-obstacle-inputs` transfer flag clears weights on features that were always
zero in Stage A, preserving the learned policy when obstacle slots become active.
Those weights remain trainable; do not apply this flag to the final branches.

Both final variants should branch from the same no-noise curriculum checkpoint
with equal additional budgets. Inspect learning first; these commands alone do
not guarantee convergence. `--device cpu` is the default; `--render` enables GUI.
`tensorboard --logdir models` displays training and reward-component logs.

## Evaluation and plots

The primary comparison uses each run’s best callback-validation checkpoint,
selected by mean return before the fixed benchmark is run. Final checkpoints
are also retained. Use the supplied models, or substitute paths to your own:

```bash
python scripts/evaluate.py --config configs/eval_clean.yaml \
  --clean-model models/ppo_clean/best/best_model.zip \
  --noise-model models/ppo_noise/best/best_model.zip --run-id my_eval_clean
python scripts/evaluate.py --config configs/eval_noise.yaml \
  --clean-model models/ppo_clean/best/best_model.zip \
  --noise-model models/ppo_noise/best/best_model.zip --run-id my_eval_noise
python scripts/plot_results.py --inputs \
  results/raw/my_eval_clean/episodes.csv results/raw/my_eval_noise/episodes.csv
```

Each command defaults to 100 scenarios (seeds 10000–10099) per method, with noise
seeds offset by 1,000,000. Scene and noise RNGs are separate. The script asserts
that all methods received identical initial configurations, obstacles and targets,
and saves these in `scenarios.json`. `--episodes N` changes the count; `--methods
rrt` evaluates only the planner. Run IDs cannot overwrite existing raw results.

Alternatively, `python scripts/benchmark.py --prefix my_benchmark` runs both
conditions and plots all six cells. Final metrics always use true simulated poses.
Continuous metrics include mean, sample standard deviation and count. Planning
and IK fields for PPO are intentionally missing. Wall-clock timings depend on
hardware and concurrent load; path lengths are summed at physics substeps.

`python scripts/verify_detour.py` demonstrates a real robot case in which the
straight joint path is blocked and RRT executes an alternative. This diagnostic
is distinct from the sparse random benchmark, where many direct paths are valid.

## Graphical demo

```bash
python scripts/demo.py --method rrt
python scripts/demo.py --method ppo --model models/ppo_clean/best/best_model.zip
python scripts/demo.py --method ppo --model models/ppo_noise/best/best_model.zip \
  --config configs/eval_noise.yaml
```

The GUI shows the robot, orange panel, green target, red obstacles, grey wall and
blue trajectory; the console prints errors, collision and success each step.
Seed 10002 is the default, selected as a successful demonstration scene; the
benchmark retains all 100 seeds and its failures. The window closes after the episode plus five seconds;
change `--hold-seconds` as needed. Each run also saves a rendered snapshot.

On the development Linux host, Conda's C++ runtime conflicted with Mesa. This
verified workaround selects the system runtime:

```bash
LIBGL_ALWAYS_SOFTWARE=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
  python scripts/demo.py --method rrt
```

This path is Linux-specific; do not use it on macOS/Windows. A display-free image
can be generated with `python scripts/demo.py --method rrt --headless`.

## Repository guide

```text
assets/                   Local KUKA model, meshes and attribution
configs/                  Inherited YAML configuration and curricula
src/wall_panel_rl/envs/    Gymnasium environment, noise, geometry, collision
src/wall_panel_rl/control/ Shared physical joint actuator
src/wall_panel_rl/planning/IK and transparent RRT-Connect implementation
src/wall_panel_rl/baselines/Classical trajectory controller
src/wall_panel_rl/rl/      Reward-component callback
src/wall_panel_rl/utils/   Configuration and comparable rollout metrics
scripts/                  Smoke, training, evaluation, benchmark, plots, demo
tests/                    API, collision, reward, noise, limits and detour tests
models/                   Saved policies, resolved configurations, learning logs
results/raw/              Episode CSVs, scenarios, representative trajectories
results/summary/          Aggregated measured metrics
results/figures/          Learning, comparison and trajectory plots, screenshots
```

`REPORT_DRAFT.md` contains the scientific report and measured results;
`DEVELOPMENT_SUMMARY.md` records the completed runs and acceptance checks;
`DECISIONS.md` explains design choices and deviations;
`PROJECT_SPECIFICATION.md` preserves the supplied requirements. Regenerate the
report from the primary CSVs with `python scripts/write_report.py`.
`requirements-lock.txt` additionally pins transitive packages from the isolated
Conda install (use the CPU wheel index from `requirements.txt` when installing it).

## Reproducibility and external work

Python, NumPy, PyTorch, Gymnasium, SB3 and scenario RNGs are seeded. Training is
single-seed CPU evidence, not a multi-seed statistical study. Timeout-bounded RRT
and floating-point physics can vary with platform/load. Saved scenario records
and configs make the experiment inspectable; they do not promise bitwise replay
across machines. Repeat training with multiple seeds before drawing broad claims.

External software: PyBullet supplies physics, FK, IK, low-level motors and the
KUKA model; Gymnasium supplies the environment API; Stable-Baselines3 implements
PPO; PyTorch supplies neural networks; NumPy/SciPy provide numerical/quaternion
operations; Pandas/Matplotlib handle results; PyYAML and TensorBoard support
configuration/logging. The environment, RRT tree growth/connection/smoothing and
experiment scripts are implemented in this repository. See `assets/README.md`
for source and preserved robot license. This is simulation-only, with a rigid
small panel, simplified sensing, sparse boxes and no communication/control delay.
Joint-space actions **do not automatically guarantee collision avoidance**.

## Farther start and exactly three random obstacles

The two additional configurations preserve the original benchmark and models:

- `configs/far_start.yaml`: farther starts, no obstacles. The panel's actual FK
  distance to target must be between 1.80 and 1.90 m; joint offsets alone are not
  used as a distance guarantee.
- `configs/far_three_obstacles.yaml`: the same farther starts and exactly three
  boxes with randomized centers. Box sizes are 0.08 × 0.08 × 0.10 m, 0.12 × 0.10 × 0.16 m, and
  0.16 × 0.14 × 0.20 m. Centers are
  drawn around (0.50, 0.38, 0.48) m with per-axis ranges ±(0.35, 0.48, 0.26) m.
  Samples overlapping start/target clearance are rejected, and boxes must be at
  least 0.15 m apart. This is endpoint feasibility checking, not a proof of a path.

```bash
python scripts/demo.py --method rrt --config configs/far_start.yaml \
  --seed 10002 --hold-seconds 60
python scripts/demo.py --method rrt --config configs/far_three_obstacles.yaml \
  --random-seed --hold-seconds 60
```

`--random-seed` creates a new layout on each launch and prints its seed. Reproduce
one layout with `--seed NUMBER` instead. On this Linux host prefix either command
with `LIBGL_ALWAYS_SOFTWARE=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.
Start positions are randomized too, within the farther-distance constraint.
The existing PPO models have **not** been retrained or certified for these harder
scenes, although the observation/action dimensions remain 41/7.

Reproduce the two-stage RRT audit (100 episodes per stage by default):

```bash
python scripts/verify_far_scenarios.py --run-id my_far_random_three
```

The audit records initial distance, IK/planning/execution outcomes and whether
the **straight joint-space edge to the actual IK goal** is blocked. A successful
physical execution in such a scene is counted as a successful detour. Raw scene
geometry, configs and representative trajectories are saved alongside CSV/JSON.
The earlier equal-size-box run is `results/raw/far_185_random_three/`; see its `summary.json` for
measured outcomes. This separate experiment does not replace the original report.

Earlier ~0.92 m validation on seeds 10000–10099: both farther-start stages succeeded in
100/100 executions without collision. In the three-random-box stage, 84/100 direct
joint-space edges were blocked and all 84 were successfully routed around.
The mean initial distance increased from 0.486 m to 0.922 m (observed farther range
0.804–1.032 m). These results apply to the tested seed set, not every possible
random scene. Details: [extension results](results/raw/far_random_three/README.md).
Generate its distance/trajectory figure with `python scripts/plot_far_scenarios.py`.


The current farther-start setting targets an average of approximately **1.85 m**,
keeping the original target and robot unchanged. The requested 2 m mean exceeds
the geometric upper bound (~1.908 m) for this fixed target; the 1.85 m alternative
was explicitly selected. Measured over seeds 10000–10099, the mean is **1.842 m** (range 1.833–1.849 m).
With the earlier equal-size boxes, both stages completed 100/100 RRT executions without collision; the random-three-box
stage had 100 blocked direct edges and 100 successful detours. Direct-edge checks
include all collision constraints, so this does not attribute every blockage to a box.
See [earlier equal-size-box results](results/raw/far_185_random_three/README.md).

GUI animation now defaults to **0.6x** speed. Override it with `--playback-speed 1`
for normal speed or `--playback-speed 0.6` explicitly. This only paces GUI display;
physics, control frequency and headless training remain unchanged. Change the
default in `configs/base.yaml` under `demo.playback_speed`. The farther scene also
sets `demo.camera_distance: 2.8` to keep both ends of the motion visible.

```bash
python scripts/demo.py --method rrt --config configs/far_three_obstacles.yaml \
  --random-seed --playback-speed 0.6 --hold-seconds 60
python scripts/verify_far_scenarios.py --run-id my_far_185_random_three
python scripts/plot_far_scenarios.py --run-id my_far_185_random_three
```

### Paired noise comparison with sparse, unequal boxes

`configs/far_three_obstacles.yaml` has zero target-observation noise.
`configs/far_three_obstacles_noise.yaml` uses independent Gaussian position noise
(0.005 m standard deviation per axis) and rotation-vector noise (0.5 degrees per axis).
The true target, initial state, and obstacle geometry match exactly for the same seed.
RRT plans once using the initial noisy target observation; later observations do not
replan its trajectory. Success and final errors are measured against the true target.
This noise does not represent motor disturbances. Small observation errors may not
produce an obvious visual difference, so compare recorded final errors as well.

Run these consecutively with the same seed (do not use `--random-seed` for pairing):

```bash
python scripts/demo.py --method rrt --config configs/far_three_obstacles.yaml --seed 10002
python scripts/demo.py --method rrt --config configs/far_three_obstacles_noise.yaml --seed 10002
python scripts/compare_far_noise.py --episodes 20 --run-id my_sparse_comparison
```

Edit `obstacles.dimensions_by_obstacle`, `randomization`, and `min_obstacle_gap`
in `configs/far_three_obstacles.yaml` to adjust sizes, distribution, and spacing.
Edit `noise` in `configs/far_three_obstacles_noise.yaml` to adjust noise strength.
The previous 100-episode results used the earlier geometry and are historical.

## Five moving obstacles (current default demo)

Run `python scripts/demo.py --playback-speed 0.6 --hold-seconds 60` for a fresh
random scene, or add `--seed 10002` for a reproducible replay. The default config
is now `configs/far_five_dynamic.yaml`. For target-observation noise, use
`--config configs/far_five_dynamic_noise.yaml`. Older explicit three-obstacle
commands still select the historical static scene.

Before placing obstacles, the environment plans an obstacle-free RRT-Connect route
from the sampled start joints to the original goal joints. FK of the carried panel
along that route gives a Cartesian reference curve. The five unequal boxes are
anchored at 1/6 through 5/6 of its arc length, with small transverse/vertical jitter.
If endpoint overlap would occur, placement shifts outward in 0.05 m increments
(up to 0.55 m for the current 3D mode). This is the panel-centre trajectory associated with the arm's RRT
joint trajectory, not a straight start-to-target line. The reference uses the true
fixed goal for scene generation, keeping clean/noisy scenes paired; the execution
planner still uses the observed target. The yellow GUI line shows the reference,
and the blue line shows physical execution. Reference joints and panel positions
are saved in the recorded scene.

Each box now moves smoothly in all three world axes (X/Y/Z), with independent
random phases and frequencies on each reset (`motion_mode: random_3d`). This is
reproducible bounded randomised motion, not discontinuous random teleportation.
The current user settings, amplitude 0.40 m and maximum speed 0.40 m/s, are
preserved as bounds on the TOTAL displacement from its anchor and TOTAL speed.
Each axis has amplitude 0.40/sqrt(3) (about 0.231 m); independent angular frequencies
are sampled between 0.5 and 1 times max_speed/amplitude. The swept collision
proxy covers the full resulting 3D box. To keep endpoint poses clear with this
larger sweep, placement can try either side of the reference path and shift
outward up to 0.55 m plus small jitter before reporting placement failure.
Motion advances each physics substep. Pausing for planning freezes simulation time;
this is synchronous simulation, not a real-time moving-world planning guarantee.

The default `rrt.dynamic_mode: timed` allows passage through obstacle motion
regions when they are clear at the predicted arrival time. RRT-Connect generates
geometric candidates at current/future obstacle snapshots; a time-expanded search
then schedules advancing or waiting along each candidate. Every scheduled edge
checks arm and payload at intermediate times against the actual prescribed moving
boxes. The controller tracks one scheduled waypoint per control tick, checks the
next move from the measured joint state, and reschedules if unsafe or unreachable
in one action. Waiting is checked for collisions too during scheduling. A failed
online replan commands a hold and invalidates the old schedule; holds can still be
hit by approaching obstacles, so failures remain recorded. The search has finite
candidate routes and a 30 s budget and may report no plan even if another solution
exists. It is RRT plus time scheduling, not a complete space-time RRT algorithm.

Set `rrt.dynamic_mode: swept` to restore conservative whole-region avoidance.
Full motion envelopes are still used for endpoint placement to protect the fixed
start and goal, and for auditing whether execution entered a previously forbidden
region. They are NOT used as forbidden regions by the timed route planner or IK.
The model assumes known prescribed obstacle motion and pauses simulation time
while computing. Discrete collision samples and kinematic prediction do not
constitute a continuous or real-time physical safety guarantee. GUI queries are hidden.

The original dynamic configuration has 55 values (five seven-value obstacle slots);
obstacle centres update each step. Velocities/phases are not inputs in that config.
Existing 41-input PPO models cannot be used directly; the demo reports this mismatch.
The retraining configuration below adds velocities and uses 70 inputs.

`python scripts/verify_dynamic.py --run-id my_dynamic_run` records ten executions,
resolved config, planning/hold counts, and representative obstacle trajectories.
The dynamic regression also injects an obsolete unsafe route to verify that
stopping/replanning leads to a new physically executable path. Original static
configs and recorded experiments remain available for historical reproduction.

## Dynamic PPO retraining with 70 observations

Current run status: the pilot and both 1,000,000-step training runs completed.
Both models recorded 0/10 successes at every dynamic validation checkpoint.
Formal evaluation was paused at the user's request after 60 RRT clean episodes;
the six-cell comparison is incomplete. See `results/raw/dynamic70_main/README.md`.

Follow-up failure diagnosis is in `results/diagnostics/dynamic70_failure/README.md`.
The selected policies had mostly timeouts with residual pose errors. Re-evaluating
the saved final Noise weights yielded 1/10 validation successes: callback validation
occurs before the last PPO update, whereas final weights are saved after it.
The selected policies remain unchanged and formal evaluation remains paused.

### Mixed-curriculum short-run code (not yet trained)

The new entry point `scripts/train_mixed_pilot.py` defaults to **configuration
checks only**. It loads the source model to check compatibility but performs no
rollouts, optimizer updates, output-directory creation or formal test evaluation.
`configs/ppo_mixed_pilot.yaml` controls the source checkpoints, additional budget,
episode mixture, validation interval and output location.

```bash
# Checks only: safe to run without starting training.
python scripts/train_mixed_pilot.py --condition clean
python scripts/train_mixed_pilot.py --condition noise
```

The prepared defaults resume each condition's 200,000-step
`alignment_final_model.zip`, including its optimizer state, with 50,000 **additional**
steps. Four workers collect 250 steps each, so the budget is exact. Learning rate,
reward, action scale, strict success tolerances and dynamic obstacle speed/range
are unchanged. Each episode independently selects 30% near-goal alignment without
obstacles, 30% far-start static obstacles or 40% far-start dynamic obstacles.
These are episode probabilities; unequal episode lengths mean the proportions of
transitions can differ. Near starts use the last 2–15% of collision-checked reference
joint paths; all modes preserve the 70-value observation and seven-value action.

Validation happens before training (source baseline), every 10,000 additional steps
**after parameter updates**, and on the final model reloaded from disk. Four fixed
validation suites measure full dynamic difficulty, static obstacles, far-start
obstacle-free alignment and near-goal obstacle-free alignment. No test scenes are
used. Each suite records success, collision, joint-limit and timeout rates, plus
mean position/orientation errors **among timeouts only** (`null` when none occur).
Far-start alignment success detects loss of the source policy's alignment ability.

Only when a new training run is intended, explicitly add `--train`:

```bash
# These commands WILL train. They have not been run as part of code preparation.
python scripts/train_mixed_pilot.py --condition clean --train
python scripts/train_mixed_pilot.py --condition noise --train
```

Each command runs only one short experiment, with no automatic long training or
formal evaluation afterward. Future outputs go to
`models/dynamic70_mixed_pilot/ppo_clean/` or `ppo_noise/`; existing directories are
never overwritten. `validation.jsonl` contains suite summaries,
`validation_episodes.jsonl` contains individual outcomes, and worker monitor CSVs
include `curriculum_task`. `plan.json` records source/data hashes and inherited PPO
parameters; `training.json` distinguishes source, additional and total steps.
Checkpoint selection retains dynamic success rate then mean reward; alignment
retention is reported separately, not enforced by a hidden stopping rule.

The legacy `train_dynamic_study.py` callback also now validates after updates and
explicitly handles the final update. Previously recorded models and metrics are
unchanged; use the historical run manifest when interpreting those results.

`configs/ppo_dynamic_clean.yaml` and `configs/ppo_dynamic_noise.yaml` add exact
world-frame obstacle velocities. Each of five slots has relative position (3),
size (3), velocity (3), presence (1); total observation size is 70. Other configs
retain their original 41/55 dimensions. This remains partially observable because
PPO does not receive complete motion phases/frequencies or a noise-free target.
The RRT baseline retains access to prescribed future obstacle motion.

The approved study is implemented in `scripts/train_dynamic_study.py` and
`scripts/run_dynamic_study.py`. It uses 128 frozen training scenes, 10 validation
scenes and 100 test scenes in disjoint seed ranges starting at 30000/40000/50000.
Scene generation failures are recorded; accepted scenes are replayed identically
across methods/conditions. Test scenarios are not used for checkpoint selection.

The pilot is exactly 50,000 steps (40k alignment, 5k static, 5k dynamic), followed
by fresh, independent clean/noisy PPO runs of exactly 1,000,000 steps each (200k
alignment, 200k static, 600k dynamic). Alignment starts move progressively along
collision-checked reference paths; static scenes freeze the same obstacle layouts;
final evaluation uses full dynamic difficulty. Reward and terminal tolerances are
unchanged. Four CPU workers each collect 250 steps per rollout; minibatches have
100 samples. This avoids exceeding authorized budgets through rollout rounding.
One training seed (42) is used per condition. Final-phase checkpoints are selected
by validation success rate, breaking ties by mean reward. Old models are retained.

Run state is saved to `models/dynamic70_study/pipeline_status.json`; each model has
`progress.json`, monitor CSVs, TensorBoard logs, resolved config and training counts.
The six-cell test results are written to `results/raw/dynamic70_main/` after training.
Do not infer completion or convergence from the existence of an output directory.

To reproduce the study in a fresh checkout without an existing
`models/dynamic70_study` output folder:

```bash
conda activate wall-panel-rl
python scripts/train_dynamic_study.py prepare
python scripts/run_dynamic_study.py
python scripts/report_dynamic_study.py
```

The runner completes the pilot, trains the two models concurrently, then evaluates
RRT, PPO-Clean and PPO-Noise on both observation conditions (600 episodes). It will
not overwrite an existing study. Training uses no GUI. Inspect progress with:

```bash
cat models/dynamic70_study/pipeline_status.json
cat models/dynamic70_study/ppo_clean/progress.json
cat models/dynamic70_study/ppo_noise/progress.json
```

After training, display a policy with the matching 70-input configuration:

```bash
python scripts/demo.py --method ppo \
  --config configs/ppo_dynamic_clean.yaml \
  --model models/dynamic70_study/ppo_clean/best_model.zip \
  --seed 50000 --playback-speed 0.6 \
  --snapshot results/figures/dynamic70_ppo_clean.png
python scripts/demo.py --method ppo \
  --config configs/ppo_dynamic_noise.yaml \
  --model models/dynamic70_study/ppo_noise/best_model.zip \
  --seed 50000 --playback-speed 0.6 \
  --snapshot results/figures/dynamic70_ppo_noise.png
```

Omit `--seed` for a new randomly generated scene. A demo regenerates the reference
path with a finite wall-time planning budget; the benchmark instead replays frozen
scene files to guarantee identical geometry across methods. A completed model may
still fail the task: inspect the held-out results before making performance claims.

## BC-to-PPO waypoint fine-tuning

The first static pilot completed 10,000 additional transitions in 15.07 minutes, with 162 completed training episodes. Validation remained 89/90 static successes, zero collisions and 20/20 alignment successes. The selected 5,000-step checkpoint reduced mean successful control steps only from 60.0225 to 60.0112; the final model returned to 60.0225. This is no material improvement over BC. Full results are in `results/diagnostics/waypoint_ppo_pilot_seed61/README.md`; weights are in `models/waypoint_ppo_pilot_seed61/`. Independent testing and dynamic-obstacle training have not been performed.

In the `codex/path-reference-ppo` worktree, `scripts/train_waypoint_ppo.py` provides
an explicit static 77-input training wrapper. Its default mode loads and checks
configuration/model compatibility, prints the effective parameters, and exits
without learning or creating a training output directory:

```bash
python scripts/train_waypoint_ppo.py
```

Settings are in `configs/waypoint_ppo_pilot.yaml`: 10,000 additional transitions,
four workers, 250 steps/worker/rollout, batch size 100, three epochs, learning rate
1e-5, clip 0.05, target KL 0.005 and fixed action std 0.02. Only the 14 gain/bias
coefficients and critic are trainable; velocity terms and std remain frozen even
after save/reload. The unused generic actor remains frozen. Planning failures end
an episode on the next agent step with zero reward and no physics advance. Failed
planning transitions count towards the PPO budget and are reported, not filtered.
Training scene seeds are drawn from [300000,400000); validation and independent
reserved seeds are excluded. Generation errors surface as errors rather than
silently omitting difficult scenes.

Both training and inference produce raw policy actions, which use the same
projection in the environment (Box clipping followed by 0.01-rad joint-limit
margin). SB3 keeps the original Gaussian action and log probability for PPO;
the projected action is part of the environment transition and is recorded in
info. Shaping adds `2 * (previous-current waypoint distance) / max_joint_increment`
to the original environment reward. Both distances refer to the same waypoint;
the cursor advances afterward. The original success/collision criteria remain.

Validation runs the three methods (ordinary tracking, frozen released BC, current
PPO) on identical physical starts, noise streams and a shared freshly planned path
per scene. All 110 existing release validation cases are used, including 20
alignment cases. Summaries record success/collision/limit/timeout/planning-failure
rates, conditional timeout pose errors and mean successful control steps. Near/far
alignment success is recorded separately. These reused scenes are not independent
generalization tests. Periodic validation occurs after updates; final weights are
saved, reloaded and validated. Selection prioritizes static success, lower collision
rate, retained alignment success, then fewer successful control steps. Baseline BC
is included as the initial candidate, so later regressions need not replace it.

The command used for the completed pilot is below. The existing output directory
is protected against overwrite; use a new output path for any future authorized run:

```bash
python scripts/train_waypoint_ppo.py --train
```

It writes to `models/waypoint_ppo_pilot_seed61` and refuses overwrites. Files include
resolved settings, source model hash, parameter mask, effective hyperparameters,
monitor CSVs, checkpoint selection, validation summaries and individual episodes.
There is no automatic long run, independent test execution or dynamic-obstacle
training. This remains RRT-guided servo learning; the actor does not directly use
obstacle features to plan a route.

## Autonomous dynamic-avoidance pilot

The first authorized 10k pilot is complete. All 10 easy-motion, 10 full-motion and
10 frozen-obstacle avoidance validation cases ended in collision at each checkpoint;
far/near no-obstacle alignment remained 20/20. Training completed 767 episodes
(142 alignment successes, 4/376 moving-scene successes, 7/249 static-scene successes).
The residual weights changed, but reliable avoidance was not learned; selection
retained the zero-additional-step initialization. Actual 10k weights remain in
final_model.zip. Detailed results are also available in the visible parent repo
at results/diagnostics/dynamic_autonomous_pilot_seed61/README.md.

`configs/dynamic_autonomous_pilot.yaml` and `scripts/train_dynamic_autonomous.py`
implement the user-selected dynamic task. Starting from the selected +5k static
PPO checkpoint, the new 77-input actor keeps its tracking coefficients frozen and
adds a 64/64 residual network consuming ALL observations, including five obstacle
positions and velocities. Its last layer is initialized to zero, so the initial
mean action exactly matches the source. Scale 2 allows reversal or cancellation
of a saturated tracking action. Only the residual network and critic learn;
velocity coefficients, tracking gains/biases and action std remain frozen. Critic
weights transfer; optimizer state starts fresh because the actor architecture changed.

RRT generates an obstacle-free reference during SCENE PREPARATION only. During
execution there is no dynamic planner, collision rejection, safety override or
expert action fallback. PPO decides deviations/waiting/avoidance through seven
joint increments; joint-limit actuation projection is shared with the static
wrapper. This is autonomous avoidance with a reference/tracking prior, not a
policy with no planning-derived input. Full sweeps protect endpoint placement;
they are not forbidden regions for execution. Feasibility throughout the episode
is not guaranteed. Obstacles may later move into each other's regions.

The 10k-additional-step pilot uses four workers and 250 transitions per worker per
rollout. Episode probabilities are 20% no-obstacle alignment (half near, half far),
30% frozen obstacles, 50% moving obstacles. Scene bank: 128 training layouts,
10 separate dynamic validation layouts, plus the original 20 alignment cases
with prepared obstacle-free references. Training phases are randomized each reset.
The maximum total displacement/speed are initially .10 m/.05 m/s, held for the
first 20% of training, ramped to .40 m/.40 m/s at 80%, then held at full difficulty.
A worker's completed steps determine its reset difficulty. Strength stays fixed
within an episode. Anchor layouts use full-strength endpoint clearance throughout,
so small-motion scenes can be easier than newly generated tight low-strength layouts.

Reward remains original physical task reward plus the static interface's same-
reference progress shaping; no clearance penalty or relaxed success condition is
added. The reference cursor may advance when the arm passes a waypoint within a
0.20-rad lateral corridor; this is progress bookkeeping, not an action constraint.
Observation noise and delay are zero. Learning rate is 1e-4 for the new network,
clip .05, target KL .005, three epochs; source action std .02 is retained. This is
a different actor architecture from the 14-coefficient static experiment.

Validation compares reference-only ordinary servo, the frozen source PPO tracker
and the new actor on identical stored scenes/reference paths/phases. It is NOT a
comparison against the time-aware RRT avoidance baseline. Low/full motion, frozen
obstacles and far/near alignment are reported separately, including collision,
timeout, joint-limit, error and control-step measures. Selection prioritizes full-
motion success, easy-motion success, full-motion collisions, alignment retention,
then full-motion return. Validation is post-update; final weights are reloaded.
The 211000–211099 independent static test seeds are not used.

```bash
# No files or training in default check mode.
python scripts/train_dynamic_autonomous.py
# Create frozen scenario banks only.
python scripts/train_dynamic_autonomous.py --prepare
# Explicitly authorized 10k pilot; never launches a longer run automatically.
python scripts/train_dynamic_autonomous.py --train
```

Outputs: `models/dynamic_autonomous_pilot_seed61/`; scene bank:
`data/dynamic_autonomous_pilot_seed61/`. Existing model outputs are protected from
overwrite. `sampling_progress.json` tracks collected transitions;
`validation.jsonl` and `validation_episodes.jsonl` record scores and episodes.
Initial policy compatibility does not establish dynamic avoidance competence.

## Staged autonomous avoidance v3 (BC completed; PPO withheld)

The current iteration is prepared in `configs/staged_avoidance.yaml` and
`scripts/staged_avoidance.py`. It adds a 1-static -> 1-dynamic -> 3-dynamic ->
5-low-motion -> 5-full-motion curriculum, offline collision-avoidance demonstrations,
actor-only imitation before PPO, a learned prior gate, action std 0.10, reduced
waypoint shaping (0.25), whole-arm/panel obstacle-clearance feedback, and explicit
stage acceptance. Default invocation only checks configuration/model spaces.
The completed v2 single-static trial failed: even alignment was lost after BC.
Its models/data and archived `configs/staged_avoidance_v2.yaml` are retained.
V3 fixes alignment labels, fixes the expert endpoint to the reference IK branch,
recovers reference progress after detours, masks correction without obstacles,
and makes correction vanish near a bias-free terminal servo. BC adds terminal
supervision and balanced task sampling. An inference-only BC gate checks stage
success/collisions, alignment retention and terminal actions before PPO can start.
New output roots use `staged_avoidance_v3_seed61`; v2 artifacts cannot initialize
v3 training directly. The first authorized v3 one-static run prepared 64 training
layouts and 20 target validation layouts, collected 100 successful expert episodes
(4,794 transitions, 50 scene seeds), and completed 2,000 BC updates. BC retained
20/20 alignment with no collisions but scored 0/20 obstacle success and 20/20
collisions. Terminal-action probes passed. BC release failed, so PPO was never
launched (zero additional PPO steps), and no harder stage started. This preserves
alignment; it does not establish learned obstacle avoidance. Inspect
`models/staged_avoidance_v3_seed61/one_static/bc/closed_loop_gate.json` and the visible
main-checkout report at `results/diagnostics/staged_avoidance_v3_seed61/one_static/`.

See [完整配置、奖励与操作说明](docs/STAGED_AVOIDANCE.md). Expensive operations require
separate `--prepare`, `--collect-teacher`, `--pretrain` or `--train` flags. Pretrain
ends with closed-loop validation; `--validate-bc` reruns validation only. A stage
never automatically advances or expands its 10k PPO budget. Later-stage training
requires the previous stage to pass and its selected checkpoint hash to match.

## BC budget and correction-data study

`configs/bc_budget_study.yaml` and `scripts/run_bc_budget.py` separate fixed data
contracts from optimization/output budgets, and select BC checkpoints by reloaded
closed-loop scores at 2k/5k/10k updates. Source/data/physical/reference hashes remain
checked; old results are never overwritten. New checkpoints also save BC Adam/RNG
state, and evaluation preserves the subsequent optimizer RNG stream.

The fixed-100-episode control exactly reproduced the old 2k model, then reached
10/20 one-static success at 10k (9 collisions, 1 timeout; alignment 20/20). Collision
release failed. The authorized next phase expands to 256 layouts and 1,000 success
episodes (200 alignment, 600 regular avoidance, 200 corrective avoidance), with
at least 200 distinct avoidance layout seeds. Correction prefixes are logged for
replay but are not imitation action labels. A second 10k BC budget and at most one
gated 10k PPO pilot are allowed; no harder stage starts automatically.

See [budget workflow](docs/BC_BUDGET_STUDY.md). Live main-checkout report:
`results/diagnostics/bc_budget_v3_seed61/`; log: `logs/bc_budget_v3_seed61.log`.
This is a bounded engineering validation study, not an independent test result.

The study has now finished. Expansion collected exactly 1,000 successes in 1,003
attempts, with 46,595 labeled transitions, 254 total layout seeds and 248 avoidance
layout seeds. Ten sampled corrective episodes replayed with 10/10 success and
zero observation difference. Expanded-data BC scored 0/20, 2/20, 0/20 target success
at 2k/5k/10k; its 5k checkpoint was retained despite lower final MSE. Both variants
kept 20/20 alignment, but neither passed collision release. PPO was not launched.
At the end of that study, the best BC was fixed-data 10k at 50% success / 45% collisions.
Visible report exports `best_bc_model.zip` with metrics/provenance; it is not
released for PPO and does not establish generalization or dynamic avoidance.

### Targeted BC continuation completed (2026-10-30)

The new [targeted continuation study](docs/TARGETED_BC_STUDY.md) resumes that 50%
actor and its BC Adam/RNG state. Both arms complete 10k additional BC updates.
Original-data continuation regresses and retains the initial checkpoint. Adding
100 targeted expert recoveries (3,523 labels, 34 training layouts, 107 attempts)
reaches 13/20 success, 2/20 collisions and 5 timeouts, while retaining 20/20
alignment. Its selected checkpoint is at 20k cumulative BC updates and passes the
existing BC release gate. Ten corrective replays exactly match their observations.

On 100 frozen independent one-static scenes, scored only after selections, the
baseline gets 36 successes / 61 collisions / 1 joint-limit failure / 2 timeouts;
the selected targeted model gets 60 successes / 13 collisions / 0 joint-limit
failures / 27 timeouts. Independent alignment is 20/20 for both. These are still
one-static results; the five-full diagnostic remains 0/10 success. No PPO or stage
promotion was run. The independent bank must not be reused as an unseen final
test after further tuning.

Visible report and selected export:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/targeted_bc_v3_seed61/README.md`
and `selected_bc_model.zip` in that directory. Old models/reports are preserved.

### Stall recovery BC/PPO comparison completed (2026-10-30)

The [stall recovery study](docs/STALL_RECOVERY_STUDY.md) diagnosed near-cancellation
between the prior and learned correction in training timeouts. Forty recoveries
were collected in 40 attempts (339 expert labels), with 10/10 exact physical
replays. Low-LR BC completed 10k updates but retained the original 65% model.
An independent 10k-step PPO pilot from that same starting actor, with bounded
training-only stagnation feedback, reached 14/20 validation success, 3 collisions
and 3 timeouts versus baseline 13/20, 2 collisions and 5 timeouts.

On the NEW 100-scene test bank (990000 onward), baseline scores 67 successes /
9 collisions / 24 timeouts; selected PPO scores 78 / 13 / 9. Both retain 20/20
alignment, with no joint-limit failures. This reduces timeouts but increases
collisions; it is a one-seed engineering result, not proof of significance or a
five-dynamic-obstacle result. Validation selected the model before test scoring.
PPO completed exactly 10000 additional transitions (101 complete training
episodes, including 18 alignment / 83 static; stochastic training outcomes are
separate from deterministic evaluation). No stage promotion or budget extension.
Full tests: 83 passed. Visible report/export:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/stall_recovery_v3_seed61/README.md`
and `selected_model.zip` with `selected_model.json` in the same directory.

### Matched wall-feedback PPO pilots completed (2026-10-30)

The [wall-feedback study](docs/CLEARANCE_PPO_STUDY.md) diagnosed the previous
collisions: all five training collisions and 12/13 known test collisions were
panel-wall contacts; the remaining known test collision was self contact. Keep
obstacle clearance weight .5 and add training-only wall proximity feedback below
.02m, outside the nominal .025m goal gap. Two independent continuations resume
the SAME latest PPO actor/critic/optimizer, each for exactly 10k transitions.
Only wall weight differs (control 0, treatment 1); original evaluation reward,
240-step limit, observations, actions and physical collision definition stay fixed.

Control validation regresses and retains its initial actor. Wall-feedback scores
14/20 success / 2 collisions / 4 timeouts at 5k, then 16/20 / 2 / 2 at 10k. The
predeclared non-regression selector retains only the 10k improvement. On a NEW
100-scene bank (1000000 onward), starting PPO scores 83 successes / 6 collisions /
11 timeouts; selected wall-feedback PPO scores 92 / 4 / 4. Alignment stays 20/20,
limits zero. Test scores did not select the model. Both variants have identical
initial PPO optimizer state and matching physical episodes in the first rollout.

The selected final checkpoint meets the static 80% success / <=10% collision
threshold once; 5k did not, so the two-consecutive-check promotion requirement is
NOT met and no dynamic stage started. Full tests: 88 passed. Model lineage is
20k BC updates plus 20k new PPO transitions (stored PPO total 25k includes the
old 5k source counter). Visible report/export:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/clearance_ppo_v3_seed61/README.md`
and `selected_model.zip` / `selected_model.json` in the same directory.

### Unchanged-parameter static stability check completed (2026-10-30)

`scripts/run_static_stability.py` adds exactly 10k PPO transitions from that latest
checkpoint and optimizer, preserving every training/reward/physical setting.
Validation at 0/5k/10k: success 16/20, 15/20, 16/20; collisions 2, 1, 0; timeouts
2, 4, 4. Alignment stays 20/20, limits zero. The final candidate reduces collisions
but increases timeouts, so the existing non-regression selector retains step 0.
The 5k success rate .75 fails the stage gate; two consecutive passes are absent.
No dynamic training or budget extension. On the already-seen 100-case bank, the
retained model reproduces 92 successes / 4 collisions / 4 timeouts; this is a
regression check, not a new independent result. Seventeen relevant tests pass.

Visible report: `results/diagnostics/static_stability_v3_seed61` in the main
checkout. `selected_model.zip` retains the previous selected actor; the separate
`final_candidate_unselected.zip` preserves the 10k candidate for review. Its stored
PPO counter is 35k, while the retained model stays at 25k. No model was overwritten.

### Autonomous demo and expanded model comparison (2026-10-30)

See [terminal demo commands](docs/AUTONOMOUS_PPO_DEMO.md) for the current autonomous
77-input PPO. The new demo reproduces saved reference observations with no runtime
expert avoidance route; seed 1000000 reproduces success in 65 steps.

Inference-only comparison of retained and final unselected models covers 100 known
and 100 fresh scenes. Known bank: retained 92/4/4 success/collision/timeout versus
candidate 86/6/8. Fresh bank from 1002000: retained 90/6/4 versus candidate 88/3/9.
Combined descriptive results: retained 182/200 success, 10 collisions, 8 timeouts;
candidate 174/200, 9 collisions, 17 timeouts. The combined bank is not a wholly new
independent test. All far/near alignment checks pass, limits zero. Models stay fixed.

Original validation transitions are 16 success-to-success, two timeout-to-timeout
and two collision-to-timeout. All four candidate timeouts persist to 480 steps.
Three show strong prior/residual cancellation, stale non-final reference indices,
route deviation beyond the .30rad recovery radius and joint goal error outside
the .20rad terminal blend. This suggests a recovery/terminal-handoff issue; index
stagnation alone is not proof of a coding bug. No control change or training is
performed in this study. Five targeted comparison/demo tests pass. Visible report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/static_candidate_comparison_v3_seed61/README.md`.

### Fixed-weight reference/terminal ablation completed (2026-10-30)

The [handoff ablation](docs/REFERENCE_HANDOFF_ABLATION.md) keeps the retained PPO
tensors fixed and compares original .30rad recovery / .20rad terminal settings,
recovery radius .45 only, and terminal radius .40 only. These change controller
behavior/local reference inputs; they are not new PPO training results. No default
configuration or checkpoint is changed. Nineteen relevant tests pass.

Original validation success/collision/timeout: baseline 16/2/2, wider recovery
17/3/0, earlier terminal 2/18/0. Wider recovery resolves the two old timeouts but
also turns an original success into a collision. On the same 200 known regression
scenes: baseline 182/10/8, wider recovery 161/27/12, earlier terminal 25/175/0.
Alignment totals are 60/60, 60/60, 55/60, respectively, and limits zero. Both
treatments fail non-regression checks. Keep the original controller. Consider a
future stagnation-triggered recovery test rather than treating the two failed
global changes as ready for adoption. No training, new independent claim, combined
factor, dynamic stage or automatic adoption. Source/code/case hashes verified.
Visible report: `results/diagnostics/reference_handoff_ablation_v3_seed61` in the
main checkout, with `README.md`, `overview.png`, per-case traces and verification.

### Conditional recovery inference study completed (2026-10-30)

The [conditional recovery study](docs/CONDITIONAL_RECOVERY_STUDY.md) keeps the same
retained PPO tensors, normal .30rad recovery and .20rad terminal servo. Only after
>=60 steps and a 30-step window of negligible task progress/net joint movement
with a stale non-final reference does the separate ConditionalRecoveryEnv try
.45rad forward recovery. At most three attempts, 20 steps apart, one success per
episode; no direct action overwrite, new route or expert fallback. Alignment and
dynamic scenes are excluded. Twenty-five relevant tests pass.

Original validation improves from 16/2/2 to 18/2/0 success/collision/timeout; known
200-case regression from 182/10/8 to 188/10/2. This passes the predeclared
non-regression gate, so a frozen 100-scene prospective bank from 1004000 is then
scored once: baseline 93/4/3 versus conditional 94/4/2. Do not exaggerate the
one-case fresh-test gain. Nine distinct timeouts across validation/regression/test
become successes, with no new collisions or original successes lost. All 391
untriggered episodes match original outcome, length and position/orientation
errors. All nine attempts recover, bounded limits verified, all alignment retained.

This is a reference-interface/control-rule improvement, not new PPO learning.
No weights/default config are replaced; no training, dynamic stage or automatic
adoption. Source, tensor, code and case hashes verified. Visible report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_recovery_v3_seed61/README.md`.

### Conditional training integration and 10k PPO adaptation completed (2026-10-30)

The [conditional PPO adaptation](docs/CONDITIONAL_PPO_ADAPTATION.md) integrates
the validated guard into training with the existing wall/stall rewards, and uses
the matching execution wrapper for evaluation and the default autonomous demo.
It completes exactly 10,000 additional PPO transitions, with no BC updates.
Full pre-training suite: 115 tests pass, including train/eval agreement through
an actual successful recovery. Static one-obstacle course only.

Conditional baseline: 18 success / 2 collision / 0 timeout out of 20. At 5k:
16/1/3; at 10k: 17/0/3. All alignment retained and no joint-limit failures.
Both updated checks meet the minimum stage gate, but neither passes the stricter
non-regression selection. Keep step 0 and its original network tensors; this
round does not improve PPO validation success. No dynamic stage is started.

The retained model reproduces 282/300 success, 14 collisions, 4 timeouts on three
already-seen regression banks; this is not a new independent test or evidence
that updated weights generalize better. Training includes 135 completed episodes
and 264 partial-episode transitions; just one recovery attempt, none successful.
Recovery-state experience remains sparse. Source, frozen parameters, inputs/code
and exported per-case results are verified. The model requires the companion
conditional controller settings, not just a zip. Visible report with curves:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_ppo_adaptation_v3_seed61/README.md`.
