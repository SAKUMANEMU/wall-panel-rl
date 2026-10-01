# Report draft: collision-aware robotic panel installation

## 1. Introduction

Robotic installation must align a carried panel while avoiding collisions of both
payload and arm. The research question is whether PPO with incremental joint-space
actions can learn this behavior and retain accuracy under target measurement noise,
compared with a sampling-based motion planner and feedback controller.

## 2. Problem formulation

The simulator state includes a fixed-base seven-joint arm and a rigidly attached
0.30 × 0.20 × 0.03 m panel. A fixed reachable vertical target is placed 25 mm in
front of a wall. Starts and sparse box obstacles are randomized. The observation
has 41 entries: q (7), dq (7), relative target position (3), target rotation-vector
error (3), and three padded obstacle slots (center, dimensions, presence).
Normalization maps q to [-1,1], scales dq by 2 rad/s and rotation vectors by pi.

The action is seven normalized incremental joint targets, at most 4 degrees per
joint per decision. This is not Cartesian control. Reward is
`100 Δposition_progress + 12 Δorientation_progress - .015 ||a||² - .05`, plus
`-50` for collision, `-30` for a joint-limit violation and `+100` for success.
Here progress means previous minus current error, in meters and radians.
Success requires true position error <0.01 m and orientation error <2 degrees,
with no collision/limit event. These events terminate; 160-step timeout truncates.

## 3. Code explanation

`envs/wall_panel_env.py` constructs Bullet bodies, the fixed payload constraint,
Gymnasium spaces and reset/step logic. `collision.py` centralizes validity and
state-restoring configuration queries; `geometry.py` handles quaternion errors.
`noise.py` corrupts target measurements only. `control/joint_controller.py` is
the common physical actuator. `planning/ik.py` converts the observed panel target
to end-effector pose and checks multistart IK. `rrt_connect.py` implements tree
growth, connection, edge checking, extraction and shortcutting.
`baselines/classical.py` follows interpolated waypoints. Training uses SB3,
while `utils/metrics.py` runs both methods through identical episode accounting.
Scripts expose all stages without notebooks, and tests check failure cases as
well as successful motion.

## 4. Existing codes, libraries and tools

PyBullet 3.2.7 provides physics, collision geometry queries, FK, IK and position
motors. Its bundled KUKA iiwa URDF/meshes are vendored with attribution and retained
BSD-style notices (`assets/README.md`). Gymnasium provides the environment API;
Stable-Baselines3 provides PPO and callbacks; PyTorch provides neural networks.
NumPy, SciPy, Pandas, Matplotlib, PyYAML and TensorBoard supply numerical,
quaternion, tabulation, plotting, configuration and logging facilities. The
project implements its environment, RRT algorithm and evaluation pipeline. It does
not claim to implement PPO or a rigid-body simulator from scratch. No external
training data or pretrained policy was used.

## 5. Conventional baseline

Collision checking tells us whether a configuration/edge is valid. It does not
produce a replacement route when the direct route is blocked. RRT-Connect grows
two trees with random configuration samples and attempts to join them, explicitly
searching for an alternative. A real seven-joint robot detour regression confirms
that a blocked direct path can be replaced and physically executed.

RRT parameters: extension 0.22 rad, 0.15 goal bias, 3,000 iterations maximum,
5-second search budget, 0.035 rad edge interpolation, 40 shortcut attempts and
5 mm planning clearance. IK uses up to 12 seeds, 200 iterations per candidate,
4 mm/1 degree acceptance against the observed target. Waypoint tracking uses an
0.008 rad Euclidean tolerance. Robot and payload are collision-checked; adjacent
arm links and the intentional terminal-link grasp pair are excluded.

Kinematic path validity is not a guarantee for physical tracking: finite edge
resolution and motor/constraint transients can produce collisions. We detect
contacts at every physics step and retain execution failures in the metrics.
The planner never executes by resetting joints. A reject-only diagnostic is
optional in the specification and is not included in the primary comparison.

## 6. Reinforcement learning method

SB3 PPO uses MlpPolicy with two 128-unit ReLU layers for each actor/critic branch,
learning rate 0.0003, rollout length 1,024, batch size 64, 10 epochs, gamma .99,
GAE lambda .95, clip range .2 and entropy coefficient 0. No VecNormalize is used.
The first diagnostic used default action log standard deviation 0; later runs
initialize at -1. The supplied policies are trained on CPU with seed 42.
Checkpoints, final models, resolved configs and training provenance are retained.

Training first checked the no-obstacle task. Wide-start policies reduced error
but initially settled outside the 1 cm terminal region. Reachability, reward sign,
normalization and scripted target alignment were verified before introducing a
training-only near-target reset curriculum (±.035 rad). A fixed-obstacle fine-tune subsequently lost alignment, so obstacle training
uses a 25% near-target / 75% original-start reset mixture to retain terminal
reward experience. Evaluation explicitly disables that mixture. The original
benchmark start distribution and success threshold were never relaxed. Read DECISIONS.md
and the saved run configurations for the complete curriculum and diagnostic
history. An additional transfer diagnostic isolated the obstacle-input shift:
columns that always received zero in Stage A retained random initialization.
Zeroing these actor/critic input columns preserves the no-obstacle function
exactly and gave 20/20 fixed-obstacle successes before further learning. The
weights remain trainable. The successful phase2_transfer run uses this one-time
initialization adjustment; both final branches share that checkpoint.
A single training seed and limited budgets do not establish convergence.

Evaluated validation-selected checkpoints had 191792 cumulative timesteps (PPO-Clean) and 231792 (PPO-Noise). Each completed training run reached 262144; selected checkpoints precede the final models.

Recorded completed runs:

- `models/phase1_alignment`: 80,896 cumulative model timesteps; requested additional budget 30,000.
- `models/phase1_clean`: 50,176 cumulative model timesteps; requested additional budget 50,000.
- `models/phase1_precision`: 50,176 cumulative model timesteps; requested additional budget 50,000.
- `models/phase1_restored`: 131,072 cumulative model timesteps; requested additional budget 50,000.
- `models/phase2_mixed`: 181,248 cumulative model timesteps; requested additional budget 50,000.
- `models/phase2_obstacle`: 161,792 cumulative model timesteps; requested additional budget 30,000.
- `models/phase2_transfer`: 161,792 cumulative model timesteps; requested additional budget 30,000.
- `models/ppo_clean`: 262,144 cumulative model timesteps; requested additional budget 100,000.
- `models/ppo_noise`: 262,144 cumulative model timesteps; requested additional budget 100,000.

## 7. Simulation setup

The seven-DOF KUKA iiwa is the allowed fallback from the preferred UR5; no verified
standalone UR5 asset was supplied. The panel transform is configured in the URDF
link frame and converted to Bullet's inertial frame for its physical constraint.
Gravity is -9.81 m/s², simulation step 1/240 s, and each decision executes up to
12 physics steps. Both methods use gains 0.3/1.0 and 250 N m motor force limits.
The three stages use no obstacle, one fixed sparse box, and 1–3 randomized boxes.
Initial and target collision checks reject invalid placements. This does not
certify global feasibility. There is no observation or communication delay.

## 8. Experimental design

The primary matrix crosses RRT, PPO-Clean and PPO-Noise with clean/noisy target
observations. Every cell uses seeds 10000–10099 and the same scene construction.
Stored scenario JSON includes actual initial q, target pose and box geometry.
A separate noise RNG uses seed + 1,000,000. Clean noise standard deviations are
zero; noisy observations add independent Gaussian 5 mm/axis translation and
0.5 degree/axis rotation-vector perturbations. These are per-axis standard
deviations, not bounds on the norm of the pose error.

PPO observes a fresh target measurement every step. The classical method solves
IK and plans once from the first matching noisy measurement, then follows the
joint path using true joint feedback. It does not repeatedly filter/replan target
measurements. This gives PPO an opportunity for reactive correction but also
exposes it to repeated noise. Collision geometry and q/dq remain exact for both.
Ground-truth reward and termination are simulator supervision, not extra policy
inputs. A noisy planner goal may be outside the true installation tolerance.

## 9. Metrics

For every episode: installation success, collision, joint-limit event, true final
position/orientation errors, control steps, joint path length (rad), panel Cartesian
path length (m), execution wall time, nominal simulated time, IK+planning wall
time, planner/IK success and cumulative reward. Path lengths are accumulated at
physics substeps for both methods. Planning/IK metrics are not assigned to PPO.
Continuous summaries report mean, sample SD and count; binary metrics use percent.
Failures remain in all error summaries. Successful-only accuracy would answer a
different question. Time measurements depend on CPU and concurrent workloads.

## 10. Results from actual executions

The table below is generated from `results/raw/main_clean/episodes.csv` and
`results/raw/main_noise/episodes.csv`, not hand-entered experimental outcomes.

| Method | Observation | N | Success | Collision | Position (m), mean ± SD | Orientation (deg), mean ± SD |
|---|---|---:|---:|---:|---:|---:|
| ppo_clean | clean | 100 | 58.0% | 20.0% | 0.0906 ± 0.1518 | 8.49 ± 11.11 |
| ppo_clean | noise | 100 | 58.0% | 20.0% | 0.0884 ± 0.1508 | 8.33 ± 11.06 |
| ppo_noise | clean | 100 | 20.0% | 10.0% | 0.0667 ± 0.0906 | 6.22 ± 6.52 |
| ppo_noise | noise | 100 | 21.0% | 9.0% | 0.0646 ± 0.0877 | 6.06 ± 6.37 |
| rrt | clean | 100 | 100.0% | 0.0% | 0.0016 ± 0.0004 | 0.11 ± 0.03 |
| rrt | noise | 100 | 73.0% | 0.0% | 0.0079 ± 0.0034 | 0.82 ± 0.34 |

Full numerical means/SD/counts are in `results/summary/metrics.csv` and binary
rates are in `rates_percent.csv`. Plots cover success, collision, final pose error,
steps, execution/planning time, path length, learning curves and representative
3D panel trajectories. The underlying per-episode rows and configs are retained.
The two primary files contain 600 evaluated episodes total.

## 11. Discussion

RRT achieved 100% clean and 73% noisy success, with collision rates 0% and 0%. PPO-Clean achieved 58% and 58% success; PPO-Noise achieved 20% and 21%. Noise training did not improve noisy-condition installation success over PPO-Clean in this run. Read this alongside collision rates and pose errors: fewer collisions alone does not establish better installation control. These limited-budget, fixed-target results do not establish a general PPO performance ceiling.

Interpret installation success jointly with collision and residual error: a
policy that stops safely a few centimeters away has not installed the panel.
Training return can improve without crossing the terminal pose threshold.
RRT explicitly tests configuration-space alternatives and can solve a blocked
straight-path example; this directly addresses the difference between collision
rejection and collision avoidance. Its noisy-target accuracy is limited by the
single unfiltered measurement used for IK, and physical tracking remains a source
of failures even for a checked path. PPO has no IK failure mode, but can collide,
violate limits or time out; comparing an invented PPO IK rate would be misleading.

**Joint-space actions do not automatically guarantee collision avoidance.**
For PPO it must be learned from observations, rewards and experience. For RRT,
collision avoidance results from explicit state and edge checks. Sparse random
boxes do not imply every episode demands a detour. Learning and noise effects
must be interpreted within the measured six-cell table, not assumed in advance.
No general claim that noise training improves robustness is warranted without
consistent effects across multiple training seeds and longer runs.

## 12. Limitations

Simulation-only results; rigid small panel; fixed target; simplified, exact
obstacle/joint sensing; sparse boxes; independent Gaussian target noise; no delay;
one training seed; finite training and evaluation budgets. Because the target is
fixed, a policy may partly memorize a useful goal configuration. Robustness to
noisy observations here does not establish generalization to new true targets. The near-target
curriculum differs from the final start distribution and its callback successes
are not benchmark successes. RRT uses discrete collision sampling and a planning
clearance rather than continuous collision certification. There is no real robot,
gripper compliance, force control, physical wall-contact installation, visual
perception, observation filtering, or larger-panel experiment. Nonconverged runs
are retained as evidence and should not be treated as a tuned PPO performance
ceiling. The fairness comparison shares physical simulation and initial scenes,
but target-update frequency differs as explicitly described above.

## 13. Conclusion

The repository implements and executes both joint-space PPO and an explicit
collision-aware RRT-Connect pipeline using shared physical actuation and measured
simulation outcomes. The detour regression establishes that the classical method
actively finds alternative paths. The primary experiment measures strict panel
installation and noise sensitivity; the table states what the trained policies
actually achieved. This single-seed course-scale experiment supports reproduction
and further study, not a general ranking of learned control versus planning.
