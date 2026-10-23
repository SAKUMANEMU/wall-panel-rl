# Implementation decisions

1. **Robot fallback.** Use the locally available PyBullet 3.2.7 KUKA iiwa model,
   copied into assets, rather than introduce an unverified UR5 download. Seven
   controllable joints imply seven incremental actions. Asset license is retained.
2. **Predefined reachable target.** Compute the fixed target panel position and
   xyzw quaternion once from the YAML reference joint configuration using Bullet
   FK. That reference is a construction aid, never a PPO observation or a classical
   IK answer. Target is approximately (0.846762, 0, 0.546459) m with quaternion
   (0, 0.706825, 0, 0.707388). The panel is vertical. The oriented wall is 25 mm
   behind the panel surface. The optional decorative floor is omitted.
3. **Rigid attachment frames.** Convert the configured URDF link-frame panel
   transform to Bullet's inertial parent frame before creating the fixed constraint.
   Only the gripped terminal link/panel pair and adjacent arm links are excluded
   from collision checks. Nonadjacent self-collision remains active.
4. **Actuation and collision semantics.** Both methods issue incremental targets
   of at most 4 degrees/joint to the same physical position controller, with 12
   simulation steps at 240 Hz per decision. Check contacts at every physical step
   and closest points for kinematic queries and end-of-step diagnostics. Stop on
   collision or limit violation. Queries restore robot and panel state. The RRT
   edge checker is discrete (0.035 rad Euclidean spacing), not a continuous-time
   collision certificate. Physical execution can still collide and is measured.
5. **Fixed noise model, no delay.** Independent Gaussian translation (5 mm/axis)
   and rotation-vector noise (0.5 degrees/axis), redrawn each control observation.
   Collision geometry and q/dq are exact. PPO sees each fresh measurement. RRT
   uses the first measurement to solve IK and plan once; subsequent joint feedback
   uses exact q, with no target replanning. Each method receives the same initial
   noise realization and indexed stream under the same seed. Ground truth is used
   only for reward, stopping, and evaluation, never the classical IK target.
6. **Sparse curriculum.** A: no obstacle; B: one fixed obstacle at (0.5,.25,.35);
   C: 1–3 boxes sampled with rejection of start and goal collisions. A previous
   fixed box at (.45,.25,.65) overlapped some initial arm poses and was moved after
   the smoke test exposed it. Endpoint checks do not guarantee global feasibility;
   planning failures are retained in results, not filtered out. A separate real
   robot detour test verifies a blocked straight path can be planned around.
7. **Observation scaling.** q maps to [-1,1] by joint limits, dq is divided by
   2 rad/s, orientation error by pi; positions and box sizes remain in meters
   (workspace scale about 1 m). No VecNormalize, hidden goal joints, or delay.
8. **Fairness and timings.** Accumulate path lengths at physics substeps for both
   methods. Execution wall time includes high-level action inference and stepping;
   IK+RRT time is separate. Nominal simulated time is also saved. A planner failure
   has zero execution steps and final errors at its initial pose. PPO IK/planning
   fields are missing, not incorrectly reported as failures.
9. **Training scope.** CPU training and actual fixed-seed evaluation are recorded
   in model provenance and raw CSVs. SB3 rounds requested training lengths to full
   rollout buffers. No claim of convergence or superiority may be inferred from
   these basic single-seed runs. See REPORT_DRAFT.md for measured outcomes.
10. **Development environment.** Project-local virtual environment initially uses
    installed scientific packages plus downloaded PyBullet/SB3/CPU PyTorch. Exact
    tested direct versions are pinned in environment.yml and requirements.txt.
    The code uses Gymnasium; an unrelated old Gym in the host base environment
    can emit a deprecation notice during SB3 imports.
11. **Contact regression and planner clearance.** Reading physics contact manifolds
    exposed a transient 0.17 mm penetration that was no longer present in the
    post-integration closest-point query. Preserve that stricter collision event.
    Add a 5 mm clearance to all configuration validity queries (including start,
    goal, IK, and RRT); physical collision still means contact/penetration. This
    is a conservative planning buffer around the shared collision geometry.
    Reduce waypoint tolerance to 0.008 rad. The physical detour regression now
    passes with both the path checks and transient-contact checks enabled.
12. **Exploration diagnostic.** The first phase1_clean run used SB3's default
    Gaussian log standard deviation 0 (standard deviation 1 in action units).
    At 20k steps its five callback episodes had improved return but all timed out.
    Reachability, reward sign, normalization, collision termination and scripted
    progress were already tested. A separate phase1_precision run uses initial
    log standard deviation -1 to reduce random joint jitter while retaining all
    task geometry, randomized starts, action limits and success tolerances.
    Preserve both runs as measured diagnostics; do not erase the unsuccessful run.
13. **Display runtime.** The initial host OpenGL load failed to retrieve a
    framebuffer configuration. The GUI passes using the system libstdc++ with
    Mesa software rendering (see README commands). The screenshot is a real
    simulator render; no image-generation assets are used.
14. **Near-target reset curriculum after diagnostics.** Both wide-start phase-1
    runs showed positive return trends but zero success in early callback
    evaluations (up to 40k steps for reduced exploration). Physical/scripted
    success, reward direction, dimensions, random start validity and finite
    normalized observations pass regression tests. To expose the rare terminal
    reward without weakening the 1 cm/2 degree task, add phase1_alignment.yaml:
    zero start offset and uniform ±0.035 rad reset perturbations. This is a
    TRAINING-ONLY initial-state curriculum, not the original benchmark start
    distribution. Return to normal start offsets before obstacle training.
    Never report alignment-curriculum success as benchmark performance.
15. **Fresh Conda verification.** Created the environment from environment.yml
    under `.conda-env`. The first pytest invocation imported unrelated ROS Jazzy
    plugins through the host shell's PYTHONPATH and failed on ROS's missing `lark`.
    Clear that inherited variable in ROS-sourced shells; no ROS dependency belongs
    in this project. Re-run smoke, tests and pip check in the isolated environment.
    requirements-lock.txt records its actual transitive package versions.
16. **Noise RNG audit before noisy training.** Found and fixed a reset(seed=None)
    path that recreated the noise RNG from OS entropy. It now continues the seeded
    independent noise stream across training episodes. A regression test compares
    repeated unseeded resets of two initially seeded noisy environments. Earlier
    diagnostic training was clean, so zero-valued observations were unaffected;
    all final noisy training uses the corrected RNG.
17. **Preserve alignment experience while introducing obstacle observations.**
    The first fixed-obstacle fine-tune lost strict success despite positive return:
    all three five-episode callbacks timed out. Obstacles introduce nonzero input
    features that were always padded during phase 1. Add a documented mixture of
    25% near-target and 75% original-start resets during obstacle TRAINING, so
    precise terminal alignment remains represented while learning obstacle inputs.
    stage_b_curriculum.yaml and both final train configs use this mixture.
    Both evaluation configs explicitly override near_start_probability to zero;
    benchmark scenarios remain unchanged. Keep the unsuccessful phase2_obstacle
    run rather than presenting it as converged. The recovery run branches from
    the successful phase1_restored checkpoint.
18. **Checkpoint selection fixed before the primary benchmark.** Evaluate each
    final variant's best/best_model.zip, selected by SB3 callback mean return on
    its training-condition validation episodes (seed stream beginning at 1042).
    This is independent of benchmark seeds 10000–10099. Keep final_model.zip too.
    Do not select the winning checkpoint using test-set performance. Both methods
    use the same rule, architecture, common source model and additional budget.
19. **Root cause of stage-A to obstacle transfer.** Obstacle columns were always
    zero in Stage A, leaving their orthogonal random weights untouched. Their
    activation shifts the trained policy even when the fixed obstacle is safely
    away from its path. The --zero-obstacle-inputs flag clears these actor/critic
    first-layer columns ONCE when transferring a no-obstacle checkpoint. This
    provably leaves outputs on padded Stage-A observations unchanged (checked
    numerically), keeps all columns trainable, and produced 20/20 successful
    fixed-obstacle diagnostic episodes before further training. phase2_transfer
    applies this fix and the documented mixed reset curriculum. Both final
    variants branch from its final checkpoint without clearing weights again.
20. **Demo seed versus benchmark.** Default GUI seed 10002 is a successful
    demonstration scene, selected after observing benchmark outcomes. This is
    presentation-only selection and is disclosed in README. All 100 benchmark
    seeds remain in CSV/report denominators, including failures. Representative
    benchmark trajectory plots retain the preselected seed 10000.
21. **Farther starts and three random obstacles (user-requested extension).**
    Increase the first-joint start offset from 0.45 to 1.05 rad and require the
    measured panel-to-target distance to fall in [0.80, 1.20] m. Verify without
    obstacles first, then exactly three randomly positioned boxes. The user's
    follow-up superseded the initially requested fixed-position layout; the final
    configuration uses randomized positions, not three hard-coded centers.
    Preserve the original base/train/evaluation configs and all trained models.
    A 4 cm inter-obstacle gap and the existing 5 mm start/goal planning clearance
    reject overlaps. Neither reset nor sampling rejects a scene merely because
    its RRT execution fails. Audit the direct edge to the planner's actual IK
    goal, not just the construction reference posture, and retain failures.
    --random-seed enables fresh GUI scenes with the chosen seed printed for replay.
22. **User-selected 1.85 m alternative and 0.6x playback.** The original fixed
    target is 0.867048 m from the shoulder center (0,0,0.36). The remaining URDF
    joint-origin lengths plus panel offset sum to 1.041 m, giving a conservative
    geometric start-to-target upper bound of 1.908048 m. An average 2 m distance
    is therefore impossible without changing the target or robot. After a concrete
    target-shift alternative was presented, the user chose to keep the target and
    use approximately 1.85 m instead. The farther-start config now centers q on
    [2.65,1.55,0,-0.15,0,-0.13,0] rad, with ±0.03 rad randomization and an actual
    distance filter [1.80,1.90] m. The earlier 0.922 m results remain separate.
    GUI playback defaults to 0.6x, configurable with --playback-speed or
    demo.playback_speed. Wall-clock pacing accounts for computation time;
    simulation timestep, action duration, rewards and headless training are
    unchanged. Pauses before/after the animation are not scaled. The camera is
    widened for the new span. A GUI check of the previous scene matched headless
    success/steps and displayed 1.3 simulated seconds in about 2.17 wall seconds.

Final verification: unchanged target pose checked against the base environment; 18 tests passed.
Seeds 10000–10099 averaged 1.841634 m (1.833300–1.849452 m). Both stages achieved
100/100 success with zero collisions. GUI seed 10002 executed 60 steps in 5.0241 s
for 3 simulated seconds, consistent with the configured 0.6x playback.

## Sparse unequal obstacles and paired observation-noise demo

Three random boxes now use dimensions 0.08/0.08/0.10, 0.12/0.10/0.16, and 0.16/0.14/0.20 m,
a wider sampling region and a 0.15 m minimum surface gap. Optional per-obstacle sizes
leave older configurations unchanged. A noisy child config applies existing 5 mm/0.5 degree
observation noise without changing geometry. Same-seed scene equality is tested and audited.
20 paired seeds yielded clean 20/20 and noisy 12/20 successes, with no collisions or planning
failures. 19 tests passed. Existing user edits to playback speed and demo.png are preserved.

## Five moving obstacles and reactive RRT

Implemented user-selected five unequal boxes in evenly spaced longitudinal bins
along the initial-to-target reference corridor. Lateral alternation is necessary
to avoid overlap with the extended robot; nominal offset is 0.60 m, with small
random position jitter and random sinusoidal phase. Amplitude 0.10 m, peak speed
0.05 m/s, updated on every physics tick. Initial endpoint checks cover swept boxes.
A first snapshot-only planning prototype collided while holding as a box approached.
The final planner therefore uses conservative full-motion envelopes and retains
short-horizon execution checks with hold/replanning. This uses known prescribed
motion and synchronous (paused during planning) simulation. It is not a real-time
safety certificate. A regression exercises replanning via an obsolete unsafe route.
Default demo selects the new config and a fresh random seed unless explicitly set.
Old static configs and PPO models are retained; new observation dimension is 55.

## RRT-reference placement and faster motion

Replaced straight-corridor placement with five arc-length stations on the carried
panel FK trajectory of an obstacle-free RRT-Connect plan from start to goal_q.
Noisy target measurements do not affect scene generation. Anchors use small jitter
and bounded endpoint-feasibility offsets. Peak speed increased to 0.10 m/s and
amplitude to 0.20 m, preserving the sinusoidal period. GUI displays the reference
in yellow. Stored scenarios include the reference joint path and panel trajectory.

## Three-dimensional random obstacle motion

The user requested up/down, forward/back, left/right motion. Added random_3d mode
with three independently seeded sinusoidal phases and angular frequencies. Existing
user values amplitude=0.40 and max_speed=0.40 are preserved as total Euclidean
bounds; per-axis amplitude is amplitude/sqrt(3), and angular frequency is sampled
in [0.5,1]*max_speed/amplitude. This is smooth randomised periodic-component motion,
not frame-to-frame random teleportation. Placement uses either side of the RRT
reference and up to 360 attempts to keep the larger 3D envelopes clear of endpoints.
IK now checks the same motion envelopes used by RRT, avoiding goals inside a sweep.
The replan regression explicitly selects a colliding obsolete waypoint; a zero
joint vector was no longer necessarily unsafe in the new geometry.

## Timed passage through moving-obstacle regions

Added TimedRRT: generate RRT-Connect geometric candidates at several predicted
snapshots, then A* scheduling over path-index/time with advance and wait edges.
Edges sample both robot/payload and obstacle positions at matching intermediate
times. No full swept-region inflation is used for route planning/IK in timed mode.
Final verification uses the current configured speed 0.40 m/s. Swept envelopes remain endpoint-only
placement constraints and audit geometry. Feedback tracks timestamps rather than
skipping reached waypoints; stale/unsafe schedules trigger rescheduling.
Whole-region mode is retained as dynamic_mode: swept for comparison.
A synthetic timed gate regression proves waiting for an opening and rejecting an
unsafe wait; physical rollout auditing records actual overlap with swept proxies
only when no actual collision occurred. Planning pauses simulation time. Initial
parallel validation showed finite-budget planning failures, preserved in logs.

Final timed validation at 0.40 m/s: 7/10 success, 3 initial planning failures,
zero collisions; all 7 successful executions entered swept regions. 164 scheduled
wait steps total; seed 10002 waited 40 steps. Conservative comparison: 10/10
success but zero region entries. 22 tests passed. No performance superiority or
continuous safety guarantee is claimed.

## Approved 70-input dynamic PPO retraining

The dynamic study adds three instantaneous world-frame velocity values per obstacle,
raising the policy input from 55 to 70. Legacy configs retain their original input
sizes. This is continuous-action reinforcement learning: the agent receives joint
state, target pose error and obstacle information, and selects seven normalized
joint increments. It receives the existing progress/collision/limit/success reward.
No RRT action labels, imitation loss or online planner are used by the PPO policy.
Reference RRT paths are used for scene construction and curriculum reset poses.

The user approved a 50,000-step pipeline pilot followed by two fresh 1,000,000-step
runs (clean/noisy target observations) on this CPU, with no wall-time deadline.
Each formal run uses 200k obstacle-free alignment steps, 200k static-obstacle steps
and 600k fully dynamic steps. Alignment resets progress along collision-checked
reference paths instead of linearly interpolating joints through invalid states.
The static stage freezes the final scene layouts at their initial phases. Four
workers collect 250 transitions each per rollout, making every phase budget exact.
The per-model resolved config records these training overrides; the frozen scene
config was captured before the rollout-length adjustment and is used for geometry.

Training uses 128 frozen scenes; 10 separate validation scenes select checkpoints
by success rate, then mean reward, only during the final dynamic stage. A further
100 held-out scenes are shared by all six method/noise combinations. Seed ranges
are disjoint, and the accepted scenes, generation failures and hashes are saved.
These are tests of the accepted scene distribution; endpoint clearance and reference
path construction do not prove that every moving-obstacle problem is solvable.
Formal models use one training seed each (42); this is not a multiple-seed study.

The 5 mm/0.5 degree noise affects target observations, not ground-truth rewards or
metrics. Obstacle positions and velocities remain exact; full phases/frequencies
are not observed by PPO. RRT retains prescribed future motion and paused planning
time, so the comparison does not give the two methods equal predictive information.
Reward, success tolerances, final scene difficulty and authorized budgets are fixed.
Old models and results are preserved. Training completion is distinct from learning
a successful policy; failed episodes and zero success rates must remain reported.

## Dynamic PPO failure diagnosis after user-requested evaluation pause

Diagnosed only the 10 validation scenes, preserving the paused 60-row formal test.
Selected clean/noise policies had 2/1 collisions and 8/9 timeouts, no successes
or joint-limit events. All timeouts exceeded both pose tolerances. Frozen motion
and a 720-step diagnostic horizon did not restore success. Original 200k alignment
policies succeeded 10/10 each without obstacles at these same far starts; selected
later policies succeeded 0/10 each in that setting, evidence of alignment forgetting.
These ablations do not change the formal task or success criteria.

Final saved weights differ from the last callback's pre-update weights: final Noise
succeeded on seed 40003 (independently reproduced), giving 1/10, while final Clean
collided on all 10. Future validation should explicitly score post-update/final
weights. This diagnosis preserves existing checkpoint selection and makes no new
training run or automated policy replacement. Raw outcomes, trajectories, figures
and scope limitations are in results/diagnostics/dynamic70_failure/.

## Code-only mixed-curriculum preparation

At the user's request, implemented but did not run a new training experiment.
The new script checks configuration by default and requires --train to learn.
It resumes each condition's alignment checkpoint, retaining optimizer state,
and prepares 50k additional steps with a 30/30/40 episode mixture of near-goal
alignment, far-start static and far-start full-speed dynamic tasks. Rewards,
actions and success criteria are unchanged. Source and output directories differ.

SB3 calls rollout_end before its optimizer update. Both the legacy callback and
new pipeline now use the next rollout_start for periodic post-update validation
and training_end for the final update. The new callback additionally reloads the
final saved model before scoring it. Regression tests simulate callback ordering
without PPO.learn or optimizer updates. Validation logs dynamic/static outcomes,
conditional timeout errors, and both near/far obstacle-free alignment success.
No formal evaluation or automatic long run is triggered by the new entry point.
