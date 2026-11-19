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

## Static waypoint PPO interface preparation

Implemented the requested four interfaces without training. Keep published
WaypointTrackingPolicy unchanged for historical reproduction; use RawWaypointPolicy
for PPO, with 14 gain/bias actor parameters plus critic and an explicit persistent
freeze mask. Raw Gaussian actions/log probabilities remain in SB3's rollout buffer;
all actuation projection belongs to WaypointPPOEnv. This removes the earlier
forward/predict deployment mismatch without assigning an incorrect log probability
to a modified action. Same-waypoint progress is computed before advancing the cursor.
Planning failure yields a terminal zero-reward transition without physical motion.

The new opt-in entry point defaults to checks only. Post-update validation shares
one planned path across ordinary tracking, original BC and current PPO, including
far/near alignment retention. Final saved weights are reloaded before evaluation.
No independent test seeds, original dynamic-study paused evaluation, or published
weights are modified by interface preparation.

## First authorized waypoint PPO pilot

Completed exactly 10,000 additional transitions (seed 61), 162 completed episodes, in 904.37 seconds including three full paired validations. Training episode outcomes: 159 success, 2 collision, 1 planning failure. Baseline, 5k and final validation all achieved 89/90 static success and 20/20 alignment success with zero validation collisions/limits/timeouts. One RRT planning failure remained. Selected 5k by control-step tie-break (60.0112 versus BC 60.0225); final mean was 60.0225. No meaningful performance gain is established. Gain/bias weights changed; velocity/std remained frozen. Saved final weights were reloaded for validation. No independent test, dynamic extension, automatic budget increase or GitHub push was performed.

## User-selected autonomous dynamic PPO (2026-10-27)

User chose autonomous PPO avoidance, smooth randomized 3D motion, gradual .10m/.05m/s
to .40m/.40m/s, reference-path placement, +5k static checkpoint initialization,
20/30/50 mixed episodes and 10k additional steps, without observation noise.
Implemented a zero-output residual actor over all 77 inputs, with a frozen tracking
prior and copied critic. Optimizer reset is required by the architecture change.
RRT is restricted to obstacle-free scene-reference generation; no online avoidance
planner or expert override is used. Residual scale 2 permits waiting and reversing
saturated prior actions. Joint-limit projection and reward definitions are retained.
Added a passing-waypoint cursor so a detour does not require exact return to every
old point. It constrains bookkeeping only. Training anchors are generated with full-
sweep endpoint safety, and motion phase changes every episode. Validation includes
both fixed motion strengths and alignment retention, shared across three reference-
guided controllers. This is not an independent generalization test or comparison to
a time-aware RRT avoidance planner. Report failed episodes and non-improvement.

## Autonomous dynamic pilot outcome

Completed exactly 10k additional transitions from source counter 5k (total 15k), 767 complete episodes, in 35.06 seconds excluding scene generation. Cached reference scenes and early collisions avoid per-episode planning cost. Training: alignment 142/142 success, static 7/249, dynamic 4/376; remaining static/dynamic episodes collided. Validation at 0/5k/10k: easy/full/static each 0/10 success and 10/10 collisions, far/near alignment each 10/10. Selected initialization; retain final trained weights separately. Residual network changed, frozen prior/std did not. Do not claim autonomous avoidance was learned or compare directly to the previous collision-aware RRT-assisted 89/90. No budget extension or independent test was run.

## Code-only staged avoidance v2

At explicit user request, applied curriculum, offline expert imitation, stronger
exploration, reward adjustment and acceptance gates without running a new experiment.
Early stages use 1 static / 1 slow dynamic / 3 / 5 low / 5 full obstacles; final
physical success and full-motion settings are unchanged. Offline experts may use
motion prediction, but actor observations retain obstacle-free references and have
no future motion or expert path. Successful labels and failed attempts are separate.
Only residual/gate parameters receive BC updates, with scene-level heldout checks.
A learned prior gate starts at 0.5, std is 0.10, waypoint weight 0.25, and obstacle
clearance adds a bounded [-0.5,0] reward over all arm links and payload within 8cm.
Wall clearance is excluded from this shaping; physical collision rules remain.
Two distinct updated-policy validations, sufficient counts, collision and alignment
thresholds, and a passing selected checkpoint are required for manual promotion.
Independent test seeds and all previous models/results remain untouched.

## V2 failure diagnosis and code-only v3 correction (2026-10-27)

User authorized the one-static pipeline: 100 successful teacher episodes (4,829
transitions, 50 seeds), 2,000 BC updates, then exactly 10k PPO transitions. BC and
final PPO both scored 0/20 target and 0/20 alignment; the gate correctly stopped
promotion. Source and pre-BC gated initialization retained 20/20 alignment.
All 100 stored expert episodes replayed successfully with exactly matching
observations; action/log-probability, projection and serialization interfaces
matched. BC suppressed its prior (mean gate .0197 on teacher states) and output
large actions at the goal. The obstacle-free reference frequently lagged the
independent expert route. No unit/sign or label-pairing defect was found.

At user request, corrected code without further collection or optimization:
preserve legacy checkpoint classes; add ReferencePreservingPolicy with a
no-obstacle correction mask, squared terminal correction taper and bias-free
endpoint servo. Offline alignment labels now use that reference servo; avoidance
experts receive the original reference endpoint explicitly. A monotone forward
projection cursor recovers after detours without changing paths/actions or
calling a planner. BC uses balanced tasks, successful terminal anchors and mild
gate/residual priors. A fresh BC closed-loop gate is mandatory before any PPO
output, worker or learn call; failed metrics remain reviewable. V3 has separate
data/model roots and rejects v2 teacher/BC artifacts. Final installation and
stage promotion criteria and the 10k budget are unchanged.

Inference-only checks on existing validation cases: v3 initialization 20/20
alignment; old BC weights transplanted in memory into v3 also 20/20 alignment,
but only 2/20 obstacle success (initialization 0/20). No transplant was saved or
approved for training. Goal actions ~1.71e-6; old teacher trajectories with cursor
below 90% reduced from 78/100 to 10/100. These diagnostics establish specific
interface/retention properties, not trained avoidance competence.

## First authorized v3 run: BC release rejected (2026-10-27)

Prepared 64 training / 20 target validation layouts with zero generation failures.
Collected exactly 100 successful expert episodes in 100 attempts: 20 alignment,
80 avoidance, 4,794 transitions and 50 unique scene seeds. Completed 2,000 actor-
only BC updates; scene-heldout selection objective 0.00801880436. Deterministic
validation: target 0/20 success, 20 collisions; far/near alignment both 10/10 with
zero collisions, limits or timeouts. Goal probes ~1.71e-6 passed. The BC gate failed
on target success/collisions, and the pipeline stopped before invoking --train.
PPO additional steps are zero; no PPO output directory or harder stage was created.
Do not bypass the gate or interpret action-MSE reduction as obstacle competence.
Visible report/curve/CSV: main checkout results/diagnostics/staged_avoidance_v3_seed61/one_static/.

## Authorized BC budget comparison and corrective-data expansion

User authorized code changes and training. Added an explicit study entry point,
frozen teacher-reuse contract with historical source-hash attestation, BC Adam/RNG
checkpoints, RNG-isolated reloaded validation at 2k/5k/10k, and closed-loop-first
selection. Tests: 69 full regressions passed; focused RNG/release tests passed.
The same-data 2k checkpoint exactly matched every tensor of the original v3 BC.
At 5k target still 0/20; at 10k target 10/20 success, 9 collisions, 1 timeout;
alignment 20/20. The unchanged collision gate rejected release, triggering the
already authorized expanded-data branch. Its validation bank hash matches the
original bank. Expansion is bounded to 1,000 successes including 200 alignment,
600 ordinary avoidance and 200 corrective recoveries, at most 2,500 attempts,
256 prepared layouts and >=200 distinct avoidance seeds. Prefix BC commands are
stored separately and never used as expert labels; real physical state/velocity
and original reference are retained at handoff. Failed expert/prefix attempts
remain failures. A second 10k BC budget is allowed; PPO runs only after fresh BC
release, for exactly 10k transitions and without automatic stage promotion.

## Bounded budget study completed: both BC gates rejected

Expanded teacher: 1,003 attempts, 1,000 successes (200 alignment, 600 regular
avoidance, 200 corrective), 46,595 labels, 254 total seeds / 248 avoidance seeds.
Ten sampled correction prefixes plus expert sequences replayed with 10/10 success,
exactly zero observation error and matching total step counts. Source/data/code
provenance retained. Expanded BC target success at 2k/5k/10k: 0/20, 2/20, 0/20;
alignment remained 20/20. The closed-loop selector retained the 5k model, not the
final lower-MSE but zero-success model. Its 17/20 collisions failed release.
Both authorized BC budgets completed; PPO additional steps are zero. No harder
stage, budget increase, gate relaxation, independent test or GitHub push occurred.
Fixed-data 10k remains best (10/20 success, 9 collisions, 1 timeout). An export of
that BC checkpoint with failed-release metadata is visible in the main report
directory. Larger-data average exposure per training state is ~10x lower under
the same update budget; coverage and corrective data changed together, so this
study cannot isolate their individual effects or prove that more data is harmful.

## Targeted BC continuation completed (2026-10-30)

Use the original selected 50% BC and its saved BC Adam/NumPy/Torch state for both
10000-update continuations, not the separately downloaded PPO branch. Preserve
the original supervised held-out layout membership. Diagnosis reproduces all
20 original outcomes: six panel-obstacle collisions, two self collisions, one
arm-panel collision, one timeout, ten successes. Failures extend to 180 steps.

Collect 100 successful offline fixed-endpoint RRT recoveries in 107 attempts from
34 training layouts, using failure/clearance/deviation-dependent handoffs. The
policy prefixes span 11–123 steps; 99/100 exceed 12 steps. Their commands are
replayed to preserve velocity/time/reference state but are never expert labels.
There are 3523 corrective labels. Ten complete corrections replay with success
and zero observation difference. No validation or independent teacher queries.

Original-only continuation completes its budget but retains update zero at
10/20 success. Targeted mixture selects additional update 10000 (20000 cumulative
BC updates), reaching 13/20 success, 2 collisions, 5 timeouts, 0 limits and 20/20
alignment. It passes the unchanged BC gate. A separate frozen 100-scene bank from
980000 is first scored after both selections: baseline 36 success / 61 collision /
1 joint-limit / 2 timeout; targeted 60 / 13 / 0 / 27. New independent alignment
20/20. Independent test outcomes do not select a checkpoint or winner.

PPO steps remain zero; no automatic promotion or further budget. Five-full
diagnostic success stays 0/10. The current improvement is for one static obstacle;
timeouts and training instability remain limitations. Full tests: 73 passed.
Visible report: `results/diagnostics/targeted_bc_v3_seed61` in the main checkout.

## Stall recovery pilot completed (2026-10-30)

Diagnose 54 training layouts from the selected 65% BC: 39 successes, 8 collisions,
7 timeouts. Most stationary timeout actions show almost equal/opposite gated prior
and residual. Do not change cursor/actuator semantics or relax the 240-step task.
Collect 40 training-only expert recoveries in 40 attempts, at most step 160,
preserving prefixes and velocities. There are 339 suffix labels. Ten audited
prefix+suffix replays succeed with zero observation error. New merged teacher
has 240 success episodes / 8656 labels, with unchanged supervised held-out seeds.

Low-LR BC resumes the 20k Adam/RNG state, completes another 10k updates at 1e-4,
but all updated candidates lose to the starting checkpoint under existing criteria.
The separate PPO variant starts from the SAME 65% BC, passes its fresh BC gate,
and performs exactly 10000 environment transitions, with static-only stagnation
feedback and no online expert/controller fallback. Evaluation uses the original
reward and task. At 0/5k/10k PPO steps, validation success is 13/20, 13/20, 14/20;
collisions 2, 2, 3; timeouts 5, 5, 3. Both alignment suites remain 10/10.

Selections are locked before scoring a fresh 100-scene bank from 990000. Baseline
and retained BC: 67 success / 9 collision / 24 timeout. Selected 10k PPO: 78 / 13 /
9, no limits, independent alignment 20/20. Report the increased collision count
alongside reduced timeout, never compare the new bank directly against the old
980000 bank as though scenarios were identical. No causal isolation of reward,
data or algorithm, and only one optimizer seed. No automatic stage advancement.
The selected lineage is 20000 BC updates followed by 10000 new PPO steps; stored
PPO total is 15000 including the original 5000-step source counter. The failed
low-LR BC arm is not an ancestor of the selected PPO. Full tests: 83 passed;
training code hashes and frozen prior/action std verified. Visible report/export:
`results/diagnostics/stall_recovery_v3_seed61` / `selected_model.zip`.

## Matched PPO wall-feedback pilots completed (2026-10-30)

Collision diagnosis: 5/5 training collisions are panel-wall; original validation
has two panel-wall / one panel-obstacle; old known test failures are 12 panel-wall
and one self collision. This supports targeting wall approach, not increasing
the obstacle-only proximity penalty. Training feedback applies below .02m while
the nominal target gap remains .025m. Successful terminal installs get zero cost.
No actor architecture, action projection, collision margin or horizon change.

Two variants resume the same selected 15k-counter PPO checkpoint and optimizer,
using seed 61 and 10k additional transitions each. Only wall weight changes 0 to 1;
stall feedback and obstacle penalty .5 are identical. Evaluate using original
reward. Predeclare eligibility: retain alignment, no limits, success >= baseline,
collisions <= baseline, timeouts <= baseline; then prioritize fewer collisions.

Control 5k/10k validation regresses to 12/20 and 10/20 success, retaining baseline
14/20. Wall-feedback 5k has 14 success / 2 collision / 4 timeout and is ineligible
because timeout worsens; 10k has 16 / 2 / 2, so it is selected. Lock selections
before scoring the fresh 100-case bank from 1000000: starting PPO and retained
control each 83 success / 6 collision / 11 timeout; selected wall-feedback 92 / 4 /
4. All independent alignment remains 20/20, joint limits zero. New-bank percentages
must not be directly compared with the old 78% as though scenes were identical.

Final model passes a single static-stage threshold check, but the 5k success rate
is only .70; two consecutive distinct post-update passes are absent. No dynamic
stage, budget extension, or auto-promotion. Full tests 88 passed; initial optimizer
states and physical first-rollout episode records match across four workers.
Training-code hashes and frozen prior/std verified. Selected lineage: 20k BC
updates plus 20k new PPO steps, stored counter 25k including original source 5k.
Visible report/export: `results/diagnostics/clearance_ppo_v3_seed61` /
`selected_model.zip`, with metrics, hashes, paired-rollout evidence and curves.

## Unchanged static PPO stability continuation completed (2026-10-30)

Honor the requested single first step only: 10000 more static PPO transitions,
same weights/optimizer initialization from wall-feedback best, unchanged training
and evaluation settings, checkpoints at 0/5000/10000. Validation outcomes are
16 success / 2 collision / 2 timeout initially, 15 / 1 / 4 at 5k, 16 / 0 / 4 at 10k.
All checks retain alignment 20/20 and no limits. The existing selector requires
no increase in timeouts, so updated candidates are ineligible and step zero is
retained. Do not silently change selection to promote the zero-collision candidate.
That final candidate is exported separately as `final_candidate_unselected.zip`.

The first post-update success rate is .75, so two consecutive stage-gate passes
are absent despite the final .80 / zero-collision check passing once. No dynamic
stage, extra budget or new independent-test claim. Replaying the already-seen bank
reproduces the retained model's 92/4/4. Training completed 135 full episodes
(25 alignment, 110 static); these stochastic training counts are separate from
deterministic validation. Relevant tests: 17 passed. Code hashes, input hashes,
unchanged settings and frozen prior/std verified. Final unselected PPO counter
35k; retained counter 25k. Visible report: `results/diagnostics/static_stability_v3_seed61`.

## Demo and evaluation-only timeout/model comparison (2026-10-30)

Provide a terminal animation using the evaluated autonomous wrapper; never reuse
the old RRT+PPO demo that supplies a fresh collision-free expert route. Default
retained model and saved scene 1000000 reproduce 65-step success. Headless test
forbids runtime ClassicalController/RRT planning. GUI changes only the Bullet
connection and visual debug trail, preserving original action execution.

Compare fixed retained/candidate actors on identical 100 known + 100 fresh target
scenes plus alignment cases. Fresh seeds start 1002000 and do not overlap prior
banks. Known outcomes 92/4/4 versus 86/6/8; fresh 90/6/4 versus 88/3/9; combined
182/10/8 versus 174/9/17. This supports retaining the original model, with no
automatic model switch. Combined 200-scene results are descriptive, not entirely
new independent tests. Both banks retain all alignment successes and zero limits.

Diagnose all 20 original cases and extend only the four candidate timeouts to 480
steps with matched original prefixes. Original two timeouts remain; two collisions
become timeouts. All four extended runs time out. Three have strong action
cancellation, no forward cursor recovery and ~.39-.42rad route deviation beyond
.30rad recovery range, with max joint goal errors ~.24-.36rad outside .20rad
terminal blend. Prioritize a future controlled inference test of reference recovery
and terminal handoff. Do not infer that all cursor stagnation is an implementation
bug, nor extend the training horizon based on these failures. This study performs
no training/control change. Five targeted tests pass; model/code hashes verified.
Visible report: `results/diagnostics/static_candidate_comparison_v3_seed61`.

## Fixed-weight one-factor handoff inference ablation completed (2026-10-30)

Use the retained actor only. Compare baseline recovery radius .30 / terminal .20,
wider recovery .45 / terminal .20, and earlier terminal .30 / terminal .40. No
combined factor or optimization. Terminal radius affects both endpoint blending
and the existing residual taper; do not call its result a separately trained PPO.
All network tensors, source zip, default configuration, physical criteria and
240-step horizon remain unchanged. Goal convergence is probed in every group.

Baseline reproduces 16 success / 2 collision / 2 timeout on validation and
182/10/8 on the 200-case regression pool. Wider recovery: 17/3/0 validation,
161/27/12 regression. It rescues 900002 and 900003 but adds an original-success
collision and harms many other cases. Earlier terminal: 2/18/0 validation,
25/175/0 regression; alignment 55/60 versus original/wider 60/60. No joint-limit
failures. Neither treatment meets non-regression conditions, so do not adopt.

The results support investigating conditional recovery only after stagnation,
while preserving current endpoint control. This next hypothesis is not executed
in this study. No model training or automatic stage advance; all 200 cases are
already-seen regression/tuning data, not fresh independent evidence. Nineteen
targeted tests pass. Tensor fingerprints, checkpoint/code/case hashes verified.
Visible report: `results/diagnostics/reference_handoff_ablation_v3_seed61`.

## Stagnation-triggered recovery inference comparison completed (2026-10-30)

Use the unchanged retained actor and .20 terminal servo; retain ordinary .30
recovery. Only after step 60, with a 30-step window lacking target position/angle
progress, <=.02rad net joint displacement, stale non-final reference and >=.20rad
reference distance, temporarily try .45rad nearest-forward recovery. Max three
attempts / one successful extension per episode, cooldown 20 steps. Restore .30
immediately. No action/physical-state overwrite, route planner or terminal change.
Skip alignment, terminated states and dynamic tasks. Twenty-five targeted tests pass.

Original validation: baseline 16 success / 2 collision / 2 timeout, conditional
18/2/0. Known 200-case regression: 182/10/8 versus 188/10/2. Only after this passes
the predeclared non-regression checks is the fixed rule scored on 100 new scenes
from 1004000: baseline 93/4/3, conditional 94/4/2. New-test gain is just one case;
do not claim statistical significance. Nine distinct timeouts become successes,
without new collisions or original successes becoming failures. All 391 episodes
with no successful recovery match outcome, length and both final errors exactly.
All nine recovery attempts succeed and respect bounded limits. Alignment retained.

No network update, source zip overwrite, default controller switch, automatic
training or dynamic-stage promotion. Control observation rules change only at
verified stagnation; this is not a PPO-training improvement. Tensor/file/code/case
hashes verified. Frozen guard/settings and full event traces are stored in the
visible report `results/diagnostics/conditional_recovery_v3_seed61`.

## Conditional recovery training contract and bounded PPO result (2026-10-30)

User authorizes integration plus short PPO adaptation and stability checks.
ConditionalTrainingEnv composes ConditionalRecoveryEnv and WallAwareEnv, keeping
the original physics/action projection and wall/stall shaping; conditional
recovery only modifies the next local reference input. Evaluation uses the same
reference rule without training-only reward additions. Preserve the existing
77/7 interface, ordinary/conditional radii .30/.45, terminal .20 and 240-step
horizon. Full preflight suite passes 115 tests including actual recovery parity.

Run exactly 10k PPO transitions from the retained source and PPO optimizer,
checking 0/5k/10k. Same weights originally score 16/2/2 success/collision/timeout;
conditional baseline scores 18/2/0. Updated 5k scores 16/1/3; updated 10k 17/0/3.
Both meet the existing minimum static thresholds, yet both reduce success and
increase timeouts versus conditional baseline. Preserve strict selection:
selected additional step 0, original tensors, with the integrated conditional
controller. Do not label the source's control-rule gains as PPO learning.

Replay the three already-seen 100-case banks after selection, never use them for
selection: before and retained both score 282/300 success, 14 collisions, 4
timeouts; all alignment retained. No fresh independent test. No automatic dynamic
stage, BC update or budget extension. All 10k samples contain one recovery attempt
and zero successful advances, so experience after recovery remains absent.
135 episodes finish; 264 transitions belong to partial episodes. Source file,
frozen parameter, input/code and paired-output checks pass. Default demo now loads
the selected export and its conditional manifest. Full report and controller
sidecar: `results/diagnostics/conditional_ppo_adaptation_v3_seed61` in main checkout.
