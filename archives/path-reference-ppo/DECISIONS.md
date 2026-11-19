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

## Professor-directed safety selection and RRT/PPO episodes (2026-11-02)

User prefers the professor's zero-observed-collision candidate (17/20 success,
0 collision, 3 timeout) over the previous 18/20, 2 collision, 0 timeout selection.
Re-rank existing checkpoints after retaining >=80% success, <=10% collision,
alignment and goal convergence; rank collision rate before success/timeout.
Reproduce the selected candidate and export separately in safety_first_ppo_20261102.
Preserve prior models/reports. The old 282/300 score is not assigned to this model.

User specifies independent 50/50 episode choice in training AND evaluation, and
prefers the existing RRT over a standalone BC controller if usable. Implement
RRT-Connect expert vs PPO whole-episode control. Log planning failures as RRT
failures without fallback. Only successful RRT episodes supervise residual/gate
parameters via a separate BC optimizer. PPO's standard on-policy buffer contains
only PPO's sampled actions/log probabilities; perform auxiliary BC at the next
rollout start, with sampled KL rollback, before any new policy sampling.

Keep the static course and 54 training layouts, excluding held-out scenes. Four
synchronous workers retain 4 x 250 PPO batches. Future default budget: 10k PPO
transitions plus separately recorded RRT transitions. No formal training starts
in this implementation task. Teacher/model/mixed evaluation stay separate;
pure PPO metrics choose checkpoints, with common mixture assignments at checks.

Fixed known validation, 20 targets plus 20 alignments per mode: PPO 17/0/3,
RRT 20/0/0, mixture 18/0/2 success/collision/timeout on targets. Mixture targets
draw 8 RRT / 12 PPO; all 20 alignments pass per mode, no limits/planning failures.
This is inference-only comparison, not evidence of newly learned performance.
Full 128-test regression passes, plus one added eight-artificial-step integration
test confirming saved likelihoods remain on-policy across BC/PPO updates.
No online planner runs during pure PPO execution. Mixed/RRT execution explicitly
labels online planning. Design and commands: docs/EPISODE_RRT_PPO.md.

## First episode-level RRT/PPO pilot completed (2026-11-02)

User authorizes training. Complete exactly 10,000 on-policy PPO transitions,
plus 7,030 RRT transitions (17,030 total). Start 147 RRT / 142 PPO episodes;
finish all 147 RRT successfully and 138 PPO episodes (101 success, 26 collision,
11 timeout), leaving four PPO partial episodes with 158 sampled steps. Collect
7,030 successful teacher labels. Ten auxiliary BC updates pass sampled KL,
none roll back; original source file, frozen parameters and code hashes match.

Known target validation at 0/5k/10k: pure PPO 17/0/3, 18/0/2, 18/1/1
success/collision/timeout. Mixture: 18/0/2, 19/0/1, 19/0/1 with common branch
assignments. Select 5k by collision-first ranking; reloaded model reproduces
18/20 pure PPO and 19/20 mixture, alignment 20/20 for all methods, no limits.
Pure RRT is 20/20. Mixture target assignments remain 8 RRT / 12 PPO.

This is one-case pure PPO improvement on known validation, not independent
generalization or causal proof of RRT/BC benefit. No new test bank, budget
extension or dynamic stage. Export the matched selected auxiliary state (five
accepted BC updates, PPO counter40k) alongside the selected PPO and manifest;
final raw counter45k preserves the unselected 10k candidate. Demo defaults now
load the selected run. Visible report/curves mirrored to the main checkout:
results/diagnostics/episode_rrt_ppo_v1_seed61.

## Prospective verification and three-static expansion completed (2026-11-02)

User authorizes the proposed expansion. First freeze 100 new paired source scenes
from seed1006000; exclude existing/train/validation scene IDs and require legal
start/goal configurations and sparse spacing, without filtering by learned/expert
success. No generation failures. Original selected one-static actor scores 95
success / 2 collision / 3 timeout on new targets, alignment20/20. This passes the
existing gate. Three-static train/validation minimum box gap exceeds .22/.23m.

Change active_stage count only in a separate source configuration bundle, keeping
weights, physical configuration, noise, rewards, observation/action dimensions,
reference/recovery and horizon. Continue PPO model/optimizer; fresh per-stage BC
optimizer/pool with unchanged hyperparameters is explicitly recorded. Use original
54 non-held-out training layouts and 20 validation IDs. The new paired100 banks
are never supplied to training or selection.

Three-static initial validation: 0 success, 1 collision, 17 timeout, 2 limits.
After5k: 0/14/6/0; after10k: 0/20/0/0. RRT solves20/20. Complete10k PPO + 5080
RRT transitions, 5055 successful teacher labels, ten accepted auxiliary updates,
zero KL rollbacks. All23 PPO training successes are no-obstacle alignment; zero
three-static successes. All candidate scores are ineligible. Diagnostic selection
retains step0, verified identical source tensors, not a usable three-static model.

Three-static held-out100 source/diagnostic: 0 success, 4 collision, 90 timeout,
6 limits. One-static source-retaining regression reproduces95/2/3, not evidence
that rejected learned updates retain old skill. Default source remains unchanged;
no moving obstacle stage or extra budget. Data/source/partition hashes verified.

Fixed-weight same-physics slot reordering [1,2,3]->[2,1,3] yields0/20 success and
20 collisions. It does not repair transfer and is not adopted; it diagnoses
sensitivity without proving the sole cause. Consider a two-box bridge and stronger
multi-obstacle BC warm-up before further difficulty. Reports:
results/diagnostics/static_expansion_v1_seed61 and static_three_rrt_ppo_v1_seed61.

## Two-static BC attempt and protected initialization correction (2026-11-02)

User requests two static boxes, stronger RRT BC then PPO, and confirms continuation.
Add stage count2 with indices[2,3], preserving original middle obstacle in slot0;
extend stage validation. Collect256 successful trajectories in258 attempts:
128 two-static RRT,64 one-static RRT,64 reference-servo alignment. Physical replay
is exact for sampled episodes.54 gradient layouts and10 held-out layouts remain
separate.10k BC updates at1e-4 with60/30/10 task sampling, slot permutation and
correction-state weighting select2.5k: two7/20,12 collisions,1 timeout; one6/20,
13 collisions,1 timeout. Fresh100 two51/44/5; one95/2/3->42/52/6. No PPO in v1.
Archive first-trial code and all model/data records before correction.

Fixed-source diagnosis: zeroing previously inactive actor input columns30:70
raises two-static validation0/20->15/20 (4 collisions,1 timeout), one remains18/20.
Do not call that gain learning. Corrective v2 initializes those columns to zero,
allows gradients only to residual/gate first-layer columns30:70 and freezes all
other actor connections. A fresh Adam avoids stale moments moving masked weights.
PPO critic remains trainable; deterministic one-static action policy is preserved.
Reuse valid RRT teacher data (frozen reference unchanged); freeze NEW100 tests
from1010000, treating the prior1008000 bank as development evidence.

Run2,500 protected BC updates; best remains update0. Initialization15/20 meets
warm-start50% success/25% collision gate while preserving old skill, so start10k
PPO with LR5e-5,50/50 episode controller and seeded auxiliary imitation. PPO5k
15/20,4 collisions,1 timeout;10k14/20,4 collisions,2 timeouts. One always18/20,
zero collisions. Select PPO0. Raw final weights do change added columns, but all
original actor connections are unchanged; selected candidate added weights zero.

New100 two-static: original actor0/100 (100 collisions); initialization and final
candidate85/100,10 collisions,5 timeouts. Known one-static before/after95/2/3.
Selected mixture validation17/20,2 collisions,1 timeout. All gains retained are
initialization gains; second-obstacle information is still unused by selected
action weights. Release remains failed (known validation75%,20% collisions).
Do not switch the verified default or start dynamic motion. Full143 tests pass.
Reports: results/diagnostics/two_static_warmup_v1_seed61 and protected_two_static_v2_seed61.

## 2026-11-03: independent multi-obstacle adapter and targeted corrections fail generalization

The real second obstacle slot is populated and its old added-column gradients
are nonzero. There is no disconnected-input/gradient defect. The retained v2
actor has zero second-slot response; slot swapping changes its frozen base
response. Test a separate fully trainable correction network, enabled only for
at least two boxes, with zero output initialization and the original actor
entirely frozen. Disable slot swaps; BC/auxiliary updates target only the new
adapter and PPO may additionally train the critic. Preserve single-box actions.

Audit54 training layouts only. The second box blocks a one-box RRT route in29
layouts; the initial learned policy itself fails in4 layouts. Collect29 route
demonstrations plus8 policy-failure correction suffixes:37/37 physical successes,
1,947 states. Keep the original obstacle-free reference and do not label policy
prefixes as expert actions. Physical replay checks for both new kinds have zero
error. Merge with the original256 episodes:293 episodes /14,162 states. Retain
the original10-layout supervised holdout; sample new episodes at4x weight.

Run3,000 BC updates at1e-4; validate0/100/250/500/1k/2k/3k. Select250: two-static
15/20 success,2 collisions,3 timeouts versus initialization15/4/1. The second-box
15cm input probe now changes an action by up to.00893, rather than zero. This is
input sensitivity, not a safety guarantee. A learned validation gain plus old
skill retention permits the predeclared10k PPO pilot; keep50/50 RRT/PPO episodes
and strictly separate on-policy/supervised data. PPO5k14/3/3;10k14/4/2. Retain
PPO0, which is the learned BC250 actor, not an untrained adapter. Actual run
collects10,000 PPO transitions plus6,563 RRT transitions.

Freeze a NEW100-scene test from1012000 before learning, never select with it.
Initialization87 successes/8 collisions/5 timeouts; selected adapter76/14/10.
This fails generalization and release checks. One-static regression remains
95/2/3, with exact episode trajectories and original actor weights unchanged.
Full147 tests pass. Do not replace any default or start moving obstacles; keep
all old artifacts. The new test bank is now seen and becomes regression data
for later experiments, not a fresh independent test. Further work should first
broaden failed-policy training-state coverage and examine expert route
consistency; adding PPO steps alone is not supported by these results.

Report: results/diagnostics/two_static_adapter_v3_seed61/README.md.

## 2026-11-03: wider failure coverage and canonical RRT BC do not improve the policy

Add128 fresh training layouts to54 retained layouts. Freeze new64 validation
layouts from1016000 and100 test layouts from1018000 before fitting; preserve
legacy20 two-box validation and old one-box regression. Parent teacher audit
finds21/64 phase-jittered pairs with opposite active initial joint signs, but
none has identical inputs, so this alone does not establish conflicting labels.
Compare three RRT seeds at an identical state in each of182 training layouts:
166 have opposing initial directions. Select one route per layout by the
frozen geometric/reference cost; connect correction suffixes to that same tail.

The initial collection stops before fitting when a densified candidate edge
fails a repeated collision check. Archive its code/log/80-layout data prefix.
Fix unsafe-candidate rejection and add guarded pre-training collection resume.
Resume hashes and every saved observation/action/terminal array match exactly.
The new checks reject7 proposed routes; no unsafe labels are adopted. Complete
182 canonical demonstrations and99 successful corrections across44 layouts,
from47 failure layouts found by initialized/v3-failed policy mining. Remove old
multi-route two-box training labels. Final419 episodes/17,805 states; physical
replay of both new demonstration kinds has zero error. Full153 tests pass.

Run3,000 BC updates. New validation baseline58/64 success,3 collision,3 timeout;
raw final21/64,43 collision,0 timeout. Legacy baseline15/20,4/1; raw final6/20,
14/0. All learned checkpoints lose ability, so retain BC0. The learned-gain and
non-regression gate fails; do not start PPO. Final new test is87/100,6 collision,
7 timeout both before/after because the selected actor is initialization-only.
One-box regression stays95/2/3 with identical trajectories. Preserve all defaults
and prior models, and do not claim an87% BC/PPO training improvement. No dynamic
stage. These results do not support merely adding PPO steps. Teacher consistency
within each layout does not resolve cross-layout route changes or the trained
policy's own state-distribution shift.

Report: results/diagnostics/consistent_two_static_v4_seed61/README.md.

## 2026-11-03: use retained initialization with four static obstacles

At the user's request, change the active demo to four static boxes without
fitting any weights. Copy the v4 initialized_model.zip exactly, and verify its
entire policy state equals the retained BC0 candidate. Extend the projection
with4:[2,3,1,4], keeping the existing two-box slots first and leaving all legacy
one/two/three/five mappings unchanged. Five observation slots/77 inputs remain.
Source weights ignore added slots; do not claim learned four-obstacle avoidance.

Default demo now generates a fresh geometrically valid random scene before
rendering. --seed reproduces a layout; explicit frozen banks preserve archived
replays with no planner calls. An obstacle-free reference is prepared before
execution for fresh scenes; pure policy execution has no online avoidance
planner. Playback remains0.6x and sensing noise remains zero. All100 projected
four-box starts/goals are valid and all physical scenes contain four static
bodies. Paired seen-bank counts: two87/6/7 versus four82/11/7. No training or
policy-success filtering. Preserve old model files and historical demo wrappers.
Fix the pinned historical replay test to name its original frozen bank now
that the default is random. Full156 tests pass. Both fixed-seed and random
four-box headless demos complete successfully.

Profile: configs/four_static_initial.yaml.
Report: results/diagnostics/four_static_initial_v1_seed61/README.md.

## 2026-11-03: place all four static boxes on physical nominal policy motion

At the user's request, replace vicinity-only positioning with individually
verified blocking geometry. Trace initialization in an obstacle-free physical
episode, preserving its ORIGINAL observation reference. Place four unchanged
box sizes around20/40/60/80% of measured panel-center arc length, retaining
middle-first slot order. Every box has a recorded robot/payload contact at a
specific nominal joint state. Keep endpoint poses collision-free and0.1m box
spacing. Random defaults and seeded reproducibility remain. Prepare all queries
in DIRECT before opening GUI; draw the nominal path cyan and execution blue.
No weight updates, reward changes or online avoidance planning in pure policy.

New profile: four_static_on_path_v1_seed61. Of100 fixed underlying scenes,99
produce full nominal routes and placements; seed1018091 times out without any
obstacles and is reported as route-construction failure, not a filtered
with-obstacle outcome. Paired99: previous positions81 success/11 collision/7
timeout; on-motion24/75/0. Do not reuse the previous82% for the new placement.
All396 per-box intersection witnesses are verified. Full158 tests pass, and
fixed/random headless demos both confirm four static on-motion obstacles.
The fixed seed20261103 now collides at41 steps; the random verification seed
1619274898 succeeds at59 steps. Neither is selected for rate estimation.

Report: results/diagnostics/four_static_on_path_v1_seed61/README.md.

## 2026-11-04: validate four-on-path feasibility by actual RRT execution

Execute the recommended first step only; no BC/PPO training. Freeze the existing
99-scene on-motion bank and initialized weights. Replay initialization with
the same official termination reasons/step counts:24 successes/75 collisions.
Record actual first collision substep time, control step, obstacle slot and
arm/panel contacts, plus planned/executed joint paths. All75 are panel/box
contacts: dominant slot3 in47, slot4 in27, slot2 in1. Preserve default animation.

Run existing ClassicalController/RRT-Connect with one scene-seed+100000 primary
trial per scene, same endpoint IK branch,15s planner budget and240-step actuator
horizon. Plans found99/99; physical execution97 successes,2 collisions, no
timeouts/limits/planning failures. Do not reject/replace a primary plan based on
post-execution diagnostics. On the two failed scenes run exactly two extra
seeds each;4/4 diagnostic retries succeed, reported separately. All99 scenes
have successful physical-execution witnesses, but baseline score stays97/99.

Primary collision seeds1018018 (step9,slot3) and1018025 (step45,slot4) have
failed sampled planning-clearance rechecks. Two other successful primary
plans (1018029/1018087) also fail5mm recheck. At the failed-route recheck samples,
panel gaps are1.1175mm and2.2737mm; no physical penetration was detected at those
specific samples. Do not claim all continuous edges safe or a single tracking
root cause. Strengthen sampled/densified path verification before treating every
successful route as a safe teacher. Planner, controller and policy are unchanged.
Full161 tests pass. Report raw primary/retry counts, feasibility lower bounds,
and all collision records. No scene is called infeasible merely for one failure.

Report: results/diagnostics/four_on_path_rrt_validation_v1_seed61/README.md.

## 2026-11-04: reject unsafe static RRT routes before execution

Keep original collision samples and add0.005-radian checks over the actual
execution-waypoint subdivisions. Validate direct routes, shortcuts and final
candidates; continue search after a rejected candidate within the existing
budget. Independently validate the final dense controller route before exposing
planning_success. Preserve original5mm planning clearance, actuator, reward,
physical collision threshold, source weights and default initialized-policy demo.
Dynamic/timed planner behavior is unchanged. This is sampled verification, not
a continuous collision certificate.

Repeat all99 regression scenes with the same primary seed rule and unchanged
15s search/240-step physical execution limits:99 success,0 collision,0 timeout,
0 joint-limit,0 planning failure. All99 paths pass both pre-execution validation
and the original post-execution check. Initialization still reproduces24/75.
No retries are needed. Old baseline97/99 and all archived routes remain intact.
Full166 tests pass, including all four archived clearance-failing routes and an
independent controller rejection guard. No BC/PPO training or model replacement.

Report: results/diagnostics/four_on_path_rrt_verified_v2_seed61/README.md.

## 2026-11-04: complete fresh four-static RRT demonstrations, without fitting

Freeze disjoint64 train/32 validation/64 test scene banks before collection;
exclude all original99 regression scenes. Layouts intersect the initialized
policy's nominal physical motion. Record three generation rejections caused by
incomplete obstacle-free nominal motion (train1020005/1020034,test1024000).
No four-obstacle validation/test expert episodes or labels are collected.

Collect training routes with10mm planning clearance and0.005-radian checks;
physical collision termination and installation precision stay unchanged.
Require pre-execution verification, actual installation success and postcheck.
One route per layout, up to three prescribed seeds; all64 succeed on their first
attempt. Final dataset64 episodes/3,973 transitions,77-input/7-action, with raw
commands and executed actions separately preserved. Keep policy reference intact.

The interrupted run stopped after52 episodes/3,222 transitions. Archive its
partial files/protocol, collect12 additional episodes in a separate segment and
verify exact equality of all merged array prefixes. Original and resumed segments
each have three physical replay checks with maximum error0. Preserve both code
versions and resume hashes. Full166 runtime tests remain passed; collector/report
changes are checked by actual dataset, prefix and physical replay audits.

No BC/PPO updates or model replacement. This dedicated four_static dataset still
requires an explicit task/retention interface before later fitting. Reports are
also copied into the main visible results/diagnostics directory.

Report: results/diagnostics/four_safe_teacher_v1_seed61/README.md.

## 2026-11-04: four-obstacle BC interface and short-budget validation

User authorizes the recommended next BC interface/pilot. Add a separate loader
and trainer, preserving old GROUPS and workflows. Load the byte-verified source
initialization; train only multi_adapter, not original actor, critic or log_std.
Pool64 verified four-static episodes with281 old two-static/54 one-static/54
alignment training episodes. Old labels are INITIAL policy means for retention,
not old RRT actions. Exclude old heldout and all validation/test seeds before
gradient assembly. Use80/10/5/5% task sampling, batch256, LR5e-5,1,000 BC updates,
no PPO transitions. Keep the default demo/source model unchanged.

Pure-policy validation at0/25/50/100/250/500/1,000 uses32 new four layouts,
20 old two layouts and40 one/alignment cases. No RRT execution/action rejection
in these evaluations. Rank safety first with a bounded2-success tradeoff, allowing
prevented collisions to become timeouts; preserve old two success/collision and
exact one/alignment trajectories. Earlier checkpoint wins ties. Keep64 new tests
reserved. Short-PPO readiness also requires learned gain, success>=50% and
collision<=25%; never auto-launch PPO in this runner.

Four validation improves4/28/0 at initialization to17/15/0 at1,000 BC, but old
two regresses15/4/1 to8/6/6. Select BC25:four7/25/0,old two16/4/0. All one/alignment
trajectories stay identical, all frozen states/source bytes unchanged. PPO not
started and readiness false. Four training MSE decreases while old two action
drift grows; shared adapter activates for>=2 obstacles. Recommend an independent
four-only correction branch before longer fitting.14 targeted interface and old
compatibility tests pass; report includes every checkpoint and all episode rows.

Report: results/diagnostics/four_obstacle_bc_v1_seed61/README.md.

## 2026-11-04: prioritize four-obstacle success, relax two-obstacle retention

User explicitly prioritizes four-obstacle success and says two-obstacle skills
need not dominate. Continue from BC1,000 (four17/32), remove earlier-task samples
and old-two selection guards. Four-only BC4,000 updates does not improve the
source. Mine only64 training layouts using BC1,000/BC5,000 failures; reconstruct
and finely validate the same saved expert route, then teach safe-prefix recovery
to its literal tail. Collect107 successful corrections across38 layouts from113
attempts; reject4 connector failures/2 timeouts. Merge with the original64 routes
without changing any array prefix:171 episodes/8,406 transitions. Three prefix
replays have error0. Head-focused correction BC still regresses, so keep source.
Preserve both negative experiments and datasets, with no held-out test use.

Run10,000 pure PPO transitions from sourceBC1,000, using64 frozen four-static
train cases,4 workers,250-step rollouts, batch100,3epochs, LR1e-5,clip0.05,
KL0.005,vf_coef0.1. Train rewards scaled0.01 and critic output reset to matching
units; raw reward terms, physics, installation tolerance and240-step limit stay
unchanged. Update multi_adapter and critic only; freeze base actor/log_std.
No RRT-controller transitions or BC updates in PPO. Select pure deterministic
four-policy validation success, never old-two retention. Freeze selection before
the64-scene paired held-out test; no test-based checkpoint reselection.

Validation17/15/0 ->22/10/0 (success/collision/timeout,32 cases). Paired test
37/27/0 ->51/13/0 (64 cases),57.8% ->79.7% success. All37 original successes stay;
14 collision-to-success pairs. Stochastic training has183 completed episodes:
37 success/132 collision/14 timeout,9,662 completed plus338 unfinished transitions.
Report this separately from deterministic/no-sensing-noise evaluation. The64
test bank is now used and must not be reused as a fresh independent test after
further tuning. Dedicated demo_four_trained.py points to selectedPPO; old default
demo/model remains intact. No GitHub publication requested in this task.

Report: results/diagnostics/four_priority_ppo_v4_seed61/README.md.

Final verification:full178 tests passed. Trained-policy headless demos succeed
on frozen validation1022018 (62 steps) and fresh random422877867 (74 steps),
both static4 with no online planner. Save both JSON results; do not include these
illustrations in benchmark success counts. New explicit demo keeps0.6x playback.

## 2026-11-04: diagnose13, expand four-static data, continue20k and audit noise

User executes the recommended collision diagnosis, larger train/new test,
PPO continuation and noise comparison. Reproduce all13 remaining contacts with
matching reasons/steps:all panel;11 late slot4,2 early slot3. Old64 test becomes
regression/diagnostic only and never enters training. Freeze192 train (retain64,
add128),64 fresh validation,100 fresh test, with disjoint IDs and unchanged
physical difficulty. One obstacle-free nominal-generation timeout is recorded.

Scan192 train layouts using the source:PPO10k,57 failures. Sample75% uniform all
train+25% hard train only. Continue20,000 pure PPO transitions, every5k validation.
Retain critic, Adam and weights exactly at load (optimizer digest verified),
reward scale0.01 and original hyperparameters; no value/optimizer reset. Freeze
base actor/std, train multi_adapter/critic. Fresh64 controls selection; old32 is
tie-break only, old64 cannot select. Freeze hash before testing and never tune on
new100 outcomes. Noise levels declared before training:target sensing5mm/0.5deg
and separately stochastic actions on the same100 layouts, fixed per-scene RNG.

Select+10k (four-stage total20k):new64 validation56/8/0 vs51/13/0. Later+15k49/11/4
and+20k46/11/7 regress, preserve final but do not adopt. Fresh100 source74/25/1,
selected84/16/0; all74 successes retained,9 collisions+1 timeout rescued. Old64
regression51/13/0 ->55/9/0. Sensor-noise outcomes remain84/16/0; sampled actions
57/35/8. Avoid equating different noise conditions or claiming90% achieved.
Full training315 completed episodes:124success/172collision/19timeout,19,760
completed plus240 unfinished transitions, total20k. No BC or RRT-controller
training, no dynamic stage. New100 is now used, so future independent tests need
another bank. Full182 tests pass; latest explicit demo_four_expanded.py succeeds
on frozen1032001 and fresh random1302077125. Existing demos/models/reports remain.

Report: results/diagnostics/four_expanded_ppo_v5_seed61/README.md.

## 2026-11-04: episode50/50 expert/PPO interface; use prior RRT as expert

User asks independent50% BC/expert vs50% PPO episode choice in both training
and evaluation,with RRT preferred if compatible. Existing RRT-Connect supports
the same endpoint/action space and works in physical checks,so label the branch
RRT,not a BC network. Preserve PPOv5 source and all existing pure-policy entries.
Separate case RNG from episode coin/planner RNG. No action averaging,mid-episode
fallback or success-based rerouting. Planning failures remain expert failures.

Add FourCaseBankEnv + FourEpisodeMixture. Expert episodes execute in reset,
PPO episodes alone enter SB3. Accept only successful pre/post-verified expert
trajectories with original reference intact. Use separate actor-only auxiliary
BC Adam at next rollout start,KL0.003,LR1e-6,weight0.05,one update per rollout.
Continue PPO Adam/value state,retain scaled reward units. Default profile10k
PPO transitions plus extra RRT transitions; no formal training launched here.
Validation evaluates pure PPO and a reproducible50/50 mixture,with branch
subscores; pure PPO controls learned-model selection. Old banks are not new
independent tests. CLI flags explicitly start train/evaluate and refuse overwrites.

Actual functional preflight8 episodes:2 RRT success,6 PPO with4 success/2 collision.
Separate training-wrapper probe:RRT1020000 succeeds56 steps,provides56 verified
pairs;PPO1020000 succeeds73 steps. No optimizer/source file update. Unit tests
also exercise real two-rollout PPO/BC updates and validate stored likelihoods,
expert-data separation,route/reference guards and reward scaling. Current84%
remains a pure-PPO score,not a retroactive mixture result. Graphical mixture
command is available through demo_four_expanded.py --controller mixture.

Preflight:results/diagnostics/four_episode_mix_preflight_v6_seed61/README.md.

Verification complete:full188 regression tests passed;6 final targeted checks
passed after adding non-overwriting evaluation output/model CLI options. No
formal mixture model optimization started; source hash remains unchanged.

## 2026-11-04: formal mixed training completed; retain source weights

User explicitly starts formal training. Launch detached PID11003 and preserve
Adam/value state. Finish at16:09:10 after starting15:26:11:10k PPO and10,131 RRT
transitions,20,131 physical total. Start149 RRT/145 PPO episodes;complete149 RRT
successes and141 PPO(57success/73collision/11timeout),with4 unfinished PPO episodes.
Ten separate BC updates accepted,zero KL rejections. All expert labels satisfy
pre/post static verification and reference preservation; PPO data stay separate.

At0/5k/10k pure-PPO validation success56/55/50,collision8/8/11,timeout0/1/3.
Fixed24-RRT/40-PPO episode-mixture validation success60/60/55,collision4/4/7,
timeout0/0/2. RRT24/24 succeeds at every checkpoint. Both learned candidates
regress;select0 added PPO steps. Verify every policy tensor in selected_model
matches the sourcePPOv5 and source file hash is unchanged. Final candidate is
preserved separately. No new independent test or model promotion. The apparent
mixed93.75% advantage exists at initialization and is not a training gain.

Process has exited. Finalize README,curves/checkpoint CSV,paired outcomes and
verification. Correct status.completed to149 RRT/141 PPO from actual final logs,
not the stale9k rollout counter. Main visible report remains a live directory
symlink into the worktree. No further training is launched by status checks.

Report:results/diagnostics/four_episode_mix_v6_seed61/README.md.

## 2026-11-05: regression replay and matched online-BC ablation complete

Replay all six lost successes at0/5k/10k: all18 reasons/step counts match saved
validation. Three panel/slot4 collisions at47–48 steps; three traversal timeouts
retain nonterminal reference indices. Original path arrays never change.
Two stalled timeouts make three failed recovery attempts each: nearest forward
reference distances0.59/0.65rad exceed the unchanged0.45rad recovery radius.
The third timeout continues slowly and does not satisfy the little-motion gate.
Small same-observation mean-action changes accumulate with discrete reference
advancement. Repeated RRT route labels can conflict, but this alone is not causal
evidence against imitation.

User authorizes a5k single-setting ablation. Derive configuration directly from
v6 protocol; set only auxiliary_bc.updates_per_rollout1->0 algorithmically.
Preserve source7194d7515ffe59e3b6d60562b0ebd7e38d6f3cd52ed35a67c7b624c23891e0e1,
Adam/value initialization, seed61,192train/64validation banks, sampling,
hyperparameters, scaled rewards, and50/50 RRT/PPO ownership. Train detached
PID4663 from23:54:16 to00:19:52. Collect5k PPO +5,438 RRT=10,438 physics steps.
Start82 RRT/83 PPO; finish82 RRT successes and79 PPO(30S/44C/5T), four unfinished
PPO episodes. Zero BC updates; empty BC optimizer state. All5,438 expert
observation/action pairs exactly equal the old run's5k expert pool.

Matched5k BC-on versus BC-off purePPO55/8/1->50/11/3; mixture60/4/0->55/7/2.
Five old5k successes are lost: three collisions and two timeouts. Original
failures remain failures. No-BC is not adopted; selected+0 policy tensors equal
the source and default remains unchanged. No independent test used. This
single-seed result does not support auxiliary BC as the sole source of drift;
next investigate smaller PPO changes and reference-advancement sensitivity.

All19 targeted tests pass, including zero-update actor/critic/RNG preservation.
Archive actual runtime config, generate verified comparison/paired CSV/curves,
and correct final status counters from episode journals. Main visible report
directories link into the worktree. No Git commit/push or GUI launch.

Comparison:results/diagnostics/four_mix_ablation_v7_seed61/README.md.
Diagnosis:results/diagnostics/four_mix_regression_v7_seed61/README.md.

## 2026-11-05: user-confirmed slow dynamic four-obstacle inference

User confirms all eight proposed A choices: modify/demo/functional checks only,
no training; retained v5;50/50 RRT/PPO episode ownership; four unequal boxes with
smooth XYZ motion,5cm maximum Euclidean anchor displacement and0.02m/s vector
speed; randomized starts/anchors/phases; time-aware RRT allowed to traverse
moving regions at safe times; simulation time paused during planning; zero noise.

Add a separate motion profile, bounded sine-scene builder, explicitly enabled
SlowDynamicFourEnv and demo entry. Keep the old conditional environment's static
guard. Preserve immutable observation reference, target, reward, horizon and
77/7 interfaces. Motion anchors retain randomized on-path placement; stored
anchor collision witnesses are not misreported as instantaneous moving poses.
Swept boxes filter scene endpoint/wall/spacing geometry only. Runtime planning
uses TimedRRT and verifies planning does not change time/joints/boxes/reference.
No optimizer step or model checkpoint creation is part of this feature.

Functional8-case inference:2 RRT successes; PPO5success/1collision, with no
dynamic-outcome rejection or branch fallback. Preparation rejects two wall-motion
overlaps and one obstacle-free nominal-rollout timeout (no complete anchor path).
All4 boxes move inXYZ; max measured displacement0.0470073m, speed0.0175575m/s.
Source hash, policy tensors and update counters unchanged. Report code hashes
match the archived execution. This is not a dynamic success-rate benchmark.

New random-scene headless demo seed932492594 chooses RRT and installs in77steps
(position0.001243m, orientation0.10969deg). Legacy v5 static demo seed1032000
still succeeds in74steps with the same error0.001749927m as before.
Targeted24 tests pass; full194 regression tests pass in121.97seconds. After that,
clarify preparation-rejection metadata to distinguish geometry from unavailable
nominal references; no physics, source policy, or executed outcome is changed.

Docs:docs/FOUR_SLOW_DYNAMIC.md.
Report:results/diagnostics/four_slow_dynamic_preflight_v1_seed61/README.md.

## 2026-11-05: conservative dynamic mixed-training pilot completed

User executes the proposed dynamic-interface/time-verified teacher, gradual
speed curriculum and5k PPO experiment. Freeze192 train(162 reused+30 new),64
new validation and100 reserved test cases; log geometry and nominal-reference
construction failures separately. Source and old held-out/preflight banks stay
outside training. Use30% static scene retention, independent50/50 RRT/PPO
ownership, random training phases,5cm displacement bounds and zero sensor noise.

Preserve v5 actor/base/Adam/critic initialization; LR3e-6, other PPO/reward
settings unchanged. Gate0.005->0.01->0.02m/s at1k boundaries. Candidate selection
uses fixed0.02m/s pure PPO, with paired-frozen and old-four-static retention
guards. Mixture uses common coin assignments and physically executed RRT rows
cached once; no evaluation samples enter imitation. Reference-recovery rules
are monitored and remain unchanged.

Dynamic teacher labels need timed prechecks, actual installation, post-time
checks and fresh action replay matching observations/joints/time to1e-6.
Static teachers retain static verification and the replay.28 targeted tests
pass, including real teacher/replay and genuine on-policy buffer audits.
Functional wrapper:dynamic1020002 RRT69steps verified,69 pairs;following static
PPO collision retained. No updates in that probe. Formal PID5330 runs from
13:21:29 to13:38:22. Five BC updates accepted, zero rejected.

At0/1k/2k/3k/4k/5k target64 S/C/T are41/20/3,42/19/3,43/18/3,43/18/3,
42/19/3,39/21/4. Advance at1k and2k; select+2k. Its paired-frozen44/16/4 versus
source44/17/3, old-static57/7/0 versus56/8/0. Mixture remains51/11/2 versus
source51/11/2:two rescued target scenes belong to the common RRT branch,while
one rescued PPO-assigned scene and one lost PPO-assigned scene cancel out.
The5k candidate is preserved separately and not selected.

After locking+2k, run the100 reserved dynamic source/selected test once:
77/22/1 ->79/20/1. Pairing:75S->S,18C->C,1T->T,4C->S,2S->C. This is a small
single-seed change, not a broad stability claim. No tuning follows the test;
the bank is now used. Default v5 remains unchanged, candidate kept separately.

Training66 RRT successes;67 PPO completed(23S/36C/8T), four unfinished.65/66
teachers pass the label gate; seed1030121 succeeds physically/replays exactly
but fails the conservative post-time check, so its126 labels are excluded.
Save4,832 expert pairs. Count5k PPO+4,958 RRT+4,958 replay=14,916 physical
transitions excluding validation/test. Completed PPO4815 steps plus185 in
unfinished episodes. Audit all training seeds, source/data/archive hashes,
learning rate and accepted label guards. Generate report, curves and paired
dynamic/frozen CSVs under the main visible report symlink. No Git push.

Report:results/diagnostics/four_dynamic_ppo_v8_seed61/README.md.

## 2026-11-05: paired speed continuation with90% dynamic episodes

User executes the proposed fixed-speed comparison and asks to increase dynamic
and reduce static scenes. Explicitly confirms90% dynamic/10% static. Apply the
same Bernoulli scene probability to both arms; controller ownership stays an
independent50/50 RRT/PPO coin. Both add2k PPO steps from selected v8+2k, with
seed71 and common validation branch seed61. Fixed train speed0.01 vs0.02m/s;
target evaluation stays0.02m/s. No70/30 control is present, so do not attribute
results specifically to the scene-proportion change.

Restore source global PPO counter62000/update66, Adam and critic, plus matched
BC aux_2000 snapshot:Adam, RNG,1467 pairs, two inherited accepted updates. PPO
source SHA3db6ed2ff1211381ee5ad2bbe983a65f4313ef793daa25a0a8043c71f67da14d;
BC snapshot SHA605099ca554e14f27c82913f246fab9de04c06c5e87d81ad4f32f3cf5906e999.
Initial target43/18/3, frozen44/16/4, old-static57/7/0 reproduce exactly in both
arms. Start fresh physical episodes with paired new seed streams; do not claim
bitwise continuation of old unfinished episodes. Preserve inherited pool
composition. Scope90/10 to NEW sampled scene episodes.

Six initial restore/config/mode checks pass within a26-test targeted run; a
seventh restored-BC/two-rollout test initially lacks a fixture queue method,
then passes after the fixture exposes drain_demonstrations. All seven restore
tests pass, including original weights/PPO optimizer preservation, mismatched
counter/shape rejection and real-observation on-policy log probabilities.
No production algorithm correction was required for that fixture failure.

Launch detached coordinator PID11011, with separate sequential child processes
for identical initialization. Both finish all2k new transitions. Slow01 target
at0/1k/2k:43/18/3,45/17/2,47/15/2. Fast02:43/18/3,46/16/2,47/15/2.
Both select their final+2k (v5+4k along this chain). Old-static remains57/7/0.
Paired frozen slow45/15/4 vsfast46/15/3. Common-mixture source51/11/2 becomes
53/10/1(slow) or54/9/1(fast). Primary pure-PPO selection scores tie.

Selected targets agree on46 successes,14 collisions,2timeouts and disagree on
two cases:1062016 slow collision/fast success(PPO-owned),1062054 slow success/
fast collision(RRT-owned). This explains the one-case mixture difference.
Do not infer a generally superior speed from this single-seed result.

Slow training:40RRT successes/1timeout,28PPO(11S/15C/2T),66dynamic/3static
completed scenes. Fast:38RRT successes/1timeout,27PPO(10S/15C/2T),63dynamic/
3static. Finite counts differ from expected90/10. Each adds two accepted BC
updates to two inherited updates; no independent test is rerun. Source/test
isolation, source P/BC optimizer/pool/RNG equivalence and archived-code/input
hashes pass. Shared4000 new PPO +6191 RRT +5711 verification-replay transitions
=15902 training/replay physics, excluding validation. Original v5/v8 models and
default demos remain unchanged. Final paired case CSV and mixture curves are
added after the training, preserving the original coordinator archive.

Report:results/diagnostics/four_dynamic_speed_comparison_v9_seed71/README.md.

## 2026-11-05: increase obstacle speed and range for animation

User requests larger/faster obstacle movement after viewing the trained demo.
Choose a separate enhanced profile:maximum Euclidean anchor offset5->10cm,
maximum vector speed0.02->0.05m/s. Use selected v9 fast02 source, retain four
unequal on-path anchors, random starts/layouts/phases, zero noise,0.6x playback
and known-motion timed RRT. Existing slow/training configurations and checkpoints
are not overwritten; no training is requested or launched. Increase placement
attempts20->50 while keeping the endpoint/wall/sparse-spacing geometry guards.

Eight-case physical preflight:2 RRT successes,5 PPO successes/1collision.
All4 boxes move inXYZ; maximum observed offset0.099284334m, speed0.041814383m/s.
Reject8 geometry candidates and1 incomplete obstacle-free nominal reference
before dynamic execution; preserve every dynamic outcome. Source tensors,
update counters and source file hash stay unchanged. This is functionality
evidence, not an independent success-rate estimate for the harder motion.

The existing report writer had fixed v5/5cm/0.02mps prose. After execution,
generalize future prose to the actual profile and correct this report's prose
from its executed protocol, retaining original runtime snapshots and numeric
outcomes. New entry: scripts/demo_four_dynamic_faster.py.
Config: configs/four_dynamic_faster.yaml.
Report:results/diagnostics/four_dynamic_faster_preflight_v1_seed61/README.md.

## 2026-11-05: further increase demo motion to15cm /0.10mps

User requests another increase. Update the existing enhanced-demo default from
10->15cm Euclidean anchor-offset bound and0.05->0.10m/s vector-speed bound.
Retain10cm settings in configs/four_dynamic_faster_10cm.yaml. Increase geometry
sampling cap50->100 without changing wall/endpoint/minimum-spacing guards.
Keep four unequal on-path anchors, fresh random starts/phases, zero noise,
known-motion timed RRT,0.6x playback and selected v9 fast02 source. No training.

Eight accepted cases require132 geometry exclusions and4 unavailable complete
obstacle-free references before execution. Generation is substantially slower
under the larger swept bounds; optional saved-case replay is documented and
fresh generation stays the default. No actual dynamic outcome is filtered.
Physical results: RRT2 successes; PPO2 successes/3collisions/1timeout. All boxes
move inXYZ; max observed displacement0.1486995588m, speed0.0897471841m/s.
Planning clock/reference restoration and unchanged source tensors/counters/hash
verify. These eight episodes are functional evidence, not a performance study;
the existing model has not adapted to the stronger dynamics.

The same scripts/demo_four_dynamic_faster.py command now reads the new defaults.
Report:results/diagnostics/four_dynamic_faster_preflight_v2_seed61/README.md.

## 2026-11-05: large-motion pilot training authorized

`configs/four_large_motion_training.yaml` continues the selected v9 fast02
checkpoint and its matching final BC optimizer/pool/RNG state. It uses90% dynamic
and10% static episodes, with an independent50/50 RRT/PPO choice per episode.
The initial pilot budget is5,000 additional PPO transitions. RRT execution and
verification replay are counted separately.

The proposed curriculum is10cm/0.05m/s,12.5cm/0.075m/s,15cm/0.10m/s. Progression
requires validation non-regression and applies only to new episodes. Every
checkpoint is compared at the fixed final15cm/0.10m/s target. Full motion norm
bounds protect starts/goals/wall/spacing during placement; actual time-dependent
collision checking remains in the planner and execution. On-path anchors may be
resampled within the original route intervals to fit these larger bounds.

The bank contains192 training,64 validation and100 reserved test layouts.
No dynamic-policy outcome is used to construct the banks. Held-out cases never
enter PPO or auxiliary BC. The reserved test is evaluated only once after a new
checkpoint passes validation selection. Previous source/default models remain
available. Report/status/checkpoint counts and curves:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/four_large_motion_ppo_v10_seed81/`.
Preparation and interface evidence:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/four_large_motion_preparation_v10_seed81/`.

Preflight:17 targeted tests pass. An actual new-domain RRT teacher succeeds
with89 steps and independent replay error1.56e-15; a separate deterministic PPO
training-bank episode collides at42 steps. Neither probe updates optimizers.

### Large-motion pilot completed (2026-11-05)

All5,000 PPO transitions completed, reaching the15cm/0.10m/s curriculum level.
In the fixed64-case final-strength pure-PPO validation, the baseline is54
successes/10collisions/0timeouts; the selected+2,000-step checkpoint is55/9/0.
The50/50 mixture remains56successes/6collisions/1timeout/1planning failure.
Original static validation remains57/7/0; paired frozen layouts improve52/11/1
to53/10/1.17 targeted tests and runtime data/code/replay audits pass.

The fresh100-case paired test does **not** show improved success: source76/19/5
versus candidate76/20/4. One timeout becomes a success and one prior success
becomes a collision. Keep the previous source as the default; retain the selected
checkpoint only as an experimental result. No training or tuning follows this test.
See the report directory above for README.md,curves.png,checkpoints.csv,
paired_validation.csv,independent_test.json and verification.json. The matching
BC state is saved in models/four_large_motion_ppo_v10_seed81/selected_auxiliary_state.pt.

## 2026-11-05: staggered dynamic-only pilot completed

All new training, teachers and evaluation episodes are dynamic. Four unequal
obstacles retain full-body on-path witnesses and full-motion endpoint/wall/spacing
guards. Motion caps increase15->18cm and0.10->0.12m/s; vertical offsets alternate
3-7cm, with >=15cm anchor height span and >=4cm adjacent height differences.
All356 bank layouts pass parameter checks. Prior validation/test IDs, including
used v10 cases, remain excluded from training.22 distinct targeted tests pass.

TRAIN diagnosis:110success/79collision/3timeout. Collisions:25early/52middle/
2terminal. Contact evidence resolves72panel-obstacle,5arm-obstacle,2panel-wall
episodes;22 terminal-query gaps are resolved by physical-substep replay.
No optimizer updates occur in diagnosis.64 RRT attempts produce54physical
successes, of which52 pass all timed/replay guards, yielding3948 dynamic labels.
The old mixed BC pool is not reused.

BC attempts200 updates;20 accepted and180 rejected by source-KL guard.
At25/50/100/200 updates validation is32/31/1 success/collision/timeout versus
source37/26/1, so select0 and preserve the original PPO optimizer/critic/base/std.
PPO then completes5000 transitions plus5 low-strength auxiliary BC updates.
At0/1k/2k/3k/4k/5k, final-target pure PPO is37/26/1,36/27/1,35/28/1,
36/27/1,37/26/1,36/27/1. No new checkpoint beats the selection rule.
The curriculum remains15cm/0.10m/s because dynamic gates do not pass.
Offline teachers, all fixed-target validations and the enhanced demo use18cm/
0.12m/s. Amplitude/speed ordering alone did not guarantee an easier curriculum.

Mixed validation is42/19/1/2 at source; at1k43/18/1/2 (last count is planning
failure), but that checkpoint fails the pure-PPO collision guard. It is an
exploratory mixed-system result, not an adopted or independent-test improvement.
Retain source weights;100 reserved test layouts remain unused. No static
retention is reintroduced. New report:results/diagnostics/
four_staggered_dynamic_ppo_v11_seed91. Offline BC/diagnosis report:
results/diagnostics/four_staggered_dynamic_bc_v11_seed91. Enhanced demo uses
configs/four_dynamic_faster.yaml, with15cm settings backed up separately.

## 2026-11-05: corrective BC and predictive execution follow-up

Pure learned-policy trial v12 collects32 verified corrective suffixes from32
TRAIN layouts;21 start at nonzero simulation time after exact original PPO
prefixes. All32 pass timed pre/post checks and full prefix-plus-suffix physical
replay. The507 first16-step labels and2675 successful source-policy states
train two anchored BC arms (LR1e-6,3e-6;200 attempted updates each). None beats
source37success/26collision/1timeout on the fixed64-case dynamic validation.
Select0; conditional2k PPO is skipped. The original PPO remains unchanged.
Five actual-physics corrective-prefix tests pass. A read-only clamp-gradient
diagnosis finds no saturated action components in this sample; do not attribute
the failed BC to clipping on this evidence.

Separate v13 is explicitly PPO + short-horizon physical prediction, not newly
learned pure PPO. It simulates speed gains1,.75,.4,0,-.25,-.75 and restores both
Bullet and Python state before choosing an actual motor command. No global
RRT fallback occurs in this branch. Original RRT episodes remain a separate
50% branch. Three full-episode/exception/clock restoration tests pass.
Source weights, optimizer and actual terminal criteria are unchanged.

Only24 TRAIN cases select the horizon: raw12/24 successes; H4=15/24,
H8=17/24,H12=20/24. LockH12 (0.6s) before validation. On64 validation cases,
raw37/26/1 -> predictive56/0/8. Paired50/50 mixtures:42/19/1/2 ->57/0/5/2
(success/collision/timeout/planning-failure); the same RRT rows are reused.
Validation passes the declared strict-success-gain/collision-nonincrease gate.

One fresh100-case paired test is then evaluated without subsequent tuning:
raw57/41/2 -> predictive78/0/22. All57 original successes retain outcome,
length and errors with zero intervention;21 collisions become successes and
20 become timeouts. These gains are an execution-layer result with known
dynamics and discarded prediction time, not pure-PPO learning gains. No claim
of general safety follows from finite-horizon checking. The v11 reserved bank
is now consumed for this locked experiment and must not be reused as a fresh
test for further tuning.

Reports:results/diagnostics/four_dynamic_correction_study_v12_seed101 and
results/diagnostics/four_predictive_safety_v13_seed111. Preserve default pure
PPO and old demos. New explicit demo: scripts/demo_predictive_dynamic.py;
its default mixes RRT and PPO+prediction50/50. --controller ppo_predictive
shows the predictive branch alone. Current scene stays fully dynamic,18cm/
0.12m/s and alternating3-7cm anchor offsets. Selected controller parameters
are exported separately in configs/four_predictive_controller.json.

## 2026-11-05: reduce waiting timeouts with local progress recovery

Frozen v14 adds a model-based local recovery to the v13 predictive safety
controller. All scenes remain dynamic with four unequal staggered boxes,
18cm offset and0.12m/s speed. No neural weights, collision/success criteria,
reference route or240-step real episode budget change. No global RRT is used
in the predictive-PPO branch; the separate50% RRT branch remains available.

History diagnosis of22 previous-test timeouts:16 long-wait cases,3 nearly
unmodified-policy stalls and3 mixed slow-progress cases. These are development
diagnostics only. A new100-case geometry-only bank from1144000 is frozen
before evaluation, disjoint from all prior training/validation/test IDs.

At step>=100, low net progress or insufficient estimated remaining budget
triggers a baseline continuation simulation. Existing successes are protected.
Otherwise locally simulated goal-directed joint commands or eight-step
single-joint biases can make progress, with a36-step prediction and normal
safety checks on each actual action. Reuse failed-continuation predictions
only while actual observations reproduce the predicted baseline states.
The actual reference is never overwritten. Three targeted tests pass, including
a forced successful-continuation protection branch and a real timeout rescue.

Train sanity subset:20success/1collision/3timeout ->23/1/0. Validation64:
56/0/8 ->62/0/2; same RRT coin assignments yield mixed57/0/5/2 ->61/0/1/2,
where the final count is planning failure. All old successes retain length
and final errors without local recovery actions.

After validation locks the controller, the new100-case paired test gives
old76/3/21 versus new96/1/3 success/collision/timeout. All76 successes stay
unchanged;18 timeouts and2 collisions become successes;3 timeouts and1collision
remain. No tuning follows this test. Do not compare the older78/0/22 result
from a different bank as this test's baseline. The new bank is now consumed.

Actual zero-gain waiting commands decrease2876->1599. Mean concurrent-evaluation
wall time rises9.45->14.15s per case. Prediction pauses simulation time, so this
improves simulated completion rather than proving lower real-world latency.
Known dynamics and finite-horizon/model assumptions remain.

Report:results/diagnostics/four_progress_recovery_v14_seed121/README.md.
New explicit demo: scripts/demo_progress_dynamic.py, default50/50 RRT and
predictive-PPO-progress; --controller ppo_progress isolates the improved branch.
The old predictive-only and pure-PPO entries remain available.

After v14 independent success, promote the current predictive demo default
to mixture_progress using the tested parameter JSON. Keep its exact former
entry as scripts/demo_predictive_dynamic_v13.py. This changes only the entry
default, not the locked controller, source weights or any measured result.


## 2026-11-06: larger vertical stagger and measured height-change demo

Four dynamic anchors now default to alternating 8–14 cm offsets, at least 24 cm overall height span and 10 cm adjacent difference in configs/four_vertical_weave.yaml. Motion stays 18 cm / 0.12 m/s. Eight geometry-selected development scenes succeeded; a separate disclosed demonstration search along four measured routes produced three feasible successes and one geometry rejection. Fixed seed 1144002 reproduces multi-stage height changes and success in 141 steps with the unchanged PPO + predictive safety + recovery controller. This is a selected demonstration, not a new independent success-rate estimate or a guarantee of alternating above/below every box. Full report and terminal commands: results/diagnostics/four_vertical_weave_v15/README.md. Old demo entries retained as *_v14.py. No model training.


## 2026-11-06: lower demo obstacle O2 by 5 cm

The current vertical-weave demo profile applies explicit offsets [0, -0.05, 0, 0] m to GUI labels O1/O2/O3/O4, translating the full O2 motion without changing its phase, speed or amplitude. Frozen banks and historic results stay intact. Full-motion geometry rechecked; existing seeds 1144002/1144004/1144008 succeeded in 134/118/201 steps. No training or model-weight change. Applied demo offsets are idempotent when reloading an adjusted bank. See results/diagnostics/four_vertical_weave_o2_lower_v16/README.md.


## 2026-11-06: record first, then fixed-cadence playback

All common demo entries default to headless full-episode recording followed by60 FPS /0.6x display. No planning or physical stepping occurs during playback. Save configuration, exact scene and real control-state joints, payload position/quaternion, box centers and original result. Display interpolation is not a new simulation/collision result. Real waits remain visible; late frames are skipped rather than replayed in a burst. Use scripts/replay_episode.py for repeats without loading the model, --playback-mode live for the old display and --headless for recording only. Seed1144004 remains118-step success with all119 recorded states identical to v16;7 relevant tests pass. Xvfb graphical replay presents591/591 frames in9.842s (9.833s target), zero skipped frames. Documentation: docs/RECORDED_PLAYBACK.md; report: results/diagnostics/recorded_playback_v17/README.md. No training.


## 2026-11-06: frozen current-layout paired validation (v18)

Freeze sourcev9 fast02 weights, current four-dynamic profile (18cm/0.12m/s, base8–14cm heights, O2 lowered5cm), controller and runtime code.64 fresh default-generator scenes are accepted by nominal-route completeness and geometry only;121 candidates rejected. No selected success-demo bank, training or final test. Paired PPO:40S/23C/1T; RRT:55S/1C/0T/8PF; enhanced PPO:63S/0C/1T; frozen50/50 mixture:58S/1C/1T/4PF (30RRT/34enhanced episodes, complete-rollout reuse).192 actual controller episodes, not256 independent runs. All40 original PPO successes preserved. Remaining enhanced timeout seed1166054 ends at0.352m/35.47deg with cursor5/47 after3 identical joint-index3 bias recoveries (108actions);38zero waits are not the dominant count. RRT seed1166154 collides after17 failed replans/holds; contact object unavailable. Prioritize reference recovery handover and RRT failed-plan safety before further broad training. Report: results/diagnostics/four_vertical_validation_v18_seed161/README.md; complete per-case CSVs, frozen model, bank and code included.


## 2026-11-06: larger measured-route stagger and verified vertical passages (v19)

New demo-only profile four_vertical_corridor.yaml uses14–22cm offsets along a preliminary physical execution and preserves O2 down5cm. Enforce final total-height span>=35cm and adjacent difference>=14cm after O2 translation; same18cm/0.12m/s dynamic bounds, immutable observation reference, source weights and240-step task budget. Preliminary outcome is not a success filter. Keep evaluated v18 profile and entry separately.12 reused validation candidates yield7 geometric rejections and5 final episodes (2S/3T/0C); selected seed1166026 has verified whole-panel below-O1/above-O2/below-O4 passages,239-step success,1.17mm/0.094deg error. Raw generation and full execution reproduce identical geometry and states.3 extra phase variants give1S/2T but no qualifying rise/fall example, so retain original phase. These are disclosed demonstration searches, not new success-rate evidence. Record full replay and a clearly labeled contiguous motion excerpt (real steps155–239), no action/result edits. Three semantic passage tests and both replay checks pass. docs/VERTICAL_CORRIDOR_DEMO.md and results/diagnostics/four_vertical_corridor_v19/README.md contain commands, limitations and all attempts. No training.


## 2026-11-06: measured current-corridor success rate (v20)

Freeze current14–22cm height profile, O2 down5cm, post-adjustment35cm total/14cm adjacent height guards, source model and controller. Generate285 new-seed candidates:220 rejected,64 first valid scenes frozen,1 extra valid candidate unused. Preliminary physical outcome is not a success filter. On the same64 final layouts: enhanced PPO43S/1C/20T (67.1875%); pure PPO0S/64C/0T. Enhanced success with measured3+ alternating whole-panel passages and actual rise/fall:9/64 (14.0625%). All20 enhanced timeouts meet predeclared waiting/stagnation criterion; mean final error0.379m/50.23deg. No model/controller tuning, training or final-test episodes.128 actual final-scene rollouts plus construction precomputations. All64 initial physical states are identical between modes; full recordings and frozen cases/models/code saved. Four semantic metric/passage tests pass, plots visually reviewed. Report: results/diagnostics/four_corridor_validation_v20_seed181/README.md.
