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
