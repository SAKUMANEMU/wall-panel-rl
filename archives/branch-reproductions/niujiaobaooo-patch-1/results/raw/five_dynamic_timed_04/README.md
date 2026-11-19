# Time-aware passage through moving-obstacle regions

Configuration: five random 3D obstacles, amplitude 0.40 m, peak total speed 0.40 m/s,
rrt.dynamic_mode=timed, 30 s total search budget per planning call. Ten seeds
10000–10009: 7 successes, 3 initial planning failures, zero physical collisions.
All seven successful episodes entered at least one full swept AABB while the actual
moving obstacles were clear. This audit measures the conservative region that the
old planner forbade, not exact intersection with every point on an obstacle curve.
164 scheduled wait control steps were executed across the run. Seed 10002 waited
40 control steps (2 s simulated time) and then completed successfully. No online
replans were needed in these ten runs; an obsolete-route regression tests that branch.

Twenty-two tests passed, including an opening that requires waiting and an unsafe
waiting location that must be rejected. Candidate routes use RRT-Connect at several
obstacle snapshots; an advance/wait time search schedules each route. It is not a
complete space-time RRT and finite-budget failure does not prove infeasibility.
Physical feedback validates each next move and reschedules a stale/unsafe route.
No full swept obstacle exclusion is used by timed route planning or IK; swept
boxes remain endpoint placement checks and audit geometry. Planning pauses physics,
and sampled kinematic checks do not certify continuous-time or real-world safety.

Compare ../five_dynamic_swept_comparison: 10/10 success, zero collisions and zero
swept-region entries under the same motion settings/seeds with conservative routing.
Episode records retain all failures; the representative trajectory includes reference
path, obstacle positions, joint positions and swept-envelope-entry step indices.

GUI validation: seed 10002 completed in 103 steps with 40 scheduled wait steps,
zero collision, at requested 0.6x playback.
