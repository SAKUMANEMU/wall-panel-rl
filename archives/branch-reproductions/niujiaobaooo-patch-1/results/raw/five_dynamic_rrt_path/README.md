# Five moving boxes along an obstacle-free RRT reference

Final configuration is recorded in config.yaml. The environment first generates
an obstacle-free RRT-Connect joint path, samples panel-centre FK along it, then
uses five equally spaced arc-length stations for obstacle anchors. Small random
jitter and bounded endpoint-overlap corrections are applied. The scene records
reference joint/path coordinates and each obstacle anchor/axis/phase.

Motion amplitude is 0.20 m, total travel 0.40 m, peak speed 0.10 m/s.
Ten physical RRT executions (seeds 10000–10009): 10 successes, zero collisions,
zero planning failures. No online replans were needed in these executions because
the planner avoided full motion envelopes. The test suite separately verifies
obsolete-path stop/replanning. All 21 tests passed. GUI seed 10002 succeeded in
67 steps with no collision. These are finite-seed checks, not a safety guarantee.
Planning is synchronous with physics paused; collision sampling is discrete.
