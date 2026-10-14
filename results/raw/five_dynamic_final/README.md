# Five prescribed moving obstacles

Final configuration: config.yaml. Ten seeds 10000–10009 physically executed with RRT,
10 successes, zero collisions, zero initial planning failures. No online replans were
needed in these ten executions because RRT used the full obstacle motion envelopes.
The separate automated obsolete-route regression exercises stopping and replanning
and verifies successful physical execution. The complete test suite passed 21 tests.
GUI seed 10002 succeeded without collision in 62 steps at requested 0.6x playback.

Five unevenly sized obstacles are evenly spaced longitudinally along the reference
start-to-target corridor and alternate sides with lateral clearance. Random phases,
small centre jitter, 0.10 m amplitude and 0.05 m/s maximum sinusoidal speed.
Known-motion envelopes plus discrete prediction checks do not provide a continuous
safety proof; obstacles advance on physics time, paused during planning computation.
Only these ten evaluation seeds were tested for full execution. No dynamic PPO
training or noisy-condition success benchmark is claimed. See trajectory.json for
actual robot and moving-obstacle positions, and episodes.json for all outcomes.
