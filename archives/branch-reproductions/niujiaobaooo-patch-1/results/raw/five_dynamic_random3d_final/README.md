# Three-dimensional randomised obstacle motion

Five obstacles move independently along all world X/Y/Z axes. Each component has
random initial phase and frequency, reproducible for a fixed seed. Smooth motion
is prescribed by three sinusoidal components. The configured 0.40 m amplitude
bounds total Euclidean displacement; 0.40 m/s bounds total speed. Each component
has amplitude 0.40/sqrt(3) m and frequency multiplier sampled from [0.5,1].
Conservative 3D swept boxes are used for IK, RRT and endpoint placement checks.

Ten seeds 10000–10009: 10 successes, zero collisions, zero planning failures.
None needed online replanning; the separate obsolete-route regression tests it.
All 21 tests passed, including nonzero movement on all three axes and speed/range
bounds. GUI seed 10000 succeeded without collision. These finite-seed results are
not a continuous safety guarantee. Planning pauses simulation time. Old PPO models
remain incompatible with the 55-dimensional observation; no retraining is claimed.
Resolved config, trajectories, all outcomes and logs are included.
