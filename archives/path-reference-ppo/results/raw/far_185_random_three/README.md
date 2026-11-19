# Approximately 1.85 m starts, unchanged target

Seeds 10000–10099 were physically executed with RRT in each of two stages.
Both stages succeeded in 100/100 runs with zero collisions, IK failures, or planning failures.
Mean initial panel-to-target distance: 1.841634 m (range 1.833300–1.849452 m).
The second stage contains exactly three randomly positioned obstacles per scene.
All 100 direct joint-space edges were blocked and all 100 RRT executions succeeded.
The no-obstacle stage also has blocked direct edges: this metric includes robot/self/environment collision constraints, so it does not isolate obstruction caused by the three boxes.
These results describe the tested seeds, not a guarantee for all random scenes.

The original target position and orientation are unchanged; only starts and display settings changed.
GUI seed 10002 succeeded without collision in 60 steps. At 0.05 simulated seconds per step,
3 seconds of simulated motion took 5.0241 seconds of GUI execution (approximately 0.597x).
The configured playback rate is 0.6x; computation and rendering can introduce small timing overhead.
18 tests passed. Logs are included alongside resolved configurations, scenes, trajectories and episode metrics.
PPO checkpoints were not retrained for these farther starts.
