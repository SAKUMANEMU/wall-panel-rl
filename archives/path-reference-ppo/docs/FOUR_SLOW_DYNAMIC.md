# Four slow dynamic obstacles (v5 inference, no training)

The separate `demo_four_slow_dynamic.py` entry uses the selected PPOv5 source.
Each episode draws an independent 50% RRT / 50% PPO coin. Its owner executes the
whole episode, including failures; there is no within-episode switching. Scene
sampling, controller selection and motion phases use separate random streams.

Four different-size boxes keep the existing randomized anchors along the
nominal obstacle-free physical arm/panel trajectory. The anchor placement keeps
the original fraction/position jitter. The robot initial pose, anchors and
motion phases are freshly randomized by default; explicit seeds reproduce them.
The installation target, reward, 240-step horizon, 77 observation features and
seven joint-increment actions are unchanged. The fifth obstacle slot is empty.

Motion is smooth three-axis sine motion with independently randomized initial
phases and frequency fractions per box/axis. These are known prescribed
trajectories once an episode starts, not sudden random direction changes.
For extent vector `e`, frequency vector `w`, phase `p`, and simulation time `t`:

```text
center(t)   = anchor + e * sin(w*t + p)
velocity(t) = e * w * cos(w*t + p)
norm(e) = 0.05 m
norm(e * w) <= 0.02 m/s
```

Five centimeters is the maximum Euclidean offset from the anchor, not the
limit independently on every axis. Both positive and negative motion occurs.
Zero target-observation noise is used. Obstacle velocity remains visible in
the existing observation slots. PPO uses its unchanged v5 weights; it has not
been adapted to these dynamic scenes.

The RRT branch uses the existing `TimedRRT`: snapshot geometric candidates plus
arrival-time collision checks and safe waits/replanning. It may cross a moving
obstacle's swept region when that region is clear at the arrival time. It does
not replace each moving box with a full swept-volume obstacle during planning.
Only scene construction uses swept boxes to guarantee sparse inter-box spacing
and protect the start/goal/wall against the motion bounds. This rejects geometry
before choosing a controller. Building the on-path anchors also requires a
completed obstacle-free nominal rollout; failures at that preparation stage
are recorded separately. Dynamic episode outcomes are never filtered.

Planning uses the simulation clock and does not advance it. Physical joints,
box positions and the immutable original reference are checked before/after
planning queries. The existing local reference-recovery rule is explicitly
enabled in the new dynamic environment. The legacy static environment retains
its dynamic-task guard. Time-aware collision sampling is discrete, not certified
continuous collision detection.

## Terminal demo

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py
```

Default: fresh random scene, 50/50 episode owner, no noise, 0.6x animation.
Planning/setup stays hidden before playback. The terminal prints the owner,
scene seed, motion bounds and final outcome. The simulation stops when the
episode terminates; the final GUI image stays open for30 seconds.

```bash
# Explicit branch comparisons; same scene/motion seed, original v5 model.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py --seed 1040000 --controller ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py --seed 1040000 --controller rrt

# No display; normal random scene and 50/50 routing still apply.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py --headless

# Replay a geometry-verified case from the functional bank.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py --scene-bank results/diagnostics/four_slow_dynamic_preflight_v1_seed61/cases.json --seed 1040000
```

`--seed` controls scene/motion; `--policy-seed` controls the mixture coin.
`--controller ppo` / `rrt` explicitly bypass that coin for a branch comparison.
`--hold-seconds` changes the final image hold, and `--playback-speed` changes
GUI timing only. A frozen bank deliberately reuses its saved scene and phases.

Edit `configs/four_slow_dynamic.yaml`, especially
`motion.amplitude_m` and `motion.max_speed_m_s`, to change the motion bounds.
The `frequency_fraction` range controls the random axis frequencies; their
upper bound must stay at or below1 to preserve the configured speed cap.

## Functional verification

`verify_four_slow_dynamic.py` defaults to configuration checking only. The
explicit `--run` option executes eight episodes, never a training budget, and
refuses to overwrite an existing output directory. Change
`preflight.report_root` to rerun into a new directory.

The initial eight-case check completed with two RRT successes and six PPO
episodes (five successes, one collision). Before dynamic execution, two
candidates were rejected because their motion bounds intersected the wall;
one was rejected because its obstacle-free nominal rollout timed out and could
not define complete anchor positions. All four boxes moved on all three axes in every
executed episode. Maximum measured anchor offset was0.04701m and maximum
observed speed0.01756m/s, within the analytic0.05m/0.02m/s bounds. Planning
preserved the simulation clock/physical scene/reference. Source file hash,
all model tensors and training update counters were unchanged.

These eight episodes are functional evidence, not an independent success-rate
estimate. The earlier v5 static test result84/100 does not describe dynamic
performance. No PPO/BC training or new model selection is performed here.

Results are directly accessible at:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/four_slow_dynamic_preflight_v1_seed61/README.md
```

The report includes exact scenarios/phases, every outcome, motion trajectories,
and executed-code hashes. Static demonstrations and earlier reports remain
available through their existing entries.
