# Construction-site installation environment v29

This isolated version extends v28 with physical workspace boundaries and a
construction-site appearance. Earlier experiments and checkpoints are preserved.
The initial checks reuse the selected v28 PPO checkpoint with predictive
control and local recovery. The adaptation run below writes separate models.

A new adaptation pipeline was launched on 2026-11-17 under
`results/diagnostics/construction_site_v29_training_seed20261117/` in the project
root. It first generates 48 training, 16 validation and 16 test scenes with fresh
seed blocks, then runs baseline physical simulations, collects up to three
teacher episodes per obstacle count, and performs 1,000 BC updates (if at least
64 successful teacher transitions are available) followed by 65,536 PPO steps.
The run's `training_config.yaml` is the frozen training specification;
`pipeline_status.json`, `scenes/status.json` and `training/status.json` report
actual progress. Models and all evaluation recordings are written to that run,
leaving `models/current_ppo.zip` unchanged. The validation results below are
pre-training probes; consult the new run for adaptation outcomes.

## Physical changes

- The floor surface is at **z = 0.10 m** and the ceiling underside at
  **z = 1.10 m**, leaving **1.00 m** of vertical clearance. These coordinates
  describe a restricted, small-panel installation mock-up, not a full-size room.
- Every moving robot link and the held panel participates in floor/ceiling
  collision checks during simulation, configuration queries and prediction.
  Only the fixed robot mounting base may overlap the floor. Contact terminates
  an episode under the existing collision rule and penalty.
- The ceiling is shown as a cutaway for visibility. Its collision slab covers
  the whole 4 x 4 m footprint. The installation wall is extended across the bay
  without moving the target installation plane.
- Concrete joints, robot mounting bolts, hazard markings, wall framing, stored
  boards and strapped material bundles give the scene a construction context.
  Hoist cables are illustrative visuals; they have no collision or pendulum
  dynamics. Obstacle trajectories remain prescribed kinematic motion.

New scenes contain **2--5 moving obstacles**, sampled equally by count. Box
dimensions are **10% larger than v28**, with different sizes within each scene.
Each obstacle samples a speed uniformly from **0.04--0.22 m/s** per episode;
that speed stays fixed within the episode. The geometric routes and their
current progress are supplied through the existing **167-value observation**;
the action still has **7 joint components**. Motion amplitude remains 0.18 m.

Placement checks the complete obstacle motion envelopes against the floor,
ceiling, wall, fixed fixtures and protected endpoints. Each obstacle body must
intersect the nominal arm/panel guide at its anchor. Endpoint-first placement
and limited lateral variation make crowded scenes possible without relaxing
collision checks. The new boundaries prohibit excursions outside the bay;
they do not guarantee that every successful trajectory weaves vertically or
exclude legal lateral avoidance inside the bay.

## Settings and code

| File | Purpose |
|---|---|
| `configs/construction_site.yaml` | Floor, ceiling, footprint and visual settings |
| `configs/variable_weaving.yaml` | Obstacle counts, size scales, speed range and placement |
| `src/wall_panel_rl/envs/construction_site.py` | Physical slabs, fixtures and construction visuals |
| `src/wall_panel_rl/envs/collision.py` | Shared collision definition |
| `src/wall_panel_rl/rl/variable_scene.py` | On-path scene generation with site constraints |
| `scripts/demo_construction_site.py` | Execute one physical episode and save a recording |

Runtime loading resolves the two editable YAML files. Recordings save the
resolved values, so replay does not silently adopt later configuration changes.
Editing obstacle sizes or heights requires generating new scenes. Existing
recordings remain snapshots of the environment in which they were executed.

## Run

The existing project conda environment works. `environment.yml` records its
dependencies for a separate installation. PyBullet provides the simulator;
Gazebo is not required. A desktop display is needed only for interactive replay.

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/construction_site_v29
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -m pytest -q
```

Replay the new successful two-obstacle demonstration from any working directory:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  /home/johnsonchen/wall-panel-rl/experiments/construction_site_v29/scripts/replay_episode.py \
  /home/johnsonchen/wall-panel-rl/results/diagnostics/construction_site_v29_seed20261117/demo_count2/recording.json \
  --fps 30 --playback-speed 0.6 --timing smooth
```

Replay uses saved poses, precomputed joint updates and fixed frame pacing;
it runs no controller or physics steps and does not export a video.

To execute a fresh physical episode instead of replaying, use a new output path:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/demo_construction_site.py --case checks_site/count2.json \
  --output outputs/site_demo_count2
```

After changing placement settings, regenerate a scene bank with
`scripts/generate_variable_cases.py --output outputs/site_scenes` using the
same Python interpreter. Bank generation and model training are separate actions.

## Validation and current limitations

Geometry checks passed for one scene of each count (2, 3, 4 and 5). These checks
verify placement, endpoints and complete motion envelopes; they do not prove
that a controller can solve every generated scene.

Two physical probes used the unchanged v28 checkpoint:

| Obstacles / seed | Outcome | Steps | Final position error |
|---|---|---:|---:|
| 2 / 3480000 | Success | 184 | 0.874 mm |
| 4 / 3500000 | Timeout, no collision | 240 | 0.652 m |

The held panel stayed within the physical height limits in both recordings.
Neither probe registered a completed vertical-weaving passage. These are
individual demonstrations, not a measured success rate. Adaptation and a
separate multi-seed evaluation are still needed for the changed environment.
Results, diagnostics and playback timing are saved under
`results/diagnostics/construction_site_v29_seed20261117/` in the project root.
