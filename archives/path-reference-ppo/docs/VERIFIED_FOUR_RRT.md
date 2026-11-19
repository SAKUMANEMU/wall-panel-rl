# Verified static RRT and four-obstacle demonstrations

The earlier99-scene baseline found99 paths but physically succeeded97 times.
Four saved paths failed a sampled5mm-clearance recheck. Two collided during
execution; two happened to execute successfully. Original reports and their
archived code remain unchanged.

Static `ClassicalController` now validates edges at both the original grid and
a refined0.005-radian grid over the actual execution waypoint subdivisions.
RRT-Connect has an optional final-route validator. It validates direct routes,
shortcuts and completed routes; rejection continues search within the existing
iteration/time budget. The controller independently validates its final dense
route before reporting planning success. A failed verification cannot become an
executable plan. Timed/moving-obstacle behavior is unchanged. These remain finite
sample checks, not continuous collision certification.

The same99 scenes are reevaluated at their original5mm planning clearance and
15s search budget. Primary and two-seed failure diagnostics stay separate.
Source observations, physical collision thresholds, actuator gains,240-step
horizon and initialized weights are unchanged. The new report is
`results/diagnostics/four_on_path_rrt_verified_v2_seed61`.

Prepare NEW scene banks:64 training,32 validation,64 test, starting at seeds
1020000/1022000/1024000. The old99 remain regression only. Use10mm planning
clearance for the demonstrations as extra tracking margin; physical collision
termination remains unchanged. Freeze all banks before expert collection. The
nominal no-obstacle policy motion is used only for environment placement;
validation/test four-obstacle episodes are not evaluated or labelled here.

For each training layout, allow up to three fixed RRT seeds. Save only one route
that passes pre/post dense-path verification AND physically reaches installation
without collision. Failed attempts never become labels. The policy's original
obstacle-free reference stays in its77 inputs; it is not replaced by the expert
route. Labels are seven normalized joint increments actually executed by the
environment. Raw commands, episode offsets, scene seeds and terminal anchors
are saved separately. Physical replay checks observation/action equality.

`data/four_safe_teacher_v1_seed61/teacher/teacher.npz` is a dedicated
`four_static` dataset, not the historical two-static/one-static/alignment
mixture. A later training interface must explicitly handle that task and skill
retention. This workflow performs no BC/PPO updates and does not replace the
default initialized-policy demo.

The runners refuse overwrites. Configuration checks without `--run` are safe:

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_four_on_path_rrt_validation.py --config configs/four_on_path_rrt_verified.yaml
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/prepare_four_safe_teacher.py
```

See the completed reports in `results/diagnostics/four_on_path_rrt_verified_v2_seed61`
and `results/diagnostics/four_safe_teacher_v1_seed61` for actual counts and quality
checks, rather than assuming all attempts succeed.

Completed outcome: verified RRT99/99 physical successes on the unchanged regression,
versus original97/99. New teacher package64 successful episodes/3,973 transitions,
from64 attempts. Independent64/32/64 banks are frozen. The interrupted52-episode
checkpoint is archived, and every array prefix is exactly preserved after suffix
collection. Three original and three resumed episodes replay with error0. No
BC/PPO updates. `scripts/resume_four_safe_teacher.py` records the resume protocol
and refuses reuse; it does not regenerate or overwrite the frozen banks.
