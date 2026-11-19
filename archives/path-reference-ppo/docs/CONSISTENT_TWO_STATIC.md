# Wider failure coverage and canonical RRT demonstrations

The completed run adopts no learning update. Raw BC3k gives21/64 success and43
collisions on new validation versus initialization58/64 and3 collisions. Legacy
validation gives6/20 and14 collisions versus15/20 and4. Select BC0; PPO does not
start because the learned-gain/non-regression gate fails. Retained initialization
scores87/100 success,6 collisions,7 timeouts on the new test. One-static remains
95/2/3 with identical trajectories. All153 tests pass. Do not interpret the87%
as a new BC/PPO training gain.

This experiment follows the failed v3 adapter trial, while preserving the
original model, source code snapshots and all earlier datasets/results. It
starts from the retained zero-output adapter, rather than adopting the failed
v3 candidate. That failed candidate is used only to find correction states.

The previous correction data covered four failing layouts. Add128 fresh
training layouts to the54 original non-held-out layouts. Freeze64 new validation
layouts and a separate100-layout test before fitting. Keep the legacy20-box
validation cases and the old one-obstacle regression bank as regression checks.
None of the new test cases reach training or checkpoint selection.

For each training layout, trace both the initialized actor and the failed v3
actor. Compare three RRT plans from the same physical state with the same goal
and obstacle geometry. Record initial action sign disagreements and joint-route
distance. Select one canonical route by a fixed geometric/reference cost.
Do not average labels from different valid routes. Failure corrections use
checked straight connectors to a literal suffix of the same canonical path;
missing connectors are logged, rather than relabelled as safe demonstrations.

The parent dataset contains21 of64 layout pairs with opposite active-joint
initial directions. However, all64 pairs have different obstacle features
because of phase jitter. This does not establish contradictory labels at
identical observations. The new three-seed audit holds the observations fixed.
Within-layout canonical selection also does not prove consistent route topology
between different, nearby layouts; that remains a limitation.

Old multi-route two-static training labels are removed. The old one/zero-box
anchors and10-layout supervised holdout remain. New consistent correction
episodes receive4x sampling weight. Actor architecture and the3,000-update BC
budget are unchanged from v3 so this is primarily a data-coverage/label change.
The original actor stays frozen, with the learned adapter active only for at
least two obstacles.

Checkpoint selection must preserve success and collision counts on BOTH the
new64 validation cases and legacy20 cases, plus the original one-box skills.
Within that retained-ability group, prefer fewer collisions. A nonzero BC
checkpoint must improve the NEW validation and respond to the second obstacle
before the optional10,000-transition PPO pilot starts. PPO does not receive
expert transitions in its on-policy buffer. Although half of selected episodes
use online RRT, their new route labels are excluded from auxiliary imitation;
the auxiliary pool is restricted to the frozen canonical-route training data.

See the [complete report](../results/diagnostics/consistent_two_static_v4_seed61/README.md),
[checkpoint counts](../results/diagnostics/consistent_two_static_v4_seed61/comparison.csv)
and [curves](../results/diagnostics/consistent_two_static_v4_seed61/overview.png).
The report records whether learning updates were adopted, and whether the
validation improvement carries over to the untouched new100-scene test.

Run from `/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo`:

```bash
# Configuration only; no training.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_consistent_two_static.py

# Explicit diagnostic candidate; default is pure policy, 0.6x playback.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_consistent_two_static.py
```

The demo selects a successful episode for visualization and cannot estimate a
success rate. Add `--headless` to check it without opening a GUI. The original
default demo and all previous two-static candidates remain available. The runner
refuses to overwrite an existing study directory.

An interrupted pre-training collection can be continued with
`--run --resume-collection`. Configuration, frozen input hashes and model source
hashes must match; an already-started BC run cannot be resumed through this
entry point. Dangerous RRT candidates are rejected before collecting labels.
The original interrupted source, log and80-layout checkpoint remain under
`results/diagnostics/consistent_two_static_v4_seed61/interrupted_collection_1`.
