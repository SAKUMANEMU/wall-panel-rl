# Four-obstacle BC pilot

Use the unchanged initialized MultiObstaclePolicy and the64 verified four-box
RRT demonstrations. Fit only multi_adapter with1,000 Adam updates, batch256,
learning rate5e-5. Frozen base actor, critic and log standard deviation are
checked byte-for-byte after optimization. No PPO calls are made; the inherited
PPO timestep counter does not advance. No test-bank policy episodes are run.

The separate four_obstacle_bc module leaves historical two-static training
groups/loaders unchanged. Validate77-input/7-action arrays and task presence
masks. Exclude all new validation/test seeds and old heldout/validation seeds
before assembling the gradient pool. Sample80% new four-box routes and20% earlier
tasks (10% two,5% one,5% alignment). Retention targets are the INITIAL policy's
means on earlier training observations, not potentially conflicting old RRT
actions. Freeze these targets before fitting.

Validate pure deterministic policies at0/25/50/100/250/500/1,000 updates on32 new
four-box layouts,20 old two-box layouts and40 one-box/alignment cases. The API's
existing ppo branch name means policy execution, not PPO optimization. No online
planner or collision rejection modifies these evaluation actions.

Select fewer collisions before more success, but bound four-box success loss
to2 episodes; prevented collisions may become timeouts. Require old two-box
success/collision non-regression and exact one/alignment trajectories. Ties keep
the earlier checkpoint. Short PPO readiness additionally requires a learned
candidate with at least50% four-box success and no more than25% collisions.
Neither readiness nor selection changes the default animation model.

Observed results on the32 validation layouts:

| BC updates | Success | Collision | Timeout | Old two-box success/20 |
|---:|---:|---:|---:|---:|
|0|4|28|0|15|
|25 (selected)|7|25|0|16|
|500|16|16|0|13|
|1,000|17|15|0|8|

The four-box task improves with training, but the shared adapter also changes
old two-box behavior. Only25 updates satisfy the retention rule among improved
candidates. The candidate is not ready for PPO. Old one-box/alignment trajectories
remain identical; default source weights stay unchanged.14 isolation, optimizer,
serialization and compatibility tests pass.64 new test layouts remain reserved.

Run commands (a completed run refuses overwrite):

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_four_obstacle_bc.py
# Add --run only for a new, separately configured output directory.
```

Checkpoints/BC optimizer states:models/four_obstacle_bc_v1_seed61/bc/checkpoints/.
Report:results/diagnostics/four_obstacle_bc_v1_seed61/README.md. The report and
curves are mirrored under the main visible project's results/diagnostics.
