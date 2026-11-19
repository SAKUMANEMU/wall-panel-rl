# Collision diagnosis and matched wall-feedback PPO pilots

The selected predecessor is `stall_recovery_v3_seed61/ppo/best_model.zip`: 14/20
original validation success, 3 collisions, 3 timeouts; previous 100-scene test
78/13/9. Both new variants resume this exact actor/critic and PPO optimizer, with
the same fresh sampling seed 61. The task remains one static obstacle, 240 steps,
77 observations, seven joint increments, fixed target and unchanged collisions.

Diagnosis changes the proposed reward target: all five training collisions are
panel-wall; validation has two panel-wall and one panel-obstacle; the 13 known
previous-test collisions are 12 panel-wall and one self collision. Existing
obstacle-only proximity feedback cannot directly penalize these wall approaches.
Do not simply multiply the obstacle-only penalty or infer wall clearance from
the obstacle distance cap. No expert data comes from validation/test cases.

- Control: existing obstacle clearance weight .5 and stall feedback, wall weight 0.
- Treatment: identical, wall weight 1. Both use the same `WallAwareEnv`, so query
  costs are comparable and the ONLY training-setting difference is wall weight.
- Training-only wall penalty is `-weight * clip(1 - d/.02, 0, 1)^2`, with d the
  capped minimum arm/panel-to-wall distance in metres. Nominal goal gap .025 m
  lies outside the penalty band; successful terminal installations get zero wall
  penalty. This never changes actions, collision margins, waypoints or done flags.
- Each variant runs exactly 10000 additional PPO environment transitions, four
  workers x 250 steps, LR 1e-4, three epochs, clip .05, target KL .005 and fixed
  std .10. No BC update, automatic stage advance or budget extension.
- Validation at 0/5000/10000 uses the ORIGINAL unmodified evaluation reward and
  same frozen scenarios. Fresh validation of the starting actor precedes workers.
  Evaluation preserves training RNG state. Both selectors retain step 0 if needed.
- Before training, fix this non-regression rule: retain alignment/goal convergence,
  no limits, success >= starting 14/20, collision <= 3/20 and timeout <= 3/20.
  Among eligible candidates prefer fewer collisions, then more successes, then
  fewer timeouts. Apply the same rule to both variants and the baseline. Independent
  results never select the winning model. Stage promotion is separately assessed
  using the existing 80% success / <=10% collision and two distinct post-update
  checks; even a pass does not automatically start a dynamic stage.
- Freeze 100 new scenes from seed 1000000 onward, disjoint from training, original
  validation and already-seen 980000/990000 banks. After both selections, score
  baseline and selected variants on the new bank once. The old 78% and any new
  percentage are different banks; compare models within the new bank.

One sampling seed and reused 20-case validation are engineering evidence, not a
statistical significance claim. Weight 1 is a pilot choice; diagnosis identifies
the contact type, not proof that this weight must improve generalization.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_clearance_ppo.py
# --run executes the two bounded pilots once, refusing existing outputs.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/run_clearance_ppo.py --run
```

Config: `configs/clearance_ppo_study.yaml`. Main-checkout report:
`results/diagnostics/clearance_ppo_v3_seed61`.
