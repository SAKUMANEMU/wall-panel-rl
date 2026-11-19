# Conditional recovery PPO integration and bounded adaptation

The validated conditional-reference rule is now an explicit train/eval contract:

- `ConditionalRecoveryEnv`: inference/evaluation, original evaluation reward.
- `ConditionalTrainingEnv`: cooperative composition of ConditionalRecoveryEnv,
  WallAwareEnv and StallAwareEnv. Physics/projection and existing reward terms
  run once; conditional recovery changes only the next local reference input.
- Both use the same 77 features, seven actions, .30 normal recovery, .20 terminal
  radius, bounded .45 conditional extension, 240-step horizon and static scenes.
  No runtime expert route, actuator replacement, new goal or success tolerance.

One exactly-10000-transition PPO continuation from the currently retained actor
and PPO optimizer. No BC updates. Preserve four workers x 250 rollout steps,
batch 100, three epochs, LR 1e-4, clip .05, target KL .005, fixed std .10, existing
obstacle (.5), wall (weight1/distance.02) and stagnation rewards. Held-out teacher
layout seeds remain excluded from the 54-layout PPO training bank.

Preflight scores the same source weights under the old and conditional controller
to separate interface gains from learning. Initial conditional results must
reproduce 18/20 success, 2 collisions and zero timeouts before learning. Initial
release scoring and all checkpoint scoring use ConditionalRecoveryEnv, not the
old static wrapper. Validation at 0/5000/10000 excludes the unrelated five-dynamic
diagnostic. Retain the initial conditional actor if updates regress in success,
collisions or timeouts. This may keep step 0 even though the whole budget ran.

Monitor records complete episodes; the callback additionally counts recovery
attempts/successes, stall events and wall/stall reward sums over ALL 10000 sampled
transitions, including partial episodes. Stochastic training can trigger the
stagnation guard less often than deterministic evaluation; report actual counts.

Assess two distinct updated checkpoints against the existing static thresholds
(success >=80%, collisions <=10%, alignment retained), plus the selected model's
gate. No automatic dynamic stage or budget extension. Lock validation selection,
then replay three already-seen 100-target banks with alignment checks. These 300
cases are regression evidence, not fresh independent tests or selection data.

Export `selected_model.zip` together with `controller_config.json` and a manifest
containing the conditional evaluation settings. The demo detects those settings
and uses the correct execution wrapper. A plain model zip alone does not encode
the environment's recovery rule.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_conditional_ppo.py
# Explicit --run starts the one bounded adaptation, refusing existing outputs:
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/train_conditional_ppo.py --run
```

Config: `configs/conditional_ppo_adaptation.yaml`. Visible report:
`/home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_ppo_adaptation_v3_seed61`.

Completed results (2026-10-30): the conditional baseline is 18/20 success,
2 collisions, 0 timeouts. Additional 5k: 16/20, 1 collision, 3 timeouts;
additional 10k: 17/20, 0 collisions, 3 timeouts. All alignment checks pass.
Both updated checkpoints pass the minimum stage thresholds, but both fail
non-regression against the conditional baseline, so selected additional step is 0.
The 300 already-seen regression scenes reproduce 282 successes, 14 collisions,
4 timeouts for both before and retained models. This is not new-test evidence.

Actual 10k stochastic sampling completes 135 episodes (110 static, 25 alignment),
plus 264 steps of partial episodes. There is just one conditional recovery
attempt, with zero successful advances; no claim that PPO learned recovery.
Source and frozen weights, training inputs/code and selected per-case outputs
are verified. Full pre-training suite: 115 passed. No dynamic stage was started.
See the visible report's `README.md`, `overview.png`, `training_episode_summary.json`
and `verification.json`. The default autonomous demo now uses this report's
selected checkpoint together with the conditional execution settings.
