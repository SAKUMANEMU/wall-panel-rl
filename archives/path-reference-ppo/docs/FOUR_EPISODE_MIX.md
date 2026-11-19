# Four-static50/50 episode mixture

The expert branch uses the existing verified RRT-Connect/ClassicalController,
not another BC neural network. The other branch uses the selected PPOv5 model
(the84/100 pure-policy candidate). RRT is compatible with the current seven
normalized joint-increment actions and same reference endpoint, so a separate
BC-model branch is unnecessary in this implementation.

Each new episode draws one private Bernoulli coin:expert probability0.5. That
controller owns the entire episode; there is no per-step averaging or switching.
Scene sampling, controller choice and planner randomness use different RNG
streams. A finite evaluation need not contain exactly half of each branch.
Planner failure counts as an RRT episode failure; it is not silently replaced
by PPO or a favorable scene.

Training uses the existing192-case bank and inherited case probabilities. RRT
episodes finish inside the reset pipeline, are journalled, and then another new
episode is drawn. SB3 receives only PPO episodes, their sampled actions and their
original log probabilities. The PPO budget therefore counts PPO transitions;
RRT physics/planning is additional and reported separately. PPO Adam/value state
and0.01 reward scale are preserved from the source; base actor/std remain frozen.

Only RRT trajectories with successful physical installation, pre/post static
route verification and unchanged original policy reference enter the separate
expert sample pool. Auxiliary BC uses a separate actor-only Adam, LR1e-6,
weight0.05,one update per rollout,KL bound0.003. Updates occur at the NEXT rollout
start,before sampling actions. Never insert RRT actions into the PPO rollout
or update actor weights midway through a PPO collection. This is RRT expert
execution plus PPO/auxiliary imitation,not two independently trained networks.

Validation runs both full pure PPO and the50/50 mixture. A fixed coin seed gives
the same branch assignment across checkpoints. Save mixture totals and RRT/PPO
breakdowns. Pure-PPO validation selects the learned checkpoint; mixed-system
success is a separate metric. `--mode all` also runs a full pure RRT evaluation.
All use existing validation/regression cases,not a fresh independent test.

At interface installation, no formal training budget had been started.
Functional physical preflight:8 mixed episodes,2 RRT successes,
4 PPO successes and2 PPO collisions. These8 are not a rate estimate. A separate
training-wrapper probe runs one successful RRT episode(56 steps/56 verified
labels),then one successful PPO episode(73 steps),with no optimizer update.
The source model file is unchanged. Unit tests also perform8 artificial PPO
transitions and verify exact on-policy likelihoods across auxiliary BC updates.

Run commands:

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
# Check configuration only; default never launches a run.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py

# Start configured10,000 PPO transitions,plus additional RRT episodes.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py --train

# Full validation of the current source,with episode mixture.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py --evaluate --output results/diagnostics/four_episode_mix_full_validation_v6_seed61

# After training,evaluate its chosen checkpoint in all three modes.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py --evaluate --mode all --model results/diagnostics/four_episode_mix_v6_seed61/selected_model.zip --output results/diagnostics/four_episode_mix_trained_validation_v6_seed61

# Play one fresh episode using the existing current model and random routing.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_expanded.py --controller mixture --expert-probability 0.5
```

RRT planning has a15s search budget plus final checks for each expert episode;
wall time increases relative to pure PPO. The graphical demo hides planning
queries and retains0.6x playback. All runners refuse overwriting result folders.
`--limit` and `--mode/--model/--output` affect evaluation only; training always
uses the configured50/50 episode mixture. Formal settings are in
configs/four_episode_mix.yaml. Physical preflight results and executed source
snapshots are in results/diagnostics/four_episode_mix_preflight_v6_seed61.

## Completed formal run and controlled follow-up

The original 10k formal run completed on 2026-11-04. Pure-PPO validation at
0/5k/10k was 56/55/50 successes out of64; mixture validation was60/60/55.
The selected checkpoint remains the original PPOv5 source. See
[formal report](../results/diagnostics/four_episode_mix_v6_seed61/README.md).

Replay of the six lost successes found three panel/obstacle4 collisions and
three traversal timeouts. The immutable reference arrays were unchanged, while
reference-cursor advancement stalled. Repeated training-only RRT labels also
show route ambiguity, which is a possible concern for MSE imitation but does
not establish why training regressed. See
[diagnosis](../results/diagnostics/four_mix_regression_v7_seed61/README.md).

The separate `configs/four_episode_mix_nobc.yaml` profile starts from the same
original source and runs5k PPO transitions with
`auxiliary_bc.updates_per_rollout: 0`. RRT execution stays enabled at50% episode
probability. All other algorithm settings, data, case probabilities and seed
remain identical. This is a matched-budget ablation against the original5k
checkpoint, not a comparison against its10k final candidate. Existing output
directories are protected from overwriting; change output paths for a rerun.

```bash
# Inspect only; this does not start training.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_four_episode_mix.py --config configs/four_episode_mix_nobc.yaml
```

Live status: `results/diagnostics/four_episode_mix_nobc_v7_seed61/status.json`.
After completion, `scripts/summarize_four_mix_ablation.py` verifies the matched
protocol, preserved source/optimizer initialization, zero BC updates, identical
evaluation routing, and training/test separation before generating comparisons.
No independent test is used for this validation ablation, and no default model
is changed automatically.

The no-BC run finished on 2026-11-05: pure-PPO50/11/3 and mixture55/7/2 out of64,
versus matched BC-on55/8/1 and60/4/0. All5,438 expert observation/action pairs
are identical across runs; auxiliary updates are5 versus0. Selected weights
remain the original source. See
[matched comparison](../results/diagnostics/four_mix_ablation_v7_seed61/README.md).
