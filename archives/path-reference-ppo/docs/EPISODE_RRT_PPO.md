# Per-episode RRT/PPO mixture — 2026-11-02

Each episode independently selects RRT-Connect with probability 0.5, otherwise
PPO, and keeps that controller until termination. A finite group need not contain
exactly half of each. This applies to training and mixture evaluation.
RRT is the expert; **BC names the supervised learning loss, not a third runtime
controller**. The PPO branch retains its frozen BC tracking prior + PPO residual.

## Data separation and updates

- The existing ClassicalController plans to the saved reference's same joint
  endpoint using actual collision checks. Its avoidance path is never substituted
  into the policy's obstacle-free reference features.
- Entire RRT episodes run inside a wrapper reset, before a fresh PPO episode is
  returned to SB3. Log every episode. Failed planning counts as an RRT failure,
  without automatic fallback. Only successful RRT episodes provide labels.
- Stock PPO therefore stores only its original Gaussian samples, log probabilities,
  rewards and values. GAE, episode boundaries and timeout bootstrapping are retained.
- After updating a PPO rollout, perform BC at the next rollout start, before new
  PPO sampling. Update residual/gate parameters only, with separate Adam state;
  the critic and frozen prior receive no BC updates.
- Bound sampled Gaussian KL change and roll back actor plus BC optimizer state
  if it exceeds the limit. This is a sampled update check, not a safety guarantee.

Four synchronous DummyVecEnv workers preserve the 4 x 250 PPO batch. Use the
original 54 training layouts, excluding held-out teacher seeds. Current course:
one static obstacle, zero sensing noise, 240-step horizon. No dynamic stage starts.

## Professor-preferred model selection

Among candidates preserving success >=80%, collisions <=10%, alignment and goal
convergence, prefer fewer collisions, then success and timeout rates. An increase
in timeout no longer automatically vetoes a collision reduction. Exclude limits.

The adopted existing PPO reproduces 17/20 successes, 0 collisions, 3 timeouts and
20/20 alignment. No new training produced it. The former 18/20, two-collision
selection and files remain archived. Its 282/300 regression score does not belong
to the newly adopted checkpoint. Zero collisions in 20 cases is finite-sample
evidence, not a guarantee. New selection:
`results/diagnostics/safety_first_ppo_20261102/`.

## Run

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
# Check inputs only:
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/train_episode_rrt_ppo.py
# Random controller for this whole animation:
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_episode_mixture.py --seed 1000000 --playback-speed 0.6
```

Omit `--policy-seed` for a new random controller draw on each invocation. Seed 0
reproduces PPO selection; seed 2 selects RRT. `--controller ppo` / `--controller rrt`
explicitly select a controller. The original `demo_autonomous_ppo.py` still defaults
to pure PPO, now using the selected 5k model from the completed mixed experiment.

`train_episode_rrt_ppo.py --evaluate` compares pure PPO, pure RRT and mixture on
identical frozen scenes, refusing existing output paths. Each training checkpoint
also records mixture evaluation with common branch assignments. Selection uses
pure PPO scores, so RRT performance cannot hide PPO regression.

Formal training requires `--train`. The first pilot is now complete: 10,000 PPO
transitions plus 7,030 RRT steps, with checks at 5k and 10k. It completes 147 RRT
episodes (all successful) and 138 PPO episodes (101 successes, 26 collisions,
11 timeouts); four PPO episodes remain partial at the budget boundary. These
stochastic training counts are separate from deterministic evaluation.

Retain the 5k checkpoint: pure PPO validation 18/20 success, zero collisions,
two timeouts; mixture 19/20 success, zero collisions, one timeout. The 10k actor
has 18/20 success, one collision, one timeout, and is rejected by collision-first
selection. All alignment checks pass. Ten BC updates are accepted, no KL rollback;
the selected 5k checkpoint has five of those updates. Source/code/frozen-weight
and sampling checks pass. These known validation results do not replace a new
independent test or a matched pure-PPO training control.

Config: `configs/episode_rrt_ppo.yaml`. Initial BC defaults: weight .1, LR 1e-5,
one batch of 100 per rollout, capacity 50,000, sampled KL cap .005. These are
initial settings, not optimized values. PPO retains LR 1e-4, three epochs, clip
.05 and target KL .005.

Completed training outputs: `models/episode_rrt_ppo_v1_seed61` and
`results/diagnostics/episode_rrt_ppo_v1_seed61`. Read `README.md`, `overview.png`,
`validation_curve.csv` and `verification.json` in the latter. The selected model
is accompanied by `selected_aux_state.pt` with matched optimizer/RNG/pool state.
The CLI refuses to overwrite this completed run; use new output paths for a new study.
The separate no-training comparison is in
`results/diagnostics/episode_rrt_ppo_preflight_seed61`.
