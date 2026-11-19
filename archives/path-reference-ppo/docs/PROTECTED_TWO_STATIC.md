# Two-static curriculum: warm-up, initialization correction and protected PPO

All experiments below preserve the original one-static checkpoint and global
demo default. No three-obstacle or dynamic stage is automatically started.

## First attempt: stronger BC without actor protection

Add source obstacle 3 after the existing middle obstacle 2, retaining the middle
obstacle in slot0. Collect 256 successful demonstrations: 128 two-static RRT,
64 one-static RRT and 64 frozen-reference alignment episodes. Across 258 attempts,
two fail; successful labels total is recorded in teacher/metadata.json. All three
groups pass physical observation/action replay exactly. Use 54 training layouts
and ten held-out BC layouts; fresh test scenes never enter gradients or selection.

Train the shared residual and gate for10k BC updates, sampling two/one/alignment
at60/30/10%, swapping two present obstacle slots with probability.5, and upweighting
states needing expert corrections. Select2,500 BC updates: two-static validation
7/20 success,12 collision,1 timeout; one-static6/20,13 collision,1 timeout.
Fresh two-static100:51/44/5; one-static regression drops from95/2/3 to42/52/6.
The first trial does not start PPO because its warm-up gate fails. Its checkpoints,
teacher data and source versions remain archived at
`results/diagnostics/two_static_warmup_v1_seed61/` (including code_archive/).

Freezing the reference servo does not freeze the learned original one-static
policy: modifying its shared residual/gate can still cause forgetting.

## Initialization diagnosis and correction

One-static observations zero all four unused obstacle slots (input columns30:70).
Random input connections for those slots therefore did not contribute to the
trained one-static actor. Activating another obstacle can drastically change its
output. With the original weights, two-static validation gives0/20 success and
20 collisions. Zeroing those input columns in the residual and gate first layers,
without learning, gives15/20 success,4 collisions,1 timeout. One-static remains
18/20 success,0 collisions,2 timeouts.

The corrective run initializes those connections to zero and permits actor
updates only to columns30:70 of `residual.0.weight` and `prior_gate.0.weight`.
Freeze all other original actor parameters. One-static inputs are exactly zero
in those columns, so its deterministic actions and fixed-std action distribution
are preserved. PPO may train its separate critic. Reapply the protection after
loading any checkpoint for continued training; inference needs no gradient hook.
Fresh zero-moment Adam and zero weight decay are required for the protected
connections: stale actor moments can otherwise move zero-gradient old columns.

Tests include artificial PPO updates/save/reload and full real-run parameter
comparisons. Both the selected and final raw PPO actors preserve original
one-static connections. Real teacher one/alignment inputs are exactly zero in
the protected columns.

## Corrective run and outcomes

Reuse the verified teacher data, perform2,500 protected BC updates, then a bounded
10k PPO continuation if the selected model passes the warm-start gate (two-static
success>=50%, collisions<=25%, no limits; preserve one-static>=80% with zero
validation collisions, alignment and endpoint convergence). Model release uses
the stricter80% success/10% collision thresholds.

Protected BC never exceeds its initialized15/20 baseline, so selected BC update
is0, although the whole2,500-update experiment runs. PPO starts from that qualified
initialized model, with fresh Adam, LR5e-5, four synchronous250-step workers and
the same per-episode50/50 RRT/PPO mixture. Independent auxiliary BC uses three
batches per rollout, training labels only, and the actor-gradient mask.

| Additional PPO steps | Two-static success | Collision | Timeout | One-static success |
|---:|---:|---:|---:|---:|
|0|15/20|4|1|18/20|
|5,000|15/20|4|1|18/20|
|10,000|14/20|4|2|18/20|

Select PPO step0. Neither BC nor PPO updates replace the initialization-only
candidate. All10k PPO transitions genuinely execute; raw final added-input weight
norm is nonzero, selected candidate's added-input norm remains zero.

On100 **new** scenes from seed1010000, the original actor gives0 success/100
collision; initialization-only and selected candidate both give85 success,
10 collision,5 timeout. One-static known regression is exactly95/2/3 before and
after; this verifies old-skill preservation. The selected model's50/50 mixture
validation is17/20 success,2 collisions,1 timeout; it includes RRT contribution.

The85% test result comes from initialization correction, not this round's learned
updates. In particular, the selected model's new obstacle inputs still have zero
action weights: it has not learned to react to the second obstacle. The known
20-case validation remains75%, so overall release and dynamic-stage gates fail.
The original default model remains unchanged. Do not reinterpret the first
trial's used100-case bank as a fresh test for this corrective run.

Full regression suite:143 tests passed. The teacher reuse, frozen test hashes,
source checkpoint, warm-up prior/critic/std and old actor protection are verified.

## Files and demos

- Corrective config: `configs/protected_two_static.yaml`.
- Runner: `scripts/run_protected_two_static.py`; default checks only, `--run`
  refuses existing outputs.
- Resolved config/new frozen tests: `data/protected_two_static_v2_seed61/`.
- Report/model package: `results/diagnostics/protected_two_static_v2_seed61/`.
- Warm-up/raw PPO/optimizers: `models/protected_two_static_v2_seed61/`.
- `actor_protection_proof.json` distinguishes selected initialization from raw
  trained parameters; `assessment.json` explicitly marks the release gate failed.

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_two_static.py --controller ppo --playback-speed 0.6
```

This replays a labelled successful validation example (seed900001), not a success
rate estimate. `--controller rrt` shows the expert; omitting it selects RRT/PPO
once with probability.5. `--headless` performs the same episode without a GUI.
The original `demo_autonomous_ppo.py` still shows the verified one-static model.
