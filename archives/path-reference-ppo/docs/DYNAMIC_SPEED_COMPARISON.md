# Fixed-speed continuation with90% dynamic scenes

The user requests more dynamic and fewer static training scenes and confirms
90% dynamic /10% static. Apply the same new probability to both arms. Controller
ownership remains an independent50% RRT /50% PPO episode coin.

Both arms start from `four_dynamic_ppo_v8_seed61/selected_model.zip`, the selected
v5+2,000-step candidate. Restore its PPO Adam/critic and matching
`auxiliary_bc/aux_2000.pt`: global PPO counter62,000, PPO update count66,
BC update count2, BC Adam state, RNG, and1,467 verified expert pairs. The two
arms share training seed71, validation branch seed61, learning rate3e-6, all
remaining PPO/BC settings, and the same frozen192-case training bank.

Physical episodes restart in fresh environments. Unfinished old physics
episodes and old environment RNG streams are not continued bit for bit; instead
both new arms receive matching new seed streams. A resumed learned state must
not be described as full simulation-state restoration.

| Arm | Fixed training speed | New PPO transitions | Target validation speed |
|---|---:|---:|---:|
| slow01 |0.01m/s|2,000|0.02m/s|
| fast02 |0.02m/s|2,000|0.02m/s|

There is no curriculum progression within either arm. Evaluate at0,1,000 and
2,000 added PPO transitions. Keep the original target-speed, paired-frozen and
old-four-static selection guards. Include initialization as a candidate, and
keep an earlier checkpoint on a tie. The last model is not necessarily selected.
Each final model has4,000 added adaptation steps since v5 (2k inherited+2k new);
the4,000 new steps across this comparison belong to two separate models.

The90/10 probability governs newly sampled training episodes. Actual finite
episode proportions may differ. The inherited BC pool is preserved intact,
including its prior static/dynamic composition. New RRT labels still require
time checks, successful physical execution and a matching fresh physics replay.
Only PPO transitions enter the on-policy buffer. Auxiliary BC runs before
collecting each rollout, never midway through collection. New accepted/rejected
BC updates are reported separately from the two inherited updates.

The fixed64 target validation cases, frozen companions and original64 static
regression cases are unchanged. Common RRT validation rows are reused from their
actual v8 execution and matched coin assignments. The previously used100-case
test stays excluded and is not reevaluated. Any future final model requires a
fresh reserved bank after configuration selection and multi-seed checks.

This experiment isolates training speed *within* the new90/10 composition.
It has no70/30 arm and cannot attribute an improvement specifically to the
proportion change. Identical rules and seeds do not imply identical resulting
episodes: speed changes can change terminations, plans and sampled data counts.
One paired training seed also does not establish general statistical stability.

## Commands and artifacts

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
# Check only. Never starts training without --run.
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/run_dynamic_speed_comparison.py

# Current detached comparison has already launched; inspect it instead of
# launching again into the same protected directories.
cat /home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_speed_comparison_v9_seed71/status.json
tail -f /home/johnsonchen/wall-panel-rl/logs/four_dynamic_speed_comparison_v9_seed71.log
```

Comparison configuration: `configs/four_dynamic_speed_comparison.yaml`.
Arm profiles: `configs/four_dynamic_speed_slow01.yaml` and
`configs/four_dynamic_speed_fast02.yaml`.

Reports are directly visible under:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_speed_comparison_v9_seed71/
/home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_speed_v9_slow01_seed71/
/home/johnsonchen/wall-panel-rl/results/diagnostics/four_dynamic_speed_v9_fast02_seed71/
```

Each arm stores model checkpoints, episode journals, new BC updates/restoration
metadata and full validation rows separately. The comparison generates
`README.md`, `curves.png`, `checkpoints.csv`, `summary.json` and
`verification.json`. All source/default models and previous reports are kept.

## Completed result

Both arms select their+2k continuation checkpoint (v5+4k in this adaptation
chain). Their target-speed pure PPO counts tie at47success/15collision/2timeout,
versus common source43/18/3. Original-static validation remains57/7/0 for both.
Paired-frozen counts are45/15/4 (slow01) and46/15/3 (fast02).

Fixed mixture counts are53/10/1 and54/9/1, versus source51/11/2. The selected
models disagree on two target outcomes:one collision/success and one
success/collision. Their fixed mixture owners differ, making the mixed totals
different despite equal full pure-PPO counts. Both satisfy the predeclared
primary selection criterion; the one-case mixed difference needs replication.

Each arm adds two BC updates to the two inherited updates. Slow01 completes
40RRT successes/1timeout and28PPO episodes(11S/15C/2T);fast02 completes38RRT
successes/1timeout and27PPO episodes(10S/15C/2T). Actual completed scene counts
are66dynamic/3static and63dynamic/3static;90/10 is the sampling probability,
not an exact finite-run quota. Successful teacher labels pass the replay gate;
RRT timeouts do not enter BC. The inherited1467 labels remain present.

These are validation results; the old100-case dynamic test is not rerun, and
neither model has a new independent success-rate estimate yet.

To inspect the fast02 candidate's learned PPO branch (fresh random scene):

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_slow_dynamic.py --controller ppo --model results/diagnostics/four_dynamic_speed_v9_fast02_seed71/selected_model.zip --manifest results/diagnostics/four_dynamic_speed_v9_fast02_seed71/candidate_manifest.json
```

For the full50/50 system, use `--controller mixture`. Use the slow01 report
paths instead to inspect the other arm. The default demo remains unchanged.
