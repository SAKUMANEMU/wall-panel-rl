# Paired fixed-speed continuation (90% dynamic, 10% static)

Both arms start from exactly the selected v8+2k policy/PPO Adam/critic and matched BC Adam/pool/RNG. Each uses fresh physical episodes with the same training seed71 and validation seed61,50/50 RRT/PPO ownership, and2,000 new PPO transitions. Only the fixed training speed differs. All target validation uses0.02m/s. This is a single-seed validation screening, not an independent-test result.

Counts are success/collision/timeout out of64.

| Arm | Additional PPO steps | Dynamic target | Paired frozen | Original static | Mixture |
|---|---:|---|---|---|---|
| slow01 | 0 | 43/18/3 | 44/16/4 | 57/7/0 | 51/11/2 |
| slow01 | 1000 | 45/17/2 | 45/16/3 | 57/7/0 | 53/10/1 |
| slow01 | 2000 | 47/15/2 | 45/15/4 | 57/7/0 | 53/10/1 |
| fast02 | 0 | 43/18/3 | 44/16/4 | 57/7/0 | 51/11/2 |
| fast02 | 1000 | 46/16/2 | 45/16/3 | 57/7/0 | 53/10/1 |
| fast02 | 2000 | 47/15/2 | 46/15/3 | 57/7/0 | 54/9/1 |

## Selection

- slow01: select +2000 new PPO steps (v5+4000 total for this adaptation chain); target 47/15/2.
- fast02: select +2000 new PPO steps (v5+4000 total for this adaptation chain); target 47/15/2.

Validation winners beyond the common starting model: ['slow01', 'fast02']. The final2k continuation and selected checkpoint are stored separately for each arm. No default file is overwritten.

The pure-PPO primary scores tie in this run. The two selected models have different outcomes on two target cases; one slow success becomes a fast collision and one slow collision becomes a fast success. The latter is PPO-owned in the fixed mixture while the former is RRT-owned, so the fast mixture has one more success. `paired_selected_cases.csv` exposes this assignment effect. A one-case mixed advantage is not evidence of a generally superior speed.

## Limits and accounting

The90/10 probability applies to new training scene episodes; finite observed proportions can differ, and the inherited BC pool is preserved intact. Both arms change that proportion together, so this experiment cannot isolate the effect of90/10 versus70/30. It isolates training speed within the new shared proportion. RRT trajectories and episode counts can diverge as a consequence of the speed change; input rules/seeds are paired, not the resulting physics. Fresh environments replace unfinished old episodes; this is learned-state continuation, not bitwise physics/RNG continuation of the previous run.

The previously used100-case test remains excluded and is not reevaluated. A later final model needs a fresh reserved test after configuration selection and multi-seed checks. `verification.json` audits source/optimizer/pool/RNG matching and data isolation. `summary.json` contains each arm's PPO, RRT and verification-replay transition totals; audit replay steps are not training samples.

See `curves.png`, `checkpoints.csv` and each arm report's validation JSON/checkpoints. Existing v5/v8 source models are preserved.
