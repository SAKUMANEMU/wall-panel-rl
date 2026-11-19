# Frozen v31 expanded verification

- 80 fresh geometry-generated scenes: 20 each with 2, 3, 4 and 5 dynamic obstacles.
- Each scene is executed by the frozen selected v31 bundle and the original v29 bundle: 160 primary episodes.
- Existing dual-obstacle failures 4720000 and 4720100 receive a separate frozen 2 x 2 model/controller cross-check: 8 diagnostic episodes, excluded from the primary denominator.
- No retraining, parameter tuning, model selection or success-based scene filtering.
- Original 240-step budget, installation tolerances, floor/ceiling, obstacle sizes and motion distribution retained.
- Wilson intervals and paired wins/losses supplement the aggregate success rate.
- All outcomes, recordings, generation retries and per-job hashes are retained. Cached finished jobs can be reused after interruption.

Progress: `status.json`; fresh-case coverage: `scenes/coverage.json`; paired results: `expanded/results.json`; two-case analysis: `DUAL_TIMEOUT_ANALYSIS.md` and `dual_timeout_analysis.json`; final report: `RESULTS.md` and `summary.json`.

The isolated interface check starts at the goal and is not counted in any evaluation total.
