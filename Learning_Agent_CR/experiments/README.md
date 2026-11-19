# Frozen current-layout validation results

These files preserve the completed `four_corridor_validation_v20_seed181` study. They are inputs to the submission report, not outputs from its short PPO software check.

- `summary.json` and `metrics.csv`: aggregate counts and rates for the same 64 scenes per controller.
- `episodes.csv` and `paired_cases.csv`: individual outcomes and paired comparison.
- `failures.csv` and `failure_diagnosis.json`: collision and timeout diagnostics.
- `verification.json` and `frozen_inputs.json`: study integrity assertions, split/construction counts, confidence intervals and source hashes.
- `geometry_summary.json` and `weaving_examples.json`: layout checks and successful measured vertical-passage examples.
- `success_rate.png` and `timeout_errors.png`: outcome comparison and terminal-error figures.
- `environment_versions.json`: original study version information.

The original mode label `ppo_progress` means PPO plus prediction and local progress recovery; the public submission CLI calls it `enhanced`. The `ppo` label means the actor alone with the same environment wrapper. Both use the unchanged checkpoint. No BC/PPO optimizer updates were performed during this validation study.

The denominator is **64 valid scenes per mode**. Enhanced: **43 successes, one collision, 20 timeouts**. PPO alone: **zero successes, 64 collisions, zero timeouts**. Successful installation with the additional measured vertical-passage pattern: **9/64** enhanced episodes. The selected seed-1166026 demo is outside this bank.

These are validation results, not an independent final test. Complete per-episode trajectory recordings for all 128 runs are not bundled to keep the submission compact. The frozen case bank, model, execution code and aggregate/per-episode tables are included; full and excerpt recordings are provided for the selected demonstration. New runs can save their complete recordings with `scripts/evaluate_agent.py --save-recordings`.
