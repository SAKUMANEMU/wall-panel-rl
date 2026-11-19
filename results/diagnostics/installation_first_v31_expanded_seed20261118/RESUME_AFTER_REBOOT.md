# Resume expanded verification after reboot

The entire process group is paused. No training or model updates have occurred.
All 80 fresh scenes are saved; all 8 diagnostic episodes and
64 of 160 expanded verification episodes are persisted. Remaining
primary episodes: 96. Completed case/model/runtime hashes and
recording files have been checked. In-progress episodes restart after reboot.

After reboot, ask the assistant to resume validation in the background, or run
this command in a terminal and keep that terminal open:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  /home/johnsonchen/wall-panel-rl/experiments/installation_first_v31/scripts/run_expanded_validation.py \
  --source-run /home/johnsonchen/wall-panel-rl/results/diagnostics/installation_first_v31_seed20261118 \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/installation_first_v31_expanded_seed20261118 \
  --per-count 20 --workers 4 --generator-workers 2
```

The same protocol is required. Saved scene-generation jobs and completed
physical episodes are reused. Model weights, controller parameters, and frozen
inputs are verified; the held process lock is released by the reboot. Do not
launch a second copy while the original process group remains paused.

Progress: `status.json`. Detailed pause record: `pause_status.json`.
