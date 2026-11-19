# Resume after reboot

The run was paused for reboot at 32,768 of 65,536 additional PPO steps.
All seven original background processes were verified stopped. Training has
not been restarted. Model weights and the saved PPO optimizer were successfully
loaded in a read-only recovery check.

The current `ppo_32768` validation has 10 saved rows out of 16. Disk recovery
reuses completed evaluations and runs only missing scenes. BC is not repeated.
Training continues for the remaining 32,768 PPO steps, with validation and the
independent test retained. Physical environments and random streams restart;
this is not bitwise continuation of the interrupted episode.

After reboot, ask the assistant to resume in the background, or run:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  /home/johnsonchen/wall-panel-rl/experiments/construction_site_v29/scripts/resume_known_training.py \
  --output /home/johnsonchen/wall-panel-rl/results/diagnostics/construction_site_v29_training_seed20261117/training \
  --checkpoint /home/johnsonchen/wall-panel-rl/results/diagnostics/construction_site_v29_training_seed20261117/training/models/ppo_32768.zip
```

This command runs in the foreground; keep that terminal open. It refuses to
start while the original pipeline still exists. Add `--check-only` to verify
recovery without starting evaluation or training.

The run's frozen settings and resolved site runtime are used. Current project
defaults do not replace the original scene seed blocks, teacher quota, height
limits or motion settings. Original source and case-bank hashes are checked.

The best fully evaluated candidate before reboot was `ppo_24576.zip`: 8/16
success, 1 collision and 7 timeouts with predictive control and local recovery.
The latest `ppo_32768.zip` is the continuation checkpoint, not a claim of a
better validated policy.
