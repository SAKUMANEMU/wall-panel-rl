# Staggered dynamic-only experiment

The v11 experiment uses four unequal moving obstacles. All training, BC teacher
and evaluation episodes are dynamic. Maximum Euclidean motion offset is0.18m
and speed is0.12m/s, both20% above v10. Anchor heights receive alternating
positive/negative3-7cm offsets relative to their on-path witness positions,
with at least15cm total height span and4cm adjacent height separation.
Endpoint/wall/spacing guards, target, reward, action space and success criteria
remain unchanged. Complete obstacle-free references are used only for scene
construction. They are not BC training episodes.

Source: the retained v9 fast02 model, not the v10 experimental checkpoint.
The source PPO optimizer is retained. The old mixed static/dynamic BC label pool
is discarded; a new dynamic-only teacher pool and fresh BC optimizers are used.
BC changes only the four-obstacle adapter; the critic, base policy and action
standard deviation are frozen during BC. PPO may subsequently train its critic.

Stages:

1. Freeze192 train,64 validation and100 reserved test layouts. Exclude all prior
   validation/test scene IDs, including both used v10 banks.
2. Diagnose every TRAIN layout under the original source. Record early/middle/
   terminal failures, errors and collision body/link evidence. Use only TRAIN
   failures for teacher targeting and sampling probabilities.
3. Seek48 verified dynamic RRT episodes within120 attempts, with random phase
   variants (at most3 per layout). Require timed pre/post collision checks and
   independent physical action replay before retaining a label.
4. Run200 bounded actor BC updates; validate at0,25,50,100,200. Oversample the
   first8 steps with probability0.5 while retaining full-route samples. Reject
   updates exceeding sampled step/source KL bounds. Select by dynamic success,
   collision non-increase and zero joint-limit violations; keep earlier ties.
5. Continue5,000 PPO transitions in episode-level50/50 RRT/PPO mode. All scene
   probabilities are dynamic. Requested curriculum:0.15m/0.10m/s,
   0.165m/0.11m/s,0.18m/0.12m/s, with dynamic-only validation gates. Every
   checkpoint is evaluated at the full0.18m/0.12m/s target. Live episodes finish
   under their existing parameters before a new curriculum level is applied.
6. Lock the final selection before a single original-source versus final-model
   reserved test. Never use this test for labels or subsequent tuning.

From `/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo`, the staged
entry points are:

```bash
env -u PYTHONPATH ../../.conda-env/bin/python scripts/prepare_four_dynamic_training.py --config configs/four_staggered_dynamic_preparation.yaml --prepare
env -u PYTHONPATH ../../.conda-env/bin/python scripts/run_staggered_dynamic_warmup.py --run
env -u PYTHONPATH ../../.conda-env/bin/python scripts/train_four_dynamic_ppo.py --config configs/four_staggered_dynamic_training.yaml --train
env -u PYTHONPATH ../../.conda-env/bin/python scripts/summarize_staggered_dynamic.py
```

These commands refuse to overwrite existing experiment directories. Run later
stages only after the preceding stage completes successfully. Old experiment
configs remain available. The existing enhanced demo command now uses the new
staggered scene generator; the15cm profile is retained in
`configs/four_dynamic_faster_15cm.yaml`. Its default policy remains the retained
source until an updated checkpoint has supporting evidence.

```bash
env -u PYTHONPATH ../../.conda-env/bin/python scripts/demo_four_dynamic_faster.py --controller ppo
```
