# Larger dynamic obstacles v27

All three obstacle dimensions use 1.25 times the v26 scale range:
`[0.85, 1.15]` becomes `[1.0625, 1.4375]`. Shape differences and seeded random
sizes remain. Typical overall edge lengths now range from 8.5 to 25.9 cm.
The 2--5 obstacle counts, motion amplitude/speed, height staggering and task
tolerances retain the v26 settings. Generation still requires real anchor-box
intersections with the physical guide and full-motion endpoint/spacing guards.
The geometry-only seed budget increases to 100 because five larger boxes are
harder to place. Candidate execution outcomes never select generation seeds.

Four verified scenes, one per obstacle count, are stored in:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/variable_weaving_large_v27_seed20261116/cases/
```

These are new scenes, not modifications of saved v26 recordings. Previous
training results and checkpoints retain their original meaning. A separate
larger-obstacle retraining pipeline is now generating its full scene banks,
then running BC warm-up, 65,536 PPO steps and independent evaluation.

The active run is stored at:

```text
/home/johnsonchen/wall-panel-rl/results/diagnostics/variable_weaving_large_v27_seed20261116/
  pipeline_status.json        # Generation/training/evaluation pipeline stage
  scenes/status.json          # Scene generation progress
  training/status.json        # Training progress, once generation completes
  generate_scenes.log
  train_and_evaluate.log
  training/models/
  training/evaluations/
```

Initialization and training hyperparameters are kept from v26. The training RNG
seed remains 20261115 for comparison; the output suffix identifies the v27 run.

Run the physical scene/interface checks:

```bash
cd /home/johnsonchen/wall-panel-rl/experiments/variable_weaving_large_v27
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  -m pytest -q tests/test_variable_weaving.py
```

To generate a full new bank for subsequent training:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python \
  scripts/generate_variable_cases.py --output outputs/large_scenes
```

Change `generation.size_scale` in `configs/variable_weaving.yaml` to adjust
physical dimensions. Existing scene JSON files and recordings do not change
when this parameter changes; regenerate scenes to apply a new scale.
