# Learning Agent C&R submission

This folder is self-contained. It includes the learning-agent/environment source, a pinned Conda environment, a PPO checkpoint, separate datasets, the requested successful demo, measured experiment results, and an editable English report.

- **Report:** [REPORT.md](../REPORT.md)
- **Code guide:** [docs/CODE_MAP.md](../docs/CODE_MAP.md)
- **Measured results:** [experiments/metrics.csv](../experiments/metrics.csv)
- **Licences and attribution:** [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
- **Validation performed for this package:** [VALIDATION.md](../VALIDATION.md)

## 1. Environment installation

Run commands from this folder after extracting/copying it:

```bash
conda env create -f environment.yml
conda activate wall-panel-learning-agent-cr
```

The environment pins Python **3.11.5** and the actual installed Python-package versions, including Gymnasium 1.2.3, Stable-Baselines3 2.8.0, PyBullet 3.2.7, Torch 2.5.1+cpu, NumPy 1.24.3, and SciPy 1.11.1. `requirements-lock.txt` contains the complete pip freeze, including transitive/notebook dependencies from the working environment. The PyTorch CPU package index is included. No CUDA GPU, Gazebo server, or ROS installation is required.

Scripts add this folder's `src/` to the import path; no editable installation or original worktree is needed. For GUI playback, use a desktop with an X11/OpenGL display. All essential checks also run without a window using `--headless`. On a headless Linux server, Xvfb is an optional system tool for testing GUI display; it is not needed for DIRECT-mode checks.

## 2. Play the requested successful demo immediately

```bash
python scripts/replay_episode.py \
  demo/vertical_corridor_1166026_motion_excerpt.json \
  --fps 60 --playback-speed 0.6
```

This is the requested **motion excerpt**, with initial waiting omitted from display. It shows actual contiguous states from full-run steps 155–239. It passes below O1, above O2, and below O4. The complete real episode succeeds in 239 steps; the excerpt is not a shorter successful episode and is not a new policy execution.

For the full recording, including all waiting:

```bash
python scripts/replay_episode.py demo/vertical_corridor_1166026.json \
  --fps 60 --playback-speed 0.6
```

Playback uses saved joints, panel poses and moving-box positions, with display-only interpolation. It loads no policy and executes no physical steps. The final state stays visible for 30 seconds; change this with `--hold-seconds 5`.

Quick no-window playback validation:

```bash
python scripts/replay_episode.py demo/vertical_corridor_1166026_motion_excerpt.json --headless
```

## 3. Execute a new physical episode with the supplied model

```bash
python scripts/demo.py --mode enhanced
```

This recomputes the full fixed demonstration case, records it, then opens the playback window. Predictive control can take substantial CPU time before playback begins. It does not train. To record without a window:

```bash
python scripts/demo.py --mode enhanced --headless --output outputs/my_episode.json
python scripts/replay_episode.py outputs/my_episode.json
```

`--mode ppo` runs the actor without predictive execution/recovery; it can collide. `--mode rrt` selects the separate time-aware planner baseline. The enhanced mode has no global RRT fallback.

## 4. Test the learning interface

```bash
python -m pytest -q
python scripts/train_agent.py --steps 64 --output outputs/my_training_check
```

The second command performs **one short PPO rollout and update** on the four-case development training bank. It is a software check, not a convergence experiment. It saves a new model and summary under the chosen output directory and refuses to overwrite an existing run. The submitted `models/current_ppo.zip` is never modified. The included 64-scene results do not apply to this newly updated model.

A longer, still small continuation is available explicitly:

```bash
python scripts/train_agent.py --steps 2048 --output outputs/my_ppo_run
```

Settings are in `configs/training.yaml`. Only the multi-obstacle adapter and critic are trainable; the base policy is preserved. PPO collects its own sampled actions/log probabilities. Neither RRT nor the predictive recovery controller replaces training actions. Reward scaling is 0.01; the original environment reward and termination rules remain unchanged.

The four training cases are previously inspected development layouts, intended to make the interface runnable. They are not a sufficient training distribution for a strong final agent. Their seeds are disjoint from the 64-case validation bank and the selected demo.

## 5. Evaluate a model

Fast interface check; collision outcomes are legitimate algorithm outcomes:

```bash
python scripts/evaluate_agent.py --mode ppo --episodes 2 --workers 2
```

To reproduce the full enhanced-controller comparison on the supplied frozen bank:

```bash
python scripts/evaluate_agent.py --mode enhanced --episodes 64 --workers 4 --save-recordings
python scripts/evaluate_agent.py --mode ppo --episodes 64 --workers 4
```

The enhanced run is compute-intensive because each real action can require many hypothetical physics rollouts. The measured results are already supplied in `experiments/`; rerunning is optional. New outputs go to `outputs/`, never over the frozen results. To assess a newly trained model, add `--model outputs/my_ppo_run/model.zip`.

Optional `--mode rrt` and `--mode mixture` are included. Mixture draws 50% RRT / 50% enhanced PPO **once per episode** before execution; it never switches a failed episode to a different controller. These optional modes are not part of the v20 result table.

## 6. Generate additional training cases

```bash
python scripts/generate_cases.py --count 4 --start-seed 1209000 \
  --output outputs/new_train_cases.json
python scripts/train_agent.py --bank outputs/new_train_cases.json \
  --steps 2048 --output outputs/new_data_ppo_run
```

Generation can be slow. It uses a preliminary physical route and full-motion geometry guards; it does not select final scenes based on successful final execution. Construction attempts, including rejections, are saved. Existing training/validation/demo seeds are excluded. The trainer rejects overlap with the fixed validation/demo seeds.

## 7. What the current results establish

The latest **64 new valid-layout** study reports:

| Mode | Success | Collision | Timeout | Installation success |
|---|---:|---:|---:|---:|
| PPO + prediction + local recovery | 43 | 1 | 20 | **67.2%** |

Successful installation **with** the specified measured vertical-passage pattern occurred in **9/64 (14.1%)** enhanced episodes. The enhanced success-rate Wilson 95% interval is approximately 55.0–77.4%. These are frozen validation results, not training/convergence claims or a final independent test. Model and controller parameters were unchanged during that study. Older 98.4% results used a different layout and are not the current result.

The successful demo was selected separately and is excluded from the 64-case bank. Its complete and excerpt recordings are examples, not success-rate samples. Historical results and datasets are read-only inputs to the supported scripts. All instructions, comments, and report text in this package are English.
