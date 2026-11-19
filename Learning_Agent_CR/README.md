# Learning Agent C&R: Wall-Panel Installation

This submission connects a robot simulator, joint-space policy and PPO training loop. The [two-page report](report/ME5418_Learning_Agent_CR_Report.pdf) explains the integration; [REPORT.md](REPORT.md) is editable. Run all commands from this folder.

## 1. Install

```bash
conda env create -f environment.yml
conda activate wall-panel-learning-agent-cr
```

`environment.yml` pins Python 3.11.5 and dependencies. No ROS, Gazebo or GPU is required. GUI playback requires a desktop display; use `--headless` without one. In a ROS-sourced shell, prefix Python commands with `env -u PYTHONPATH` if imports conflict.

## 2. How the components connect

```text
case bank + settings -> reset -> observation -> policy -> physical step
                                  ^                         |
                                  +--- reward / next state -+
training: rollout buffer -> PPO update -> new checkpoint
execution: checkpoint -> optional predictive control -> recording -> replay
```

`submission.py` connects `CaseBankEnv` to the saved policy. Environment wrappers supply the local reference and reward feedback. `DummyVecEnv` connects this interface to SB3, which collects policy-generated transitions and updates the obstacle adapter and critic. Training uses separate cases from validation and demonstration.

Earlier BC used RRT demonstrations; its optional helpers remain in `rl/episode_imitation.py`. The submitted trainer continues PPO only. Enhanced execution adds prediction and local recovery without updating weights. Replay displays saved physical states independently of planning.

## 3. Run the workflow

The supplied cases and checkpoint are ready to use; scene generation is optional. First check the environment and perform a short PPO update:

```bash
python -m pytest -q
python scripts/train_agent.py --steps 64 --output outputs/check_01
```

This checks the rollout/update interface, not convergence. It writes `model.zip`, `progress.csv` and `training_summary.json` to the chosen directory. Use a new directory for each run.

Evaluate that new checkpoint on separate validation cases:

```bash
python scripts/evaluate_agent.py --mode ppo --episodes 2 --workers 2 \
  --model outputs/check_01/model.zip
```

Evaluation writes outcomes under `outputs/`. Use `--mode enhanced` to add predictive execution and `--save-recordings` to save trajectories. Each script provides `--help`.

For a visual demonstration using the supplied checkpoint:

```bash
python scripts/demo.py --mode enhanced
```

The complete episode is computed and recorded before playback opens. Add `--headless` to record only. To immediately play the supplied motion excerpt:

```bash
python scripts/replay_episode.py \
  demo/vertical_corridor_1166026_motion_excerpt.json \
  --fps 60 --playback-speed 0.6
```

The excerpt omits initial waiting. Use `demo/vertical_corridor_1166026.json` for the full recording. Playback performs no training or new physics simulation.

## 4. Where to read or modify

| Location | Purpose |
|---|---|
| `src/wall_panel_rl/submission.py` | Connect cases, environment and policy |
| `src/wall_panel_rl/envs/`, `control/` | Task dynamics, reward and motor commands |
| `src/wall_panel_rl/rl/`, `planning/` | Policy, reference handling, BC helpers and planning |
| `configs/runtime.json` | Physics, reward and execution-controller settings |
| `configs/training.yaml` | PPO budget and optimizer settings |
| `configs/scene.yaml`, `data/` | New-scene rules and supplied case banks |
| `models/`, `demo/` | Checkpoint and recorded demonstrations |

Changing scene rules affects newly generated cases, not existing case banks. See the [run guide](docs/RUN_GUIDE.md) for generation commands, [code guide](docs/CODE_MAP.md) for module details, and [attribution](THIRD_PARTY_NOTICES.md) for external libraries and assets. Software checks are documented in [VALIDATION.md](VALIDATION.md).
