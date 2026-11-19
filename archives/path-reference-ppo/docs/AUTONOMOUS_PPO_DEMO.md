# Current autonomous PPO animation

As of2026-11-06, demos first record the entire episode without a GUI and then
replay at60 FPS /0.6x. Planning cannot block the replay; real controller waits
remain visible. Recordings include configuration, scene, real joint/panel/box
states and the unchanged final result. See [current commands](RECORDED_PLAYBACK.md).
Use `--playback-mode live` to retain the former live-computation display.

## Historical static scene (2026-11-03)

The2026-11-03 default was the retained initialization with **four
static obstacles intersecting its obstacle-free physical trajectory**, a fresh
random start/layout and0.6x playback:

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py
```

Add `--seed 20261103` for a fixed scene or `--headless` without a GUI. A frozen
scene bank can be selected explicitly with `--scene-bank ... --seed ...`.
Fresh-scene preparation computes an obstacle-free reference before rendering;
pure policy execution has no online avoidance planner. The cyan line is the
initialized model's actual obstacle-free motion, computed before playback;
blue is current execution. Each box has its own verified contact witness on the
cyan route, while initial/goal poses remain clear. Source weights are unchanged,
with no retraining. [Results and full details](../results/diagnostics/four_static_on_path_v1_seed61/README.md).

## Previous default (2026-11-02)

As of the completed mixed training on 2026-11-02, the default is the selected
5k PPO checkpoint in `results/diagnostics/episode_rrt_ppo_v1_seed61`: 18/20
validation successes, zero observed collisions and two timeouts. The corresponding
50/50 mixture scores 19/20, zero collisions and one timeout. For its demo,
see [mixture instructions](EPISODE_RRT_PPO.md).

The original notes below describe the 2026-10-30 release. To replay it exactly,
pass both its model and manifest; the current default has since changed.

Run from the working copy using the existing project Python environment:

```bash
cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/demo_autonomous_ppo.py \
  --seed 1000000 --playback-speed 0.6 --hold-seconds 30
```

The default model is the retained PPO in the visible conditional-PPO-adaptation
report (`results/diagnostics/conditional_ppo_adaptation_v3_seed61` in the main
checkout). It retains the original weights (selected additional step 0) together
with the validated conditional recovery controller. The 10k updated candidate
was rejected by the non-regression selector.
This exact seed succeeds in 65 steps with 2.45mm / .28 degree final errors.
The default bank is `data/clearance_ppo_v3_seed61/independent_cases.json`, containing
previously frozen one-static scenes; it is not a fresh random training rollout.
Noise is zero and actions are deterministic. Closing/interruption ends the demo;
the final state normally stays visible for 30 seconds. Playback speed affects
wall-clock display only, not physics steps or control frequency.

To replay a case that actually needs conditional recovery:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python -u scripts/demo_autonomous_ppo.py \
  --scene-bank data/staged_avoidance_v3_seed61/one_static/validation_staged.json \
  --seed 900002 --playback-speed 0.6 --hold-seconds 30
```

To compare this adaptation's final unselected candidate with the same controller,
append:

```text
--model /home/johnsonchen/wall-panel-rl/results/diagnostics/conditional_ppo_adaptation_v3_seed61/final_candidate.zip
```

The demo selects ConditionalRecoveryEnv from the current manifest, retaining
StagedAvoidanceEnv's evaluated reset/cursor/action projection/step underneath,
replacing only its Bullet rendering connection for GUI mode. It reads the saved
obstacle-free reference from the frozen scene, never a fresh collision-free expert
route. A regression test forbids ClassicalController and RRT planning at runtime
and confirms the retained actor's 65-step success. `--headless` runs inference-only
verification without opening a GUI. It does not train or change model files.

For a historical baseline comparison, pass both its `--model` and `--manifest`
from `results/diagnostics/static_stability_v3_seed61` in the main checkout; that
manifest intentionally selects the old controller without conditional recovery.
