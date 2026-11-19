# Submission validation

Validated on **6 November 2026**, on Linux with the Python 3.11.5 environment whose installed versions are pinned in `environment.yml` and `requirements-lock.txt`.

## Checks performed

| Check | Observed result |
|---|---|
| Automated tests | **16 passed**, both in the submission folder and in a separately relocated copy under `/tmp` |
| Gymnasium/SB3 interface | `check_env` passed; finite 77-value observations, seven-value actions, rewards and valid reset/step signatures |
| Dependency consistency | `pip check`: **No broken requirements found** |
| Custom policy loading after relocation | Checkpoint loaded from local packaged modules; resolved project root and policy source were inside the relocated folder |
| PPO continuation smoke check | **64 additional steps**, three PPO optimizer epochs, finite parameters, and a new output checkpoint; supplied checkpoint unchanged |
| Multiprocess evaluation smoke check | Two PPO-only episodes completed: zero successes, two collisions; all outcomes retained |
| Additional-scene generation | One valid new scene at seed 1209003 after three recorded geometry rejections |
| Enhanced physical demo reproduction | Success at step **239**; all **240 recorded physical frames exactly match** the supplied complete demo |
| Independent RRT execution check | Selected demo case completed successfully in 73 steps; this is a software check, not an RRT success-rate estimate |
| GUI excerpt replay after relocation | **421/421 frames**, zero skipped frames, approximately 7.004 seconds at 60 FPS and 0.6× speed |
| Replay independence | Zero controller calls and zero physics steps during GUI playback |
| Dataset separation | Training, validation and selected-demo seed sets are disjoint |

The GUI check used Xvfb with Mesa software OpenGL. PyBullet printed debug-drawing warnings under that virtual display; pose playback completed. A normal desktop display can be used directly with the README commands.

Machine-readable evidence is supplied in `validation_evidence/`. The short training log starts fresh monitoring buffers instead of carrying historical rolling statistics from the loaded checkpoint. The logged short-run success rate is not evidence of convergence. `additional_steps` in `training_summary.json` records the actual CLI budget; its nested `settings.steps` retains the YAML default that the CLI overrides.

## Scope and reproducibility limits

The full enhanced demonstration was physically recomputed with the packaged code. The frozen 64-scene comparison in `experiments/` was copied from the existing completed study, checked against its summaries, and **was not rerun in full during packaging**. The 64-step software-check checkpoint is separate from the submitted model and is not used for that result table. The new training and evaluation entry points never overwrite the bundled model or historical results.

The package was tested with the existing interpreter and its exact installed dependency versions, including execution after relocation without the original worktree on the import path. A fresh network-based `conda env create` was **not** performed during packaging. Conda/pip installation therefore also depends on package-index availability and the target system. Linux CPU is the validated platform; a desktop X11/OpenGL display is required only for GUI playback.

This submission provides runnable learning/environment interfaces and documented experimental limitations. It makes no claim that the current PPO actor has converged on the latest obstacle distribution.

## Integrity

`MANIFEST.sha256` lists packaged-file SHA-256 digests, excluding the manifest itself. From this folder on Linux, run:

```bash
sha256sum -c MANIFEST.sha256
```

The original supplied model SHA-256 is:

```text
f059554915581537e35b57f7226346577ea4bfbc99cb1db0946b75ae2c539aba
```
