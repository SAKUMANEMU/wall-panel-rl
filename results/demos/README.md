# PPO 3D demonstration

`waypoint_ppo_3d.mp4` is an actual PyBullet rollout using the selected 5,000-step PPO checkpoint after BC initialization. RRT plans the route; PPO tracks the next joint waypoint. The five obstacles are static and target observations are noise-free. Seed 81009 completes in 56 control steps (2.80 simulated seconds), with final position error 3.33 mm and orientation error 0.094 degrees.

The motion plays at 0.6x, with introductory/final holds and a paused final orbit. Total video duration: 11.67 s, 1280x800, 30 fps, H.264. Blue dots show the executed panel trajectory; yellow dots show the RRT route. Trails and ground have no collision geometry. The JSON sidecar stores the model hash and measured result. This is one demonstration episode, not a success-rate estimate.

Files: `waypoint_ppo_3d.mp4`, `waypoint_ppo_3d.png` (preview), `waypoint_ppo_3d_final.png`, `waypoint_ppo_3d.json`.

To open a live 3D GUI and record a new video, run from any directory:

```bash
env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo/scripts/demo_waypoint_3d.py --gui --output /home/johnsonchen/wall-panel-rl/results/demos/waypoint_ppo_3d_gui.mp4
```

The GUI command requires a graphical desktop and refuses an existing output filename. Without `--gui`, rendering is headless. No training is performed.
