import _bootstrap
import argparse
import time
import json
import secrets
from pathlib import Path
import numpy as np
import pybullet as p
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["rrt", "ppo"], default="rrt")
    parser.add_argument("--model")
    parser.add_argument("--config", default="configs/far_five_dynamic.yaml")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, help="Fixed seed for reproducible replay; otherwise choose a fresh seed")
    seed_options.add_argument("--random-seed", action="store_true",
                              help="Choose a fresh scenario seed and print it for replay")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--playback-speed", type=float,
                        help="GUI playback multiplier; 0.6 plays at 60%% speed")
    parser.add_argument("--hold-seconds", type=float, default=5)
    parser.add_argument("--start-hold-seconds", type=float, default=2,
                        help="Pause the GUI after RRT planning so the initial scene is visible")
    parser.add_argument("--snapshot", default="results/figures/demo.png")
    args = parser.parse_args()
    if args.method == "ppo" and not args.model:
        parser.error("--model is required for PPO")
    if args.random_seed or args.seed is None:
        args.seed = secrets.randbelow(2**31)
    print(f"Scenario seed: {args.seed}", flush=True)
    config = load_config(args.config)
    print(f"Target observation noise: position std={config['noise']['position_std']} m; "
          f"orientation std={config['noise']['orientation_std_deg']} deg", flush=True)
    playback_speed = args.playback_speed if args.playback_speed is not None else config.get("demo", {}).get("playback_speed", 1.0)
    if not np.isfinite(playback_speed) or playback_speed <= 0:
        parser.error("--playback-speed must be a finite positive number")
    print(f"GUI playback speed: {playback_speed:g}x", flush=True)
    print(f"Dynamic planning mode: {config['rrt'].get('dynamic_mode', 'swept')}", flush=True)
    env = WallPanelEnv(config, "rgb_array" if args.headless else "human")
    model = PPO.load(args.model, device="cpu") if args.method == "ppo" else None
    if model is not None and model.observation_space.shape != env.observation_space.shape:
        env.close()
        parser.error("Model observation size differs from this environment; five obstacles require a newly trained/adapted model.")
    previous = []
    last_frame_time = None

    def display(e, step, info):
        nonlocal last_frame_time
        pos = e.panel_pose()[0]
        if previous:
            p.addUserDebugLine(
                previous[-1], pos, [0, 0.4, 1], 2, physicsClientId=e.client
            )
        previous.append(pos)
        print(
            f'step={step} position={info["position_error"]:.4f} m orientation={np.rad2deg(info["orientation_error"]):.2f} deg collision={info["collision"]} success={info["success"]}'
        )
        if not args.headless:
            # Pace wall-clock display, accounting for computation/rendering time.
            # Bullet still advances the same number of fixed-size physics steps.
            frame_duration = config["simulation"]["timestep"] * config["simulation"]["substeps"] / playback_speed
            now = time.perf_counter()
            elapsed = 0.0 if last_frame_time is None else now - last_frame_time
            time.sleep(max(0.0, frame_duration - elapsed))
            last_frame_time = time.perf_counter()

    def preview_scene(e, controller):
        nonlocal last_frame_time
        if not args.headless and e.reference_positions:
            for a, b in zip(e.reference_positions[:-1], e.reference_positions[1:]):
                p.addUserDebugLine(a, b, [0.85, 0.7, 0.1], 1, physicsClientId=e.client)
            print("Yellow line: obstacle-free RRT panel reference path; blue: executed path.", flush=True)
        if not args.headless:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=e.client)
        print(f"Initial distance: {e.info()['position_error']:.3f} m; obstacles: {len(e.obstacles)}", flush=True)
        if not args.headless and args.start_hold_seconds > 0:
            time.sleep(args.start_hold_seconds)
        last_frame_time = time.perf_counter()

    try:
        if not args.headless and args.method == "rrt":
            print("Planning path; display resumes when planning finishes...", flush=True)
            # Reset, IK and collision queries temporarily move the robot. Hide
            # these queries until on_plan restores the initial scene display.
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=env.client)
        result, _ = rollout(env, args.method, args.seed, model, display, on_plan=preview_scene)
        print(json.dumps(result, indent=2))
        env.render_mode = "rgb_array"
        from PIL import Image

        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render()).save(args.snapshot)
        if not args.headless:
            time.sleep(args.hold_seconds)
    finally:
        if not args.headless and p.isConnected(env.client):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=env.client)
        env.close()


if __name__ == "__main__":
    main()
