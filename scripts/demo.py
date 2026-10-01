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
    parser.add_argument("--config", default="configs/eval_clean.yaml")
    seed_options = parser.add_mutually_exclusive_group()
    seed_options.add_argument("--seed", type=int, default=10002)
    seed_options.add_argument("--random-seed", action="store_true",
                              help="Choose a fresh scenario seed and print it for replay")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hold-seconds", type=float, default=5)
    parser.add_argument("--start-hold-seconds", type=float, default=2,
                        help="Pause the GUI after RRT planning so the initial scene is visible")
    parser.add_argument("--snapshot", default="results/figures/demo.png")
    args = parser.parse_args()
    if args.method == "ppo" and not args.model:
        parser.error("--model is required for PPO")
    if args.random_seed:
        args.seed = secrets.randbelow(2**31)
    print(f"Scenario seed: {args.seed}", flush=True)
    config = load_config(args.config)
    env = WallPanelEnv(config, "rgb_array" if args.headless else "human")
    model = PPO.load(args.model, device="cpu") if args.method == "ppo" else None
    previous = []

    def display(e, step, info):
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
            time.sleep(
                config["simulation"]["timestep"] * config["simulation"]["substeps"]
            )

    def preview_scene(e, controller):
        print(f"Initial distance: {e.info()['position_error']:.3f} m; obstacles: {len(e.obstacles)}", flush=True)
        if not args.headless and args.start_hold_seconds > 0:
            time.sleep(args.start_hold_seconds)

    try:
        result, _ = rollout(env, args.method, args.seed, model, display, on_plan=preview_scene)
        print(json.dumps(result, indent=2))
        env.render_mode = "rgb_array"
        from PIL import Image

        Path(args.snapshot).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render()).save(args.snapshot)
        if not args.headless:
            time.sleep(args.hold_seconds)
    finally:
        env.close()


if __name__ == "__main__":
    main()
