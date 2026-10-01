"""Quantify error reduction without confusing return with installation success."""
import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    env = WallPanelEnv(load_config(args.config))
    rows = []
    try:
        for seed in range(10000, 10000 + args.episodes):
            obs, initial = env.reset(seed=seed)
            for _ in range(env.config["simulation"]["max_steps"]):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, truncated, final = env.step(action)
                if done or truncated:
                    break
            rows.append(
                {
                    "seed": seed,
                    "initial_position_error_m": initial["position_error"],
                    "final_position_error_m": final["position_error"],
                    "initial_orientation_error_deg": float(
                        np.rad2deg(initial["orientation_error"])
                    ),
                    "final_orientation_error_deg": float(
                        np.rad2deg(final["orientation_error"])
                    ),
                    "success": final["success"],
                    "collision": final["collision"],
                }
            )
    finally:
        env.close()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(rows, indent=2))
    print(
        "Mean initial/final position error:",
        np.mean([r["initial_position_error_m"] for r in rows]),
        np.mean([r["final_position_error_m"] for r in rows]),
    )
    print("Success rate:", np.mean([r["success"] for r in rows]))


if __name__ == "__main__":
    main()
