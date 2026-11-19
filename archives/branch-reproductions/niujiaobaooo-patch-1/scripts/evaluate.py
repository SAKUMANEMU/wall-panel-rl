import _bootstrap
import argparse
import json
from pathlib import Path
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.utils.metrics import rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval_clean.yaml")
    p.add_argument("--clean-model", default="models/ppo_clean/best/best_model.zip")
    p.add_argument("--noise-model", default="models/ppo_noise/best/best_model.zip")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["rrt", "ppo_clean", "ppo_noise"],
        choices=["rrt", "ppo_clean", "ppo_noise"],
    )
    p.add_argument("--episodes", type=int)
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    config = load_config(args.config)
    episodes = args.episodes or config["evaluation"]["episodes"]
    output = Path("results/raw") / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    (output / "provenance.json").write_text(json.dumps(vars(args), indent=2))
    print(yaml.safe_dump(config), flush=True)
    torch.set_num_threads(1)
    rows = []
    scenarios = None
    for method in args.methods:
        model = (
            None
            if method == "rrt"
            else PPO.load(
                args.clean_model if method == "ppo_clean" else args.noise_model,
                device="cpu",
            )
        )
        env = WallPanelEnv(config)
        method_scenarios = []
        try:
            for episode in range(episodes):
                seed = config["evaluation"]["start_seed"] + episode
                result, trajectory = rollout(env, method, seed, model)
                result.update(
                    run_id=args.run_id,
                    method=method,
                    training_condition=method.removeprefix("ppo_")
                    if method != "rrt"
                    else "none",
                    evaluation_noise="noise"
                    if config["noise"]["position_std"]
                    or config["noise"]["orientation_std_deg"]
                    else "clean",
                )
                rows.append(result)
                method_scenarios.append(trajectory["scenario"])
                if episode < 3:
                    (output / f"{method}_trajectory_{seed}.json").write_text(
                        json.dumps(trajectory)
                    )
                pd.DataFrame(rows).to_csv(output / "episodes.csv", index=False)
                if (episode + 1) % 10 == 0:
                    print(
                        f'{method}: {episode+1}/{episodes}; success={sum(r["success"] for r in rows if r["method"]==method)/(episode+1):.1%}',
                        flush=True,
                    )
        finally:
            env.close()
        if scenarios is None:
            scenarios = method_scenarios
        else:
            assert scenarios == method_scenarios, "Methods received different scenarios"
    (output / "scenarios.json").write_text(json.dumps(scenarios, indent=2))
    print(f"Saved {len(rows)} episodes to {output}", flush=True)


if __name__ == "__main__":
    main()
