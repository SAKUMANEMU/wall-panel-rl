import _bootstrap
import argparse
from pathlib import Path
import json
import random
import hashlib
import importlib.metadata
import sys
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from wall_panel_rl.rl.callbacks import RewardTermsCallback
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_clean.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--zero-obstacle-inputs", action="store_true",
                        help="Neutralize previously unused obstacle columns during stage-A transfer")
    args = parser.parse_args()
    config = load_config(args.config)
    config["seed"] = args.seed
    if args.total_timesteps is not None:
        config["ppo"]["total_timesteps"] = args.total_timesteps
    print(yaml.safe_dump(config), flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    directory = Path("models") / args.run_name
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(yaml.safe_dump(config))
    provenance = dict(vars(args))
    provenance["python"] = sys.version
    provenance["packages"] = {name: importlib.metadata.version(name) for name in
                              ["pybullet", "gymnasium", "stable-baselines3", "torch", "numpy"]}
    if args.resume:
        provenance["source_model_sha256"] = hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
    env = Monitor(
        WallPanelEnv(config, "human" if args.render else None),
        str(directory / "train.monitor.csv"),
    )
    evaluation = Monitor(WallPanelEnv(config))
    evaluation.reset(seed=args.seed + 1000)
    pc = config["ppo"]
    eval_cb = EvalCallback(
        evaluation,
        best_model_save_path=str(directory / "best"),
        log_path=str(directory),
        eval_freq=pc["eval_freq"],
        n_eval_episodes=pc["eval_episodes"],
        deterministic=True,
    )
    checkpoint = CheckpointCallback(
        save_freq=pc["eval_freq"], save_path=str(directory / "checkpoints")
    )
    try:
        if args.resume:
            model = PPO.load(
                args.resume,
                env=env,
                device=args.device,
                seed=args.seed,
                tensorboard_log=str(directory / "tensorboard"),
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                seed=args.seed,
                device=args.device,
                verbose=1,
                tensorboard_log=str(directory / "tensorboard"),
                policy_kwargs={
                    "net_arch": pc["net_arch"],
                    "activation_fn": torch.nn.ReLU,
                    "log_std_init": pc["log_std_init"],
                },
                **{
                    k: pc[k]
                    for k in [
                        "learning_rate",
                        "n_steps",
                        "batch_size",
                        "n_epochs",
                        "gamma",
                        "gae_lambda",
                        "clip_range",
                        "ent_coef",
                    ]
                },
            )
        if args.zero_obstacle_inputs:
            if not args.resume:
                raise ValueError("--zero-obstacle-inputs is only for a no-obstacle checkpoint transfer")
            start = 2 * env.unwrapped.n + 6
            with torch.no_grad():
                model.policy.mlp_extractor.policy_net[0].weight[:, start:] = 0
                model.policy.mlp_extractor.value_net[0].weight[:, start:] = 0
        starting_timesteps = model.num_timesteps
        model.learn(
            total_timesteps=pc["total_timesteps"],
            callback=[eval_cb, checkpoint, RewardTermsCallback()],
            reset_num_timesteps=not bool(args.resume),
        )
        model.save(directory / "final_model")
        (directory / "training.json").write_text(
            json.dumps(
                {
                    "total_model_timesteps": model.num_timesteps,
                    "starting_timesteps": starting_timesteps,
                    "actual_additional_timesteps": model.num_timesteps - starting_timesteps,
                    "requested_additional_timesteps": pc["total_timesteps"],
                },
                indent=2,
            )
        )
    finally:
        env.close()
        evaluation.close()


if __name__ == "__main__":
    main()
