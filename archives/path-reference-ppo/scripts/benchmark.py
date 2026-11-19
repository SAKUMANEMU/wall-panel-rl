"""Run the primary six-cell experiment, then regenerate summaries and figures."""
import _bootstrap
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-model", default="models/ppo_clean/best/best_model.zip")
    parser.add_argument("--noise-model", default="models/ppo_noise/best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--prefix", default="main")
    args = parser.parse_args()
    paths = []
    for condition in ["clean", "noise"]:
        run = f"{args.prefix}_{condition}"
        subprocess.run(
            [
                sys.executable,
                "scripts/evaluate.py",
                "--config",
                f"configs/eval_{condition}.yaml",
                "--clean-model",
                args.clean_model,
                "--noise-model",
                args.noise_model,
                "--episodes",
                str(args.episodes),
                "--run-id",
                run,
            ],
            check=True,
        )
        paths.append(f"results/raw/{run}/episodes.csv")
    subprocess.run(
        [sys.executable, "scripts/plot_results.py", "--inputs", *paths], check=True
    )


if __name__ == "__main__":
    main()
