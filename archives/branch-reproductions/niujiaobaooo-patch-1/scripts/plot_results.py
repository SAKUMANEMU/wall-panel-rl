import _bootstrap
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "results/raw/main_clean/episodes.csv",
            "results/raw/main_noise/episodes.csv",
        ],
    )
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    summary = Path("results/summary")
    summary.mkdir(parents=True, exist_ok=True)
    metrics = [
        "success",
        "collision",
        "position_error_m",
        "orientation_error_deg",
        "steps",
        "joint_path_length_rad",
        "cartesian_path_length_m",
        "planning_time_s",
        "execution_time_s",
        "cumulative_reward",
    ]
    grouped = data.groupby(["method", "evaluation_noise"])
    table = grouped[metrics].agg(["mean", "std", "count"])
    table.columns = ["_".join(x) for x in table.columns]
    table.to_csv(summary / "metrics.csv")
    rates = grouped[["success", "collision", "planning_success", "ik_success"]].mean()
    rates["planning_failure"] = 1 - rates["planning_success"]
    rates["ik_failure"] = 1 - rates["ik_success"]
    (100 * rates).to_csv(summary / "rates_percent.csv")
    titles = [
        ("success", "Installation success (%)", 100),
        ("collision", "Collision (%)", 100),
        ("position_error_m", "Final position error (m)", 1),
        ("orientation_error_deg", "Final orientation error (deg)", 1),
        ("steps", "Control steps", 1),
        ("execution_time_s", "Execution wall time (s)", 1),
        ("planning_time_s", "IK + planning wall time (s)", 1),
        ("joint_path_length_rad", "Joint path length (rad)", 1),
    ]
    for metric, title, scale in titles:
        means = grouped[metric].mean().unstack() * scale
        fig, ax = plt.subplots(figsize=(7, 4))
        means.plot.bar(ax=ax, rot=0)
        ax.set_ylabel(title)
        ax.set_xlabel("Method")
        ax.set_title(title + " by observation condition")
        ax.legend(title="Observation")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for path in sorted(Path("models").glob("*/train.monitor.csv")):
        if path.parent.name not in {"ppo_clean", "ppo_noise"}:
            continue
        frame = pd.read_csv(path, comment="#")
        if len(frame):
            ax.plot(
                frame.l.cumsum(),
                frame.r.rolling(20, min_periods=1).mean(),
                label=path.parent.name,
            )
    ax.set(
        xlabel="Training control steps (within run)",
        ylabel="Episode return (20-episode mean)",
        title="Observed PPO training returns",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_reward.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for path in sorted(Path("models").glob("*/evaluations.npz")):
        if path.parent.name not in {"ppo_clean", "ppo_noise"}:
            continue
        archive = np.load(path)
        if "successes" in archive:
            ax.plot(
                archive["timesteps"],
                archive["successes"].mean(axis=1) * 100,
                label=path.parent.name,
            )
    ax.set(
        xlabel="Cumulative training steps",
        ylabel="Evaluation success (%)",
        title="PPO validation success (training reset mixture)",
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "training_success.png", dpi=180)
    plt.close(fig)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for path in Path(args.inputs[0]).parent.glob("*_trajectory_10000.json"):
        tr = json.loads(path.read_text())
        xyz = np.array(tr["panel_positions"])
        ax.plot(*xyz.T, label=path.stem.split("_trajectory")[0])
    if "tr" in locals():
        target = tr["scenario"]["target_position"]
        ax.scatter(*target, marker="*", s=100, label="Target")
        for obstacle in tr["scenario"]["obstacles"]:
            center = np.array(obstacle["center"])
            dims = np.array(obstacle["dimensions"])
            ax.bar3d(*(center - dims / 2), *dims, color="red", alpha=0.2)
    ax.set(
        xlabel="X (m)",
        ylabel="Y (m)",
        zlabel="Z (m)",
        title="Panel trajectories: evaluation seed 10000",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "trajectories.png", dpi=180)
    plt.close(fig)
    print(table.to_string())


if __name__ == "__main__":
    main()
