import _bootstrap
import numpy as np
from stable_baselines3.common.env_checker import check_env
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.utils.config import load_config


def main():
    for stage in ["A", "B", "C"]:
        config = load_config()
        config["obstacles"]["stage"] = stage
        env = WallPanelEnv(config)
        try:
            check_env(env)
            obs, info = env.reset(seed=42)
            assert obs.shape == (41,) and not info["collision"]
            planner = ClassicalController(env, 42)
            assert planner.planning_success, f"{stage}: IK/RRT failed"
            for step in range(config["simulation"]["max_steps"]):
                obs, reward, done, truncated, info = env.step(planner.action(obs))
                if done or truncated:
                    break
            assert info["success"], (stage, info)
            print(
                f"Stage {stage}: SB3 API, collision checks, IK, RRT and physical installation PASS ({step+1} steps)"
            )
        finally:
            env.close()


if __name__ == "__main__":
    main()
