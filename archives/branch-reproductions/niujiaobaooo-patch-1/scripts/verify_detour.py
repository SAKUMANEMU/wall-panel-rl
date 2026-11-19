"""Find and save a physical-robot scenario whose straight joint path is blocked."""
import _bootstrap
import json
from pathlib import Path
import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.utils.config import load_config


def main():
    config = load_config()
    env = WallPanelEnv(config)
    try:
        env.reset(seed=42)
        start = env.joint_state()[0]
        for fraction in [0.35, 0.5, 0.65]:
            with env.collision.at_configuration(
                start * (1 - fraction) + env.goal_q * fraction
            ):
                position = np.array(env.panel_pose()[0])
            for delta in [
                [0, 0, 0],
                [0, 0, 0.10],
                [0, 0, -0.10],
                [0.08, 0, 0],
                [-0.08, 0, 0],
            ]:
                center = position + delta
                dims = np.array([0.07, 0.07, 0.07])
                body = env.box(dims, center, [0.8, 0.1, 0.1, 1])
                env.obstacle_ids.append(body)
                if (
                    env.collision.is_configuration_valid(start)
                    and env.collision.is_configuration_valid(env.goal_q)
                    and not env.collision.edge_valid(start, env.goal_q)
                ):
                    env.obstacles = [(center, dims)]
                    controller = ClassicalController(env, 42)
                    if controller.planning_success:
                        for step in range(config["simulation"]["max_steps"]):
                            _, _, done, trunc, info = env.step(controller.action())
                            if done or trunc:
                                break
                        if info["success"]:
                            result = {
                                "seed": 42,
                                "initial_q": start.tolist(),
                                "obstacle_center": center.tolist(),
                                "obstacle_dimensions": dims.tolist(),
                                "straight_path_valid": False,
                                "rrt_success": True,
                                "execution_success": True,
                                "steps": step + 1,
                                "rrt_iterations": controller.planner.iterations,
                                "planning_time_s": controller.planning_time,
                                "position_error_m": info["position_error"],
                                "orientation_error_deg": np.rad2deg(
                                    info["orientation_error"]
                                ),
                            }
                            Path("results/summary/detour.json").write_text(
                                json.dumps(result, indent=2)
                            )
                            print(result)
                            return
                        env.set_configuration(start)
                env.obstacle_ids.remove(body)
                p.removeBody(body, physicsClientId=env.client)
                env.obstacles = []
        raise RuntimeError("No successful detour found")
    finally:
        env.close()


if __name__ == "__main__":
    main()
