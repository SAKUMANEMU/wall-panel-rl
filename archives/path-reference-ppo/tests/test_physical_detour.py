import numpy as np
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.baselines.classical import ClassicalController


def test_robot_detours_and_executes():
    e = WallPanelEnv()
    try:
        e.reset(seed=42)
        center = [0.7549682259559631, 0.3600556254386902, 0.6588894724845886]
        e.obstacle_ids.append(e.box([0.07, 0.07, 0.07], center, [1, 0, 0, 1]))
        assert not e.collision.edge_valid(e.joint_state()[0], e.goal_q)
        controller = ClassicalController(e, 42)
        assert controller.planning_success and controller.planner.iterations > 0
        for _ in range(160):
            _, _, done, truncated, info = e.step(controller.action())
            if done or truncated:
                break
        assert info["success"] and not info["collision"]
    finally:
        e.close()
