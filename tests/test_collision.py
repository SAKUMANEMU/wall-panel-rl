import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv


def test_collision_and_query_restoration():
    e = WallPanelEnv()
    try:
        e.reset(seed=42)
        assert e.collision.is_configuration_valid(e.goal_q)
        q, dq = e.joint_state()
        pose = e.panel_pose()
        e.collision.is_configuration_valid(e.goal_q)
        assert np.allclose(e.joint_state()[0], q) and np.allclose(
            e.joint_state()[1], dq
        )
        assert np.allclose(e.panel_pose()[0], pose[0])
        body = e.box([0.1, 0.1, 0.1], pose[0], [1, 0, 0, 1])
        e.obstacle_ids.append(body)
        assert e.collision.is_collision()
        assert not e.collision.is_configuration_valid(q)
        e.obstacle_ids.remove(body)
        p.removeBody(body, physicsClientId=e.client)
        assert not e.collision.is_collision()
    finally:
        e.close()


def test_robot_obstacle_collision_independent_of_panel():
    e = WallPanelEnv()
    try:
        e.reset(seed=42)
        arm_position = p.getLinkState(e.robot, 3, physicsClientId=e.client)[0]
        obstacle = e.box([.05, .05, .05], arm_position, [1, 0, 0, 1])
        e.obstacle_ids.append(obstacle)
        assert not p.getClosestPoints(e.panel, obstacle, 0, physicsClientId=e.client)
        assert e.collision.is_collision()
    finally:
        e.close()
