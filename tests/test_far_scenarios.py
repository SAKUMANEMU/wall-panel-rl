import numpy as np
import pybullet as p
import pytest
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.utils.config import load_config, ROOT


@pytest.mark.parametrize('name,expected_count', [('far_start',0), ('far_three_obstacles',3)])
def test_far_physical_installation(name, expected_count):
    e = WallPanelEnv(load_config(ROOT / f'configs/{name}.yaml'))
    try:
        obs, initial = e.reset(seed=10002)
        q = e.joint_state()[0]
        assert .8 <= initial['position_error'] <= 1.2
        assert len(e.obstacles) == expected_count
        assert obs.shape == (41,)
        controller = ClassicalController(e, 10002)
        assert controller.planning_success
        if expected_count:
            assert not e.collision.edge_valid(q, controller.goal_q)
            assert controller.planner.iterations > 0
            assert all(e.collision.edge_valid(a,b) for a,b in zip(controller.path[:-1], controller.path[1:]))
        for _ in range(e.config['simulation']['max_steps']):
            _,_,done,truncated,info = e.step(controller.action())
            if done or truncated:
                break
        assert info['success'] and not info['collision']
    finally:
        e.close()


def test_three_boxes_are_random_reproducible_and_separated():
    c = load_config(ROOT / 'configs/far_three_obstacles.yaml')
    e = WallPanelEnv(c)
    try:
        obs, _ = e.reset(seed=42)
        first = np.array(e.obstacles)
        assert len(first) == 3 and np.array_equal(obs[-21:].reshape(3,7)[:,-1], np.ones(3))
        e.reset(seed=10002)
        assert not np.array_equal(first, np.array(e.obstacles))
        e.reset(seed=42)
        assert np.array_equal(first, np.array(e.obstacles))
        for i, body in enumerate(e.obstacle_ids):
            for other in e.obstacle_ids[:i]:
                assert not p.getClosestPoints(body,other,.04,physicsClientId=e.client)
        centers=first[:,0]
        assert np.all(np.abs(centers-c['obstacles']['center']) <= np.array(c['obstacles']['randomization']))
    finally:
        e.close()


def test_impossible_distance_constraint_is_not_silently_ignored():
    c = load_config(ROOT / 'configs/far_start.yaml')
    c['robot']['start_position_distance_range'] = [3., 4.]
    e = WallPanelEnv(c)
    try:
        with pytest.raises(RuntimeError, match='initial configuration'):
            e.reset(seed=42)
    finally:
        e.close()
