import numpy as np
import pybullet as p
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config, ROOT
from wall_panel_rl.baselines.classical import ClassicalController


def make_env():
    return WallPanelEnv(load_config(ROOT / 'configs/far_five_dynamic.yaml'))


def test_motion_spacing_speed_reproducibility_and_query_restore():
    e = make_env()
    try:
        obs, _ = e.reset(seed=10002)
        assert obs.shape == (55,)
        initial = np.array(e.obstacles)
        specs = e.obstacle_motion
        start = np.array(e.panel_pose()[0])
        direction = e.target_pose[0] - start
        fractions = [(s['anchor'] - start) @ direction / (direction @ direction) for s in specs]
        assert np.allclose(fractions, np.arange(1, 6) / 6, atol=.005)
        for i,body in enumerate(e.obstacle_ids):
            for other in e.obstacle_ids[:i]:
                assert not p.getClosestPoints(body, other, .10, physicsClientId=e.client)
        for t in np.linspace(0, 15, 31):
            e.move_obstacles(t)
            a = np.array([c.copy() for c,d in e.obstacles])
            e.move_obstacles(t + .001)
            b = np.array([c.copy() for c,d in e.obstacles])
            assert np.all(np.linalg.norm(b-a,axis=1) <= .00005001)
            assert all(np.linalg.norm(c-s['anchor']) <= .100001 for (c,d),s in zip(e.obstacles,specs))
        e.move_obstacles(0)
        ids = e.obstacle_ids.copy()
        try:
            with e.planning_obstacles():
                raise RuntimeError('query error')
        except RuntimeError:
            pass
        assert e.obstacle_ids == ids
        assert np.array_equal(initial, np.array(e.obstacles))
        e.reset(seed=10002)
        assert np.array_equal(initial,np.array(e.obstacles))
        e.reset(seed=10003)
        assert not np.array_equal(initial,np.array(e.obstacles))
    finally:
        e.close()


def test_dynamic_physical_execution_and_replanning():
    e = make_env()
    try:
        e.reset(seed=10002)
        c = ClassicalController(e,10002)
        assert c.planning_success
        # An obsolete route through the middle of the workspace must trigger
        # the stop/replan branch; the replacement is then physically executed.
        q = e.joint_state()[0]
        c.path = [q, np.zeros(e.n), c.goal_q]
        c.index = 1
        a = c.action()
        assert c.replans == 1 and np.array_equal(a,np.zeros(e.n))
        for _ in range(e.config['simulation']['max_steps']):
            _,_,done,truncated,info=e.step(a)
            if done or truncated:
                break
            a=c.action()
        assert info['success'] and not info['collision']
    finally:
        e.close()
