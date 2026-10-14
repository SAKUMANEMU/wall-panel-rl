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
        points = np.array(e.reference_positions)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0),axis=1))]
        for i, spec in enumerate(specs):
            expected = [np.interp((i+1)/6 * arc[-1], arc, points[:,k]) for k in range(3)]
            assert np.allclose(spec['reference'], expected)
            assert np.linalg.norm(spec['anchor']-spec['reference']) < .61
        assert all(e.collision.within_limits(q) for q in e.reference_path)
        for i,body in enumerate(e.obstacle_ids):
            for other in e.obstacle_ids[:i]:
                assert not p.getClosestPoints(body, other, .10, physicsClientId=e.client)
        samples = []
        for t in np.linspace(0, 30, 61):
            e.move_obstacles(t)
            a = np.array([c.copy() for c,d in e.obstacles])
            samples.append(a)
            e.move_obstacles(t + .001)
            b = np.array([c.copy() for c,d in e.obstacles])
            assert np.all(np.linalg.norm(b-a,axis=1) <= e.config['obstacles']['dynamic']['max_speed'] * .001 + 1e-8)
            assert all(np.linalg.norm(c-s['anchor']) <= e.config['obstacles']['dynamic']['amplitude'] + 1e-8 for (c,d),s in zip(e.obstacles,specs))
        assert np.all(np.ptp(np.array(samples), axis=0) > .1)
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
        rng = np.random.default_rng(123)
        unsafe = next(candidate for candidate in rng.uniform(e.lower,e.upper,(1000,e.n))
                      if not e.collision.is_configuration_valid(candidate))
        c.path = [q, unsafe, c.goal_q]
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
