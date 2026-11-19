import numpy as np
from wall_panel_rl.planning.rrt_connect import RRTConnect, interpolate
from wall_panel_rl.utils.config import load_config


def test_rrt_detour():
    def valid(q):
        return not (0.4 <= q[0] <= 0.6 and 0.1 <= q[1] <= 0.8)

    def edge(a, b):
        return all(valid(q) for q in interpolate(a, b, 0.01))

    start, goal = np.array([0.1, 0.5]), np.array([0.9, 0.5])
    assert not edge(start, goal)
    planner = RRTConnect([0, 0], [1, 1], valid, edge, load_config()["rrt"], 42)
    path = planner.plan(start, goal)
    assert path is not None and len(path) > 2
    assert np.allclose(path[0], start) and np.allclose(path[-1], goal)
    assert all(edge(a, b) for a, b in zip(path[:-1], path[1:]))
