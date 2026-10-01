import time
import numpy as np
from wall_panel_rl.planning.ik import solve_goal
from wall_panel_rl.planning.rrt_connect import RRTConnect, interpolate


class ClassicalController:
    """Plan once from the first observed target; follow through common actuator."""

    def __init__(self, env, seed=0):
        self.env = env
        start = time.perf_counter()
        goal = solve_goal(env, env.observed_target, seed)
        self.ik_success = goal is not None
        self.path = None
        self.planner = RRTConnect(
            env.lower,
            env.upper,
            env.collision.is_configuration_valid,
            env.collision.edge_valid,
            env.config["rrt"],
            seed,
        )
        if self.ik_success:
            self.path = self.planner.plan(env.joint_state()[0], goal)
        self.planning_time = time.perf_counter() - start
        self.planning_success = self.path is not None
        self.index = 1
        if self.planning_success:
            dense = [self.path[0]]
            for a, b in zip(self.path[:-1], self.path[1:]):
                dense.extend(interpolate(a, b, env.max_increment)[1:])
            self.path = dense

    def action(self, observation=None):
        if not self.planning_success:
            return np.zeros(self.env.n)
        q, _ = self.env.joint_state()
        while (
            self.index < len(self.path) - 1
            and np.linalg.norm(self.path[self.index] - q)
            < self.env.config["rrt"]["waypoint_tolerance"]
        ):
            self.index += 1
        return np.clip((self.path[self.index] - q) / self.env.max_increment, -1, 1)
