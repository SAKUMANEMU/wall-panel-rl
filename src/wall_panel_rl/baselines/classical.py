import time
import numpy as np
import pybullet as p
from wall_panel_rl.planning.ik import solve_goal
from wall_panel_rl.planning.rrt_connect import RRTConnect, interpolate


class ClassicalController:
    """Plan once from the first observed target; follow through common actuator."""

    def __init__(self, env, seed=0):
        self.env = env
        self.replans = 0
        self.replan_failures = 0
        self.hold_steps = 0
        self.next_retry = 0
        start = time.perf_counter()
        with env.planning_obstacles():
            goal = solve_goal(env, env.observed_target, seed)
        self.goal_q = goal
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
            with env.planning_obstacles():
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
        if self.env.obstacle_motion:
            # Kinematic queries must not appear as rapid movements in the GUI.
            human = self.env.render_mode == 'human'
            if human:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.env.client)
            try:
                safe = self.upcoming_path_safe(q)
                if not safe:
                    self.env.controller.command(q)
                    if self.env.steps >= self.next_retry:
                        begin = time.perf_counter()
                        with self.env.planning_obstacles():
                            path = self.planner.plan(q, self.goal_q)
                        self.planning_time += time.perf_counter() - begin
                        self.replans += 1
                        self.next_retry = self.env.steps + self.env.config['rrt'].get('dynamic_retry_steps', 5)
                        if path is not None:
                            dense = [path[0]]
                            for a, b in zip(path[:-1], path[1:]):
                                dense.extend(interpolate(a, b, self.env.max_increment)[1:])
                            self.path, self.index = dense, min(1, len(dense) - 1)
                        else:
                            self.replan_failures += 1
                    # Hold for this physics step; re-check the new path next time.
                    self.hold_steps += 1
                    return np.zeros(self.env.n)
            finally:
                if human:
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.env.client)
        return np.clip((self.path[self.index] - q) / self.env.max_increment, -1, 1)

    def upcoming_path_safe(self, q):
        """Check upcoming joint edges against current and predicted obstacles.

        Snapshot prediction is conservative sampling, not continuous-time safety
        certification. Query poses and obstacle locations are restored afterwards.
        """
        e = self.env
        dt = e.config['simulation']['timestep'] * e.config['simulation']['substeps']
        end = min(len(self.path), self.index + e.config['rrt'].get('dynamic_lookahead_steps', 3))
        try:
            previous = q
            for offset, waypoint in enumerate(self.path[self.index:end]):
                for future in [0., (offset + 1) * dt, (offset + 2) * dt]:
                    e.move_obstacles(e.sim_time + future)
                    if not e.collision.edge_valid(previous, waypoint):
                        return False
                previous = waypoint
            return True
        finally:
            e.move_obstacles(e.sim_time)
