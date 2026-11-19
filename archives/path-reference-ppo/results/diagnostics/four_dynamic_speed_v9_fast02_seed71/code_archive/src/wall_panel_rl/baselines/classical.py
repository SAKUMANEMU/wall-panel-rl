import time
from contextlib import nullcontext
import numpy as np
import pybullet as p
from wall_panel_rl.planning.ik import solve_goal
from wall_panel_rl.planning.rrt_connect import RRTConnect, interpolate
from wall_panel_rl.planning.timed_rrt import TimedRRT
from wall_panel_rl.envs.geometry import pose_error
from wall_panel_rl.planning.static_validation import execution_route,execution_edge_valid,dense_route_valid


class ClassicalController:
    """Track a joint route, with predictive scheduling for moving obstacles."""

    def __init__(self, env, seed=0, *, goal_q=None):
        self.env = env
        self.replans = 0
        self.replan_failures = 0
        self.hold_steps = 0
        self.next_retry = 0
        self.timed = bool(env.obstacle_motion and env.config['rrt'].get('dynamic_mode') == 'timed')
        self.plan_start_step = env.steps
        self.scheduled_waits = 0
        self.schedule_stale = False
        self.verify_static_path = not bool(env.obstacle_motion) and env.config['rrt'].get('verify_static_path',True)
        self.static_validation_resolution = float(env.config['rrt'].get('static_validation_resolution',.005))
        self.static_path_verified = None
        if self.static_validation_resolution<=0:raise ValueError('Invalid static path validation resolution')
        start = time.perf_counter()
        with (nullcontext() if self.timed else env.planning_obstacles()):
            if goal_q is None:
                goal = solve_goal(env, env.observed_target, seed)
            else:
                # Offline imitation can require the SAME IK branch as its input
                # reference. Invalid supplied endpoints never fall back to IK.
                goal = np.asarray(goal_q, dtype=float).copy()
                if goal.shape != (env.n,) or not env.collision.is_configuration_valid(goal):
                    goal = None
                else:
                    with env.collision.at_configuration(goal):
                        dp, dr = pose_error(env.observed_target, env.kinematic_panel_pose())
                    if (dp >= env.config['rrt']['ik_position_tolerance'] or
                        dr >= np.deg2rad(env.config['rrt']['ik_orientation_tolerance_deg'])):
                        goal = None
        self.goal_q = goal
        self.ik_success = goal is not None
        self.path = None
        self.planner = RRTConnect(
            env.lower,
            env.upper,
            env.collision.is_configuration_valid,
            self.verified_edge if self.verify_static_path else env.collision.edge_valid,
            env.config["rrt"],
            seed,
            path_valid=self.verified_route if self.verify_static_path else None,
        )
        if self.timed:
            self.planner = TimedRRT(env, seed)
        if self.ik_success:
            with (nullcontext() if self.timed else env.planning_obstacles()):
                self.path = self.planner.plan(env.joint_state()[0], goal)
        self.planning_success = self.path is not None
        self.index = 1
        if self.planning_success and not self.timed:
            self.path = execution_route(self.path,env.max_increment)
            if self.verify_static_path:
                self.static_path_verified=dense_route_valid(env.collision,self.path,self.static_validation_resolution)
                if not self.static_path_verified:
                    self.path=None;self.planning_success=False
        if self.verify_static_path and not self.planning_success:self.static_path_verified=False
        # Include final densification/validation in the reported planning time.
        self.planning_time = time.perf_counter() - start

    def verified_edge(self,a,b):
        return execution_edge_valid(self.env.collision,a,b,self.env.max_increment,self.static_validation_resolution)

    def verified_route(self,path):
        return dense_route_valid(self.env.collision,execution_route(path,self.env.max_increment),self.static_validation_resolution)

    def action(self, observation=None):
        if not self.planning_success:
            return np.zeros(self.env.n)
        if self.timed:
            return self.timed_action()
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

    def timed_action(self):
        """Track scheduled arrivals; invalidate timing after a hold or deviation."""
        e = self.env
        q, _ = e.joint_state()
        human = e.render_mode == 'human'
        if human:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=e.client)
        try:
            index = min(e.steps-self.plan_start_step+1,len(self.path)-1)
            target = self.path[index]
            safe = (not self.schedule_stale and np.max(np.abs(target-q)) <= e.max_increment
                    and self.planner.edge_valid(q,target,e.sim_time))
            if not safe:
                e.controller.command(q)
                begin = time.perf_counter()
                path = self.planner.plan(q,self.goal_q)
                self.planning_time += time.perf_counter()-begin
                self.replans += 1
                if path is not None:
                    self.path = path
                    self.plan_start_step = e.steps
                    self.schedule_stale = False
                    index = min(1,len(path)-1)
                    target = path[index]
                    safe = self.planner.edge_valid(q,target,e.sim_time)
                else:
                    self.replan_failures += 1
                if not safe:
                    # A hold consumes simulation time. Mark the schedule stale
                    # so the next call cannot resume it at the wrong timestamps.
                    self.schedule_stale = True
                    self.hold_steps += 1
                    return np.zeros(e.n)
            if index and np.allclose(self.path[index],self.path[index-1]):
                self.scheduled_waits += 1
            return np.clip((target-q)/e.max_increment,-1,1)
        finally:
            if human:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=e.client)

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
