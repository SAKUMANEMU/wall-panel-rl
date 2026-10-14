"""Physical joint-space panel installation environment (no observation delay)."""
import copy
from contextlib import contextmanager
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
from scipy.spatial.transform import Rotation
from wall_panel_rl.utils.config import load_config, ROOT
from wall_panel_rl.envs.geometry import pose_error, rotation_error
from wall_panel_rl.envs.noise import observe_target
from wall_panel_rl.envs.collision import CollisionChecker
from wall_panel_rl.control.joint_controller import JointPositionController


class WallPanelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = copy.deepcopy(config or load_config())
        self.render_mode = render_mode
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self.client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        c = self.config
        p.setTimeStep(c["simulation"]["timestep"], physicsClientId=self.client)
        p.setGravity(0, 0, c["simulation"]["gravity"], physicsClientId=self.client)
        p.setPhysicsEngineParameter(
            numSolverIterations=100,
            deterministicOverlappingPairs=1,
            physicsClientId=self.client,
        )
        self.robot = p.loadURDF(
            str(ROOT / c["robot"]["urdf"]),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            physicsClientId=self.client,
        )
        self.joints = [
            i
            for i in range(p.getNumJoints(self.robot, physicsClientId=self.client))
            if p.getJointInfo(self.robot, i, physicsClientId=self.client)[2]
            == p.JOINT_REVOLUTE
        ]
        self.n = len(self.joints)
        infos = [
            p.getJointInfo(self.robot, j, physicsClientId=self.client)
            for j in self.joints
        ]
        self.lower = np.array([i[8] for i in infos])
        self.upper = np.array([i[9] for i in infos])
        self.ee = c["robot"]["end_effector"]
        self.goal_q = np.array(c["robot"]["goal_q"])
        for j, q in zip(self.joints, self.goal_q):
            p.resetJointState(self.robot, j, q, physicsClientId=self.client)
        panel_pose = self.kinematic_panel_pose()
        self.target_pose = (np.array(panel_pose[0]), np.array(panel_pose[1]))
        self.panel = self.box(
            c["panel"]["dimensions"],
            panel_pose[0],
            [0.85, 0.42, 0.12, 1],
            mass=c["panel"]["mass"],
            orientation=panel_pose[1],
        )
        # Bullet constraints use the parent's inertial frame, whereas offset is
        # specified in the URDF link frame. Convert explicitly.
        link = p.getLinkState(
            self.robot,
            self.ee,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        inv_inertial = p.invertTransform(link[2], link[3])
        parent_frame = p.multiplyTransforms(
            *inv_inertial, c["panel"]["offset"], c["panel"]["orientation"]
        )
        self.constraint = p.createConstraint(
            self.robot,
            self.ee,
            self.panel,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            parent_frame[0],
            [0, 0, 0],
            parentFrameOrientation=parent_frame[1],
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client,
        )
        p.changeConstraint(self.constraint, maxForce=1000, physicsClientId=self.client)
        p.setCollisionFilterPair(
            self.robot, self.panel, self.ee, -1, 0, physicsClientId=self.client
        )
        normal = Rotation.from_quat(panel_pose[1]).apply([0, 0, 1])
        wall_pos = np.array(panel_pose[0]) + normal * (
            c["panel"]["dimensions"][2] / 2
            + c["target"]["wall_clearance"]
            + c["target"]["wall_thickness"] / 2
        )
        self.wall = self.box(
            [0.8, 0.8, c["target"]["wall_thickness"]],
            wall_pos,
            [0.6, 0.65, 0.7, 0.4],
            orientation=panel_pose[1],
        )
        self.box(
            c["panel"]["dimensions"],
            panel_pose[0],
            [0.1, 0.9, 0.3, 0.3],
            orientation=panel_pose[1],
            collision=False,
        )
        self.obstacle_ids, self.obstacles = [], []
        self.obstacle_motion = []
        self.sim_time = 0.0
        self.collision = CollisionChecker(self)
        self.controller = JointPositionController(self)
        self.max_increment = np.deg2rad(c["controller"]["max_joint_increment_deg"])
        self.action_space = spaces.Box(-1, 1, (self.n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (2 * self.n + 6 + 7 * c["obstacles"]["max_count"],),
            dtype=np.float32,
        )
        demo = c.get("demo", {})
        self.camera_distance = demo.get("camera_distance", 1.9)
        self.camera_target = demo.get("camera_target", [0.3, 0, 0.65])
        p.resetDebugVisualizerCamera(
            self.camera_distance, -50, -25, self.camera_target, physicsClientId=self.client
        )

    def box(
        self,
        dimensions,
        position,
        color,
        mass=0,
        orientation=(0, 0, 0, 1),
        collision=True,
    ):
        half = (np.array(dimensions) / 2).tolist()
        shape = (
            p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=self.client
            )
            if collision
            else -1
        )
        visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=self.client
        )
        return p.createMultiBody(
            mass, shape, visual, position, orientation, physicsClientId=self.client
        )

    def kinematic_panel_pose(self):
        link = p.getLinkState(
            self.robot,
            self.ee,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return p.multiplyTransforms(
            link[4],
            link[5],
            self.config["panel"]["offset"],
            self.config["panel"]["orientation"],
        )

    def panel_pose(self):
        return p.getBasePositionAndOrientation(self.panel, physicsClientId=self.client)

    def joint_state(self):
        states = p.getJointStates(self.robot, self.joints, physicsClientId=self.client)
        return np.array([s[0] for s in states]), np.array([s[1] for s in states])

    def set_configuration(self, q):
        """Reset/query only. Physical episode execution uses the controller."""
        for j, angle in zip(self.joints, q):
            p.resetJointState(self.robot, j, angle, physicsClientId=self.client)
        p.resetBasePositionAndOrientation(
            self.panel, *self.kinematic_panel_pose(), physicsClientId=self.client
        )
        p.resetBaseVelocity(
            self.panel, [0, 0, 0], [0, 0, 0], physicsClientId=self.client
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        noise_seed = (options or {}).get("noise_seed", seed)
        if noise_seed is not None:
            self.noise_rng = np.random.default_rng(noise_seed)
        elif not hasattr(self, "noise_rng"):
            self.noise_rng = np.random.default_rng(int(self.np_random.integers(2**32)))
        # reset(seed=None) continues the seeded noise stream rather than
        # silently reseeding it from OS entropy at every training episode.
        for body in self.obstacle_ids:
            p.removeBody(body, physicsClientId=self.client)
        self.obstacle_ids, self.obstacles = [], []
        self.obstacle_motion = []
        self.sim_time = 0.0
        c = self.config
        probability = c["robot"].get("near_start_probability", 0.0)
        near_start = probability > 0 and self.np_random.random() < probability
        start_offset = np.zeros(self.n) if near_start else np.array(c["robot"]["start_offset"])
        spread = c["robot"]["near_start_randomization"] if near_start else c["robot"]["start_randomization"]
        for _ in range(1000):
            q = self.goal_q + start_offset + self.np_random.uniform(-spread, spread, self.n)
            if self.collision.is_configuration_valid(q):
                distance_range = c["robot"].get("start_position_distance_range")
                if distance_range is not None:
                    with self.collision.at_configuration(q):
                        distance = pose_error(self.target_pose, self.kinematic_panel_pose())[0]
                    if not distance_range[0] <= distance <= distance_range[1]:
                        continue
                break
        else:
            raise RuntimeError("Could not sample a valid initial configuration")
        self.set_configuration(q)
        start_position = np.array(self.panel_pose()[0])
        oc = c["obstacles"]
        count = (
            0
            if oc["stage"] == "A"
            else (
                1
                if oc["stage"] == "B"
                else int(self.np_random.integers(oc["min_count"], oc["max_count"] + 1))
            )
        )
        sizes = oc.get("dimensions_by_obstacle")
        if sizes is not None and len(sizes) < count:
            raise ValueError("dimensions_by_obstacle must provide a size for each obstacle")
        for index in range(count):
            dimensions = np.array(oc["dimensions"] if sizes is None else sizes[index], dtype=float)
            for attempt in range(oc["max_placement_attempts"]):
                center = np.array(oc["center"], dtype=float)
                dynamic = oc.get("dynamic", {}).get("enabled", False)
                if dynamic:
                    # Even longitudinal spacing; small transverse random offsets.
                    fraction = (index + 1) / (count + 1)
                    direction = self.target_pose[0] - start_position
                    axis = np.cross(direction, [0., 0., 1.])
                    axis /= np.linalg.norm(axis)
                    center = start_position + fraction * direction
                    # Alternate sides of the reference corridor: boxes on its
                    # centreline would overlap the initially extended arm.
                    side = 1 if index % 2 == 0 else -1
                    center += axis * (side * (oc['dynamic'].get('lateral_offset', .48)
                                              + .05 * (attempt // 30))
                                      + self.np_random.uniform(-0.04, 0.04))
                    center[2] += self.np_random.uniform(-0.04, 0.04)
                    phase = self.np_random.uniform(0, 2 * np.pi)
                    anchor = center.copy()
                    center += axis * oc['dynamic']['amplitude'] * np.sin(phase)
                elif oc["stage"] == "C":
                    center += self.np_random.uniform(
                        -np.array(oc["randomization"]), oc["randomization"]
                    )
                body = self.box(dimensions, center, [0.75, 0.16, 0.2, 1])
                self.obstacle_ids.append(body)
                gap = oc.get("min_obstacle_gap")
                separated = gap is None or all(
                    not p.getClosestPoints(body, other, gap, physicsClientId=self.client)
                    for other in self.obstacle_ids[:-1]
                )
                if dynamic and separated:
                    # Keep endpoint poses outside the entire obstacle sweep.
                    envelope = self.box(dimensions + 2 * oc['dynamic']['amplitude'] * np.abs(axis), anchor, [0, 0, 0, 0])
                    self.obstacle_ids.append(envelope)
                    try:
                        separated = (self.collision.is_configuration_valid(q)
                                     and self.collision.is_configuration_valid(self.goal_q))
                    finally:
                        self.obstacle_ids.pop()
                        p.removeBody(envelope, physicsClientId=self.client)
                if (separated and self.collision.is_configuration_valid(q)
                        and self.collision.is_configuration_valid(self.goal_q)):
                    self.obstacles.append((center, dimensions.copy()))
                    if dynamic:
                        self.obstacle_motion.append(dict(anchor=anchor, axis=axis, phase=phase))
                    break
                self.obstacle_ids.pop()
                p.removeBody(body, physicsClientId=self.client)
            else:
                raise RuntimeError(
                    "Could not place obstacle without start/goal overlap"
                )
        self.controller.command(q)
        self.steps = 0
        self.previous_error = pose_error(self.target_pose, self.panel_pose())
        self.joint_path_length = self.cartesian_path_length = 0.0
        self.observed_target = observe_target(
            self.target_pose, self.noise_rng, c["noise"]
        )
        return self.observation(), self.info()

    def move_obstacles(self, simulation_time):
        """Prescribed sinusoidal motion on the physics clock, not wall-clock time."""
        if not self.obstacle_motion:
            return
        motion = self.config['obstacles']['dynamic']
        amplitude = motion['amplitude']
        omega = motion['max_speed'] / amplitude
        for i, (body, spec) in enumerate(zip(self.obstacle_ids, self.obstacle_motion)):
            center = spec['anchor'] + spec['axis'] * amplitude * np.sin(omega * simulation_time + spec['phase'])
            p.resetBasePositionAndOrientation(body, center, [0, 0, 0, 1], physicsClientId=self.client)
            self.obstacles[i] = (center, self.obstacles[i][1])

    @contextmanager
    def planning_obstacles(self):
        """Conservative swept AABBs let a static RRT plan around moving boxes.

        Used only for collision queries; actual physics bodies remain untouched.
        """
        if not self.obstacle_motion:
            yield
            return
        originals, proxies = self.obstacle_ids, []
        try:
            amplitude = self.config['obstacles']['dynamic']['amplitude']
            for (_, dimensions), spec in zip(self.obstacles, self.obstacle_motion):
                proxies.append(self.box(dimensions + 2 * amplitude * np.abs(spec['axis']),
                                        spec['anchor'], [0, 0, 0, 0]))
            self.obstacle_ids = proxies
            yield
        finally:
            self.obstacle_ids = originals
            for body in proxies:
                p.removeBody(body, physicsClientId=self.client)

    def observation(self):
        q, dq = self.joint_state()
        pose = self.panel_pose()
        # Joint midpoint/range normalization; velocities scaled by 2 rad/s.
        values = [
            2 * (q - self.lower) / (self.upper - self.lower) - 1,
            dq / 2,
            np.asarray(self.observed_target[0]) - pose[0],
            rotation_error(self.observed_target[1], pose[1]) / np.pi,
        ]
        for center, dimensions in self.obstacles:
            values.append(np.r_[center - np.asarray(pose[0]), dimensions, 1.0])
        values.append(
            np.zeros(7 * (self.config["obstacles"]["max_count"] - len(self.obstacles)))
        )
        return np.concatenate(values).astype(np.float32)

    def info(self):
        dp, dr = pose_error(self.target_pose, self.panel_pose())
        return {
            "position_error": dp,
            "orientation_error": dr,
            "success": dp < self.config["target"]["position_tolerance"]
            and dr < np.deg2rad(self.config["target"]["orientation_tolerance_deg"]),
            "collision": self.collision.is_collision(),
            "joint_limit_violation": False,
            "true_panel_pose": self.panel_pose(),
            "observed_target_error": np.r_[
                np.asarray(self.observed_target[0]) - self.panel_pose()[0],
                rotation_error(self.observed_target[1], self.panel_pose()[1]),
            ],
            "joint_path_length_rad": self.joint_path_length,
            "cartesian_path_length_m": self.cartesian_path_length,
        }

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.n,) or not np.isfinite(action).all():
            raise ValueError(
                "Action must be a finite vector with one entry per arm joint"
            )
        invalid_action = bool(np.any(np.abs(action) > 1.000001))
        action = np.clip(action, -1, 1)
        q, _ = self.joint_state()
        desired = q + action * self.max_increment
        violation = invalid_action or not self.collision.within_limits(desired)
        self.controller.command(np.clip(desired, self.lower, self.upper))
        hit = False
        for _ in range(self.config["simulation"]["substeps"]):
            self.sim_time += self.config['simulation']['timestep']
            self.move_obstacles(self.sim_time)
            old_q, _ = self.joint_state()
            old_pos = np.array(self.panel_pose()[0])
            p.stepSimulation(physicsClientId=self.client)
            new_q, _ = self.joint_state()
            self.joint_path_length += float(np.linalg.norm(new_q - old_q))
            self.cartesian_path_length += float(
                np.linalg.norm(np.asarray(self.panel_pose()[0]) - old_pos)
            )
            hit = self.collision.is_collision(use_contacts=True)
            violation = violation or bool(
                np.any(new_q < self.lower - 1e-4) or np.any(new_q > self.upper + 1e-4)
            )
            if hit or violation:
                break
        self.steps += 1
        self.observed_target = observe_target(
            self.target_pose, self.noise_rng, self.config["noise"]
        )
        info = self.info()
        info["collision"] = hit or info["collision"]
        info["joint_limit_violation"] = violation
        info["success"] = bool(
            info["success"] and not info["collision"] and not violation
        )
        weights = self.config["reward"]
        terms = {
            "reward_position_progress": weights["position_progress"]
            * (self.previous_error[0] - info["position_error"]),
            "reward_orientation_progress": weights["orientation_progress"]
            * (self.previous_error[1] - info["orientation_error"]),
            "reward_action": -weights["action"] * float(action @ action),
            "reward_step": -weights["step"],
            "reward_collision": weights["collision"] * int(info["collision"]),
            "reward_joint_limit": weights["joint_limit"] * int(violation),
            "reward_success": weights["success"] * int(info["success"]),
        }
        info.update(terms)
        info["is_success"] = info["success"]
        self.previous_error = info["position_error"], info["orientation_error"]
        terminated = bool(info["success"] or info["collision"] or violation)
        truncated = bool(
            self.steps >= self.config["simulation"]["max_steps"] and not terminated
        )
        return (
            self.observation(),
            float(sum(terms.values())),
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode == "rgb_array":
            view = p.computeViewMatrixFromYawPitchRoll(
                self.camera_target, self.camera_distance, -50, -25, 0, 2
            )
            proj = p.computeProjectionMatrixFOV(55, 4 / 3, 0.05, 5)
            return np.asarray(
                p.getCameraImage(
                    800,
                    600,
                    view,
                    proj,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=self.client,
                )[2]
            )[:, :, :3]

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)
