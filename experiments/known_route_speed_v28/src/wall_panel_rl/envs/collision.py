"""One collision definition for simulated execution, IK, and RRT edges."""
from contextlib import contextmanager
import numpy as np
import pybullet as p


class CollisionChecker:
    def __init__(self, env):
        self.env = env

    def within_limits(self, q):
        e = self.env
        return bool(
            np.all(np.isfinite(q)) and np.all(q >= e.lower) and np.all(q <= e.upper)
        )

    def is_collision(self, use_contacts=False, margin=None):
        e = self.env
        kw = {"physicsClientId": e.client}
        margin = e.config["collision"]["margin"] if margin is None else margin

        def contacts(a, b):
            if use_contacts and margin == 0:
                return [c for c in p.getContactPoints(a, b, **kw) if c[8] <= 0]
            return p.getClosestPoints(a, b, margin, **kw)

        for body in e.obstacle_ids + [e.wall]:
            for moving in [e.robot, e.panel]:
                if contacts(moving, body):
                    return True
        for contact in contacts(e.robot, e.robot):
            # Parent/child geometries overlap by construction. Other links count.
            a, b = contact[3:5]
            if abs(a - b) > 1:
                return True
        for contact in contacts(e.robot, e.panel):
            if contact[3] != e.ee:
                return True
        return False

    @contextmanager
    def at_configuration(self, q):
        """Temporary kinematic query; restore positions AND velocities afterwards."""
        e = self.env
        states = p.getJointStates(e.robot, e.joints, physicsClientId=e.client)
        panel_pose = p.getBasePositionAndOrientation(e.panel, physicsClientId=e.client)
        panel_velocity = p.getBaseVelocity(e.panel, physicsClientId=e.client)
        try:
            e.set_configuration(q)
            p.performCollisionDetection(physicsClientId=e.client)
            yield
        finally:
            for j, state in zip(e.joints, states):
                p.resetJointState(
                    e.robot, j, state[0], state[1], physicsClientId=e.client
                )
            p.resetBasePositionAndOrientation(
                e.panel, *panel_pose, physicsClientId=e.client
            )
            p.resetBaseVelocity(e.panel, *panel_velocity, physicsClientId=e.client)
            p.performCollisionDetection(physicsClientId=e.client)

    def is_configuration_valid(self, q):
        q = np.asarray(q)
        if not self.within_limits(q):
            return False
        with self.at_configuration(q):
            return not self.is_collision(
                margin=self.env.config["collision"].get("planning_clearance", 0.0)
            )

    def edge_valid(self, a, b):
        resolution = self.env.config["collision"]["edge_resolution"]
        count = max(1, int(np.ceil(np.linalg.norm(np.asarray(b) - a) / resolution)))
        return all(self.is_configuration_valid(q) for q in np.linspace(a, b, count + 1))
