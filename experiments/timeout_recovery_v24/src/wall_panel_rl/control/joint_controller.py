import numpy as np
import pybullet as p


class JointPositionController:
    """Shared physical actuator; never teleports joints during execution."""

    def __init__(self, env):
        self.env = env

    def command(self, q_des):
        e = self.env
        c = e.config["controller"]
        p.setJointMotorControlArray(
            e.robot,
            e.joints,
            p.POSITION_CONTROL,
            targetPositions=np.asarray(q_des).tolist(),
            forces=[c["force"]] * e.n,
            positionGains=[c["position_gain"]] * e.n,
            velocityGains=[c["velocity_gain"]] * e.n,
            physicsClientId=e.client,
        )
