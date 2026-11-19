import numpy as np
import pybullet as p
from wall_panel_rl.envs.geometry import pose_error


def solve_goal(env, observed_target, seed=0):
    """Multistart IK of the observed panel pose; verify FK and collisions."""
    c = env.config["rrt"]
    inv = p.invertTransform(
        env.config["panel"]["offset"], env.config["panel"]["orientation"]
    )
    ee_pose = p.multiplyTransforms(*observed_target, *inv)
    rng = np.random.default_rng(seed)
    current, _ = env.joint_state()
    for attempt in range(c["ik_attempts"]):
        initial = current if attempt == 0 else rng.uniform(env.lower, env.upper)
        with env.collision.at_configuration(initial):
            result = np.array(
                p.calculateInverseKinematics(
                    env.robot,
                    env.ee,
                    *ee_pose,
                    lowerLimits=env.lower.tolist(),
                    upperLimits=env.upper.tolist(),
                    jointRanges=(env.upper - env.lower).tolist(),
                    restPoses=initial.tolist(),
                    maxNumIterations=c["ik_iterations"],
                    residualThreshold=1e-5,
                    physicsClientId=env.client,
                )
            )[: env.n]
            env.set_configuration(result)
            dp, dr = pose_error(observed_target, env.kinematic_panel_pose())
        if (
            dp < c["ik_position_tolerance"]
            and dr < np.deg2rad(c["ik_orientation_tolerance_deg"])
            and env.collision.is_configuration_valid(result)
        ):
            return result
    return None
