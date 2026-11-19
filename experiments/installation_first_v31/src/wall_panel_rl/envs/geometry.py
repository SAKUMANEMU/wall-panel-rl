"""Quaternion operations use Bullet's xyzw convention."""
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_error(target, actual):
    return (Rotation.from_quat(target) * Rotation.from_quat(actual).inv()).as_rotvec()


def pose_error(target, actual):
    return float(np.linalg.norm(np.asarray(target[0]) - actual[0])), float(
        np.linalg.norm(rotation_error(target[1], actual[1]))
    )
