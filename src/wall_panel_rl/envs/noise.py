import numpy as np
from scipy.spatial.transform import Rotation


def observe_target(pose, rng, config):
    """Independent per-observation Gaussian translation and rotation-vector noise."""
    position = np.asarray(pose[0]) + rng.normal(0, config["position_std"], 3)
    delta = rng.normal(0, np.deg2rad(config["orientation_std_deg"]), 3)
    orientation = (Rotation.from_rotvec(delta) * Rotation.from_quat(pose[1])).as_quat()
    return position, orientation
