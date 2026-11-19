"""Explicit demo-only anchor translations; frozen scene files stay intact."""
import copy
import numpy as np
from wall_panel_rl.rl.slow_dynamic_four import validate_placement


def adjust_obstacle_heights(config, original, offsets):
    """Move each entire sine trajectory vertically, then recheck full bounds.

    Offsets follow GUI labels O1, O2, ... (not the order along the route).
    Amplitude, speed, phase and robot endpoints are preserved.
    """
    case = copy.deepcopy(original)
    scene = case['scene']
    values = np.asarray(offsets, dtype=float)
    if values.shape != (len(scene['obstacles']),) or not np.isfinite(values).all():
        raise ValueError('One finite height offset per obstacle is required')
    previous = np.asarray(scene.get('demo_height_offsets_m', np.zeros_like(values)), dtype=float)
    if previous.shape != values.shape or not np.isfinite(previous).all():
        raise ValueError('Invalid previously applied demo height offsets')
    # Replaying an already adjusted saved scene must not lower O2 a second time.
    for box, motion, dz in zip(scene['obstacles'], scene['obstacle_motion'], values-previous):
        box['center'][2] += float(dz)
        motion['anchor'][2] += float(dz)
        motion['reference'][2] += float(dz)
    scene['demo_height_offsets_m'] = values.tolist()
    # Original on-path collision witnesses refer to the unadjusted anchors.
    # Keep those under a clearly historical key, without claiming they certify
    # the translated layout. Endpoint/spacing validity is rechecked below.
    if 'motion_anchor_placement' in scene:
        scene['unadjusted_motion_anchor_placement'] = scene.pop('motion_anchor_placement')
    validate_placement(config, case)
    return case
