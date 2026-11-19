"""Measured vertical-passage diagnostic, extracted unchanged from the experiment.

This is postprocessing only; it never changes rewards, actions, or success.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def passage_measurements(payload, radius=.24, margin=.01, minimum_height_change=.07):
    """Ordered near-box passages using the WHOLE panel vertical bounding box.

    Four and three-box alternating above/below sequences also require alternating
    actual height changes >=7cm. Checking control samples is a display diagnostic;
    physical collision outcomes still come from all actual physics substeps.
    """
    frames = payload['trajectory']
    scene = payload['case']['scene']
    original = scene.get('unadjusted_motion_anchor_placement', scene.get('motion_anchor_placement'))
    order = original['route_slot_order']
    points = np.array([f['panel'] for f in frames])
    orientations = np.array([f['panel_orientation'] for f in frames])
    panel_half_z = np.abs(Rotation.from_quat(orientations).as_matrix())[:, 2, :] @ (np.array(payload['configuration']['panel']['dimensions'])/2)
    possibilities, nearest = {}, []
    for slot in order:
        centers = np.array([f['obstacle_centers'][slot] for f in frames])
        xy = np.linalg.norm((points-centers)[:, :2], axis=1)
        half_z = scene['obstacles'][slot]['dimensions'][2]/2
        above = points[:, 2]-panel_half_z-centers[:, 2]-half_z
        below = centers[:, 2]-half_z-points[:, 2]-panel_half_z
        i = int(np.argmin(xy))
        nearest.append({'slot': slot, 'frame': i, 'xy_distance_m': float(xy[i]),
                        'panel_z': float(points[i,2]), 'box_z': float(centers[i,2]),
                        'panel_above_clearance_m': float(above[i]), 'panel_below_clearance_m': float(below[i])})
        for side, distances in [(1, above), (-1, below)]:
            possibilities[slot, side] = [dict(slot=slot, frame=int(i), time_s=frames[i]['time_s'],
                side='above' if side==1 else 'below', panel_z=float(points[i,2]), box_z=float(centers[i,2]),
                xy_distance_m=float(xy[i]), vertical_clearance_m=float(distances[i]))
                for i in np.where((xy<=radius)&(distances>=margin))[0]]
    # Prefer each box's closest XY encounter. This avoids presenting a remote
    # state as the passage when a tighter, ordered encounter also qualifies.
    closest_found = None
    for count in [4,3]:
        for start in range(5-count):
            selected = nearest[start:start+count]
            sides = [1 if r['panel_above_clearance_m']>=margin else
                     -1 if r['panel_below_clearance_m']>=margin else 0 for r in selected]
            dz = np.diff([r['panel_z'] for r in selected])
            if (all(r['xy_distance_m']<=radius for r in selected) and
                all(a['frame']+1<b['frame'] for a,b in zip(selected,selected[1:])) and
                all(a*b==-1 for a,b in zip(sides,sides[1:])) and
                np.all(np.abs(dz)>=minimum_height_change) and np.all(dz[:-1]*dz[1:]<0)):
                closest_found = [dict(slot=r['slot'], frame=r['frame'], time_s=frames[r['frame']]['time_s'],
                    side='above' if side==1 else 'below', panel_z=r['panel_z'], box_z=r['box_z'],
                    xy_distance_m=r['xy_distance_m'], vertical_clearance_m=r['panel_above_clearance_m']
                    if side==1 else r['panel_below_clearance_m']) for r,side in zip(selected,sides)]
                break
        if closest_found:
            break
    failed = set()
    def search(slots, side, direction, history):
        if not slots:
            return history
        key=(tuple(slots),side,direction,history[-1]['frame'] if history else -1)
        if key in failed:
            return None
        for choice in possibilities[slots[0], side]:
            if history and (choice['frame']<=history[-1]['frame']+1 or
                            direction*(choice['panel_z']-history[-1]['panel_z'])<minimum_height_change):
                continue
            result = search(slots[1:], -side, -direction if history else direction, history+[choice])
            if result:
                return result
        failed.add(key)
        return None
    found = closest_found
    for count in ([] if found else [4,3]):
        for start in range(5-count):
            for side in [1,-1]:
                for direction in [1,-1]:
                    found = search(order[start:start+count], side, direction, [])
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if found:
            break
    z = points[:,2]
    return {'ordered_alternating_passages': found or [], 'passage_count': len(found or []),
            'xy_radius_m': radius, 'whole_panel_vertical_margin_m': margin,
            'minimum_actual_alternating_height_change_m': minimum_height_change,
            'nearest_xy_observations': nearest, 'upward_travel_m': float(np.maximum(np.diff(z),0).sum()),
            'downward_travel_m': float(-np.minimum(np.diff(z),0).sum()), 'height_range_m': float(np.ptp(z))}
