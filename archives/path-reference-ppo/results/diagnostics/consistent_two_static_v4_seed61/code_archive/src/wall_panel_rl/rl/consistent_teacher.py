"""Static expert route diagnostics and canonical suffix teachers.

This is an offline imitation tool, never a policy-time avoidance controller.
Select one feasible RRT route per layout instead of averaging labels from
different routes. A correction reconnects to that saved route through a checked
straight edge; a missing connector is a recorded failure, not an unsafe label.
"""
import numpy as np
from wall_panel_rl.planning.rrt_connect import interpolate


def resample_route(path, count=32):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('Expected a finite (N,7) joint route')
    lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if lengths[-1] < 1e-12: return np.repeat(path[:1], count, axis=0)
    keep = np.r_[True, np.diff(lengths) > 1e-12]
    return np.stack([np.interp(np.linspace(0, lengths[-1], count), lengths[keep], path[keep, j])
                     for j in range(7)], axis=1)


def route_length(path):
    return float(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1).sum())


def reference_departure(path, reference):
    q = resample_route(path); ref = np.asarray(reference, dtype=float)
    v = np.diff(ref, axis=0); starts = ref[:-1]
    fractions = np.clip(np.sum((q[:, None]-starts)*v, axis=2)/
        np.maximum(np.sum(v*v, axis=1), 1e-12), 0, 1)
    distance = np.linalg.norm(q[:, None]-starts-fractions[:, :, None]*v, axis=2)
    return float(distance.min(axis=1).mean())


def path_diagnostics(paths, first_actions):
    pairs = []
    for i, a in enumerate(paths):
        for j in range(i+1, len(paths)):
            x, y = np.asarray(first_actions[i]), np.asarray(first_actions[j])
            active = (np.abs(x) > .05) & (np.abs(y) > .05)
            pairs.append({'first_action_mse': float(np.mean((x-y)**2)),
                'opposing_active_joint_fraction': float(np.mean(np.sign(x[active]) != np.sign(y[active]))) if active.any() else 0.,
                'mean_route_distance_rad': float(np.linalg.norm(resample_route(a)-resample_route(paths[j]), axis=1).mean())})
    return {'route_pairs': pairs,
        'has_opposing_initial_directions': any(p['opposing_active_joint_fraction'] > 0 for p in pairs),
        'max_pair_route_distance_rad': max((p['mean_route_distance_rad'] for p in pairs), default=0.)}


def select_canonical(candidates, reference, base_action):
    """Use a geometric/reference cost, not validation or test outcomes."""
    if not candidates: raise ValueError('No feasible expert route')
    scores = []
    for item in candidates:
        length = route_length(item['path']); departure = reference_departure(item['path'], reference)
        mismatch = float(np.mean((np.asarray(item['first_action'])-base_action)**2))
        scores.append({'seed': item['seed'], 'length_rad': length,
            'reference_departure_rad': departure, 'initial_action_mse_to_base': mismatch,
            'cost': length + .5*departure + .25*mismatch})
    index = min(range(len(candidates)), key=lambda i: (round(scores[i]['cost'], 10), scores[i]['seed']))
    return index, scores


def canonical_suffix(q, path, edge_valid):
    """Choose a checked connector and reuse a literal tail of the canonical route."""
    q = np.asarray(q, dtype=float); path = np.asarray(path, dtype=float)
    nearest = int(np.argmin(np.linalg.norm(path-q, axis=1)))
    start = max(1, nearest-2)
    tail_lengths = np.r_[np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)[::-1])[::-1], 0.]
    indices = sorted(range(start, len(path)), key=lambda k: (float(np.linalg.norm(q-path[k])+tail_lengths[k]), k))
    for index in indices:
        if edge_valid(q, path[index]):
            return np.vstack([q, path[index:]]), index
    return None, None


class FixedRouteTeacher:
    """Track an already collision-checked static route using the existing servo."""
    def __init__(self, env, path):
        if env.obstacle_motion: raise ValueError('Canonical teacher is static only')
        self.env = env; self.index = 1; self.path = None; self.invalid_edge_index = None
        self.planning_success = path is not None
        if self.planning_success:
            dense = [np.asarray(path[0], dtype=float)]
            for edge, (a, b) in enumerate(zip(path[:-1], path[1:])):
                if not env.collision.edge_valid(a, b):
                    self.planning_success = False; self.invalid_edge_index = edge
                    return
                dense.extend(interpolate(a, b, env.max_increment)[1:])
            self.path = np.asarray(dense)
            if len(self.path) < 2: raise ValueError('Empty canonical route')

    def action(self, observation=None):
        if not self.planning_success: raise RuntimeError('No safe canonical connector')
        q, _ = self.env.joint_state()
        while self.index < len(self.path)-1 and np.linalg.norm(self.path[self.index]-q) < self.env.config['rrt']['waypoint_tolerance']:
            self.index += 1
        return np.clip((self.path[self.index]-q)/self.env.max_increment, -1, 1)
