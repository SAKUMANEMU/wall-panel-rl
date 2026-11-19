"""Known closed 3D geometric routes with independent physical travel speed.

p(u) = anchor + extent * sin(harmonics*u + phase). An arc-length lookup separates
the shape from its timing. Speed is sampled once per obstacle/episode, in m/s;
no future random schedule is generated or exposed to either policy/controller.
"""
from functools import lru_cache
import copy
import numpy as np

KIND = 'known_route_arc'
SLOT_FEATURES = 18
OBSERVATIONS = 77 + 5*SLOT_FEATURES


@lru_cache(maxsize=512)
def _arc_table(extent, harmonics, phase):
    u = np.linspace(0., 2*np.pi, 2049)
    points = np.asarray(extent)*np.sin(u[:, None]*harmonics+phase)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if arc[-1] <= 1e-6 or np.any(np.diff(arc) <= 0):
        raise ValueError('Route must have positive finite arc length')
    return u, arc


def arc_table(motion):
    return _arc_table(*(tuple(float(v) for v in motion[key]) for key in ('extent','harmonics','phase')))


def route_state(motion, seconds):
    u, arc = arc_table(motion)
    distance = (motion['start_distance_m']+motion['speed_m_s']*float(seconds)) % arc[-1]
    progress = float(np.interp(distance, arc, u))
    extent, k, phase = (np.asarray(motion[key]) for key in ('extent','harmonics','phase'))
    point = np.asarray(motion['anchor'])+extent*np.sin(k*progress+phase)
    tangent = extent*k*np.cos(k*progress+phase)
    velocity = motion['speed_m_s']*tangent/max(np.linalg.norm(tangent),1e-12)
    return point, velocity, progress


def validate_motion(motion, amplitude=.18, speed_range=(.03,.18)):
    if motion.get('kind') != KIND:
        raise ValueError('Known geometric route required')
    for key in ('anchor','extent','harmonics','phase'):
        v = np.asarray(motion[key])
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError('Finite 3D route parameters required')
    if np.any(np.asarray(motion['extent']) <= 0) or np.linalg.norm(motion['extent']) > amplitude+1e-12:
        raise ValueError('Invalid geometric amplitude')
    k = np.asarray(motion['harmonics'])
    if sorted(k.tolist()) != [1,2,3]:
        raise ValueError('Expected a permutation of three integer shape harmonics')
    if not np.isfinite([motion['speed_m_s'],motion['start_distance_m']]).all() or not speed_range[0] <= motion['speed_m_s'] <= speed_range[1]:
        raise ValueError('Travel speed must be inside the declared physical range')
    arc_table(motion)


def new_motion(anchor, extent, rng, speed_range=(.03,.18), conflict_time=None):
    motion = {'kind':KIND, 'anchor':list(anchor), 'reference':list(anchor), 'extent':list(extent),
              'harmonics':rng.permutation([1,2,3]).tolist(),
              'phase':(rng.integers(0,2,3)*np.pi).tolist(),
              'speed_m_s':float(rng.uniform(*speed_range)), 'start_distance_m':0.}
    _, arc = arc_table(motion)
    motion['start_distance_m'] = (float(rng.uniform(0.,arc[-1])) if conflict_time is None else
                                  float((-motion['speed_m_s']*conflict_time) % arc[-1]))
    validate_motion(motion, speed_range=speed_range)
    return motion


def upgrade_case(original, rng, speed_range=(.03,.18), *, preserve_conflict=True):
    case = copy.deepcopy(original)
    scene = case['scene']
    records = scene['motion_anchor_placement']['obstacles']
    for slot, (box, previous) in enumerate(zip(scene['obstacles'],scene['obstacle_motion'])):
        record = records[slot]
        conflict = record['guide_time_s'] if preserve_conflict and record['timed_conflict'] else None
        motion = new_motion(previous['anchor'],previous['extent'],rng,speed_range,conflict)
        box['center'] = route_state(motion,0.)[0].tolist()
        scene['obstacle_motion'][slot] = motion
    case['strength'] = [.18,float(speed_range[1])]
    scene['known_route_v28'] = {'geometry_known':True, 'speed_sampling':'independent once per obstacle/episode',
        'physical_speed_range_m_s':list(speed_range), 'future_random_schedule_exposed':False,
        'route_features_per_slot':SLOT_FEATURES, 'observations':OBSERVATIONS}
    return case


def observation_features(env):
    panel = np.asarray(env.panel_pose()[0])
    features = np.zeros((5,SLOT_FEATURES),np.float32)
    for slot, motion in enumerate(env.obstacle_motion):
        _, _, progress = route_state(motion,env.sim_time)
        features[slot] = np.r_[np.asarray(motion['anchor'])-panel, np.asarray(motion['extent'])/.18,
            np.asarray(motion['harmonics'])/3., np.sin(motion['phase']),np.cos(motion['phase']),
            np.sin(progress),np.cos(progress),motion['speed_m_s']/.18]
    return features.ravel()
