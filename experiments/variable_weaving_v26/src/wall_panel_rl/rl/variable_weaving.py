"""Variable-count dynamic environment with unchanged 77/7 policy interface."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.stall_recovery import StallDetector


class VariableDynamicEnv(ConditionalRecoveryEnv):
    supports_dynamic_obstacles = True

    def reset(self, *, seed=None, options=None):
        if not options or options.get('task') != 'dynamic':
            raise ValueError('An explicit dynamic scenario is required')
        scene = options['scenario']
        count = len(scene['obstacles'])
        if count not in (2, 3, 4, 5) or len(scene['obstacle_motion']) != count:
            raise ValueError('Expected matching 2--5 moving boxes')
        amplitude, speed = options['strength']
        if not np.isfinite([amplitude, speed]).all() or min(amplitude, speed) <= 0:
            raise ValueError('Positive finite motion bounds required')
        for box, motion in zip(scene['obstacles'], scene['obstacle_motion']):
            extent, omega, phase, anchor = [np.asarray(motion[k]) for k in ('extent', 'omega', 'phase', 'anchor')]
            if any(v.shape != (3,) or not np.isfinite(v).all() for v in (extent, omega, phase, anchor)):
                raise ValueError('Finite 3D motion vectors required')
            if np.any(extent <= 0) or np.any(omega <= 0) or np.linalg.norm(extent) > amplitude+1e-12 or np.linalg.norm(extent*omega) > speed+1e-12:
                raise ValueError('Motion bound exceeded')
            if not np.allclose(box['center'], anchor+extent*np.sin(phase), atol=1e-12, rtol=0):
                raise ValueError('Initial box position does not match motion phase')
        obs, info = super().reset(seed=seed, options=options)
        if obs.shape != (77,) or self.action_space.shape != (7,):
            raise AssertionError('Policy interface changed')
        return obs, dict(info, obstacle_count=count, all_obstacles_dynamic=True)


class VariableCaseBankEnv(VariableDynamicEnv):
    """Uniform count selection followed by uniform scene selection for that count.

    Only raw PPO actions enter step(). Predictive/recovery demonstrations belong
    to a separate supervised stream and never replace on-policy actions.
    """
    def __init__(self, runtime, cases, training, seed=0, *, shaping=True):
        super().__init__(copy.deepcopy(runtime['configuration']), [], copy.deepcopy(runtime['settings']))
        self.bank = copy.deepcopy(cases)
        self.by_count = {n: [c for c in self.bank if len(c['scene']['obstacles']) == n] for n in (2, 3, 4, 5)}
        if any(not cases for cases in self.by_count.values()):
            self.close()
            raise ValueError('A nonempty bank for every count 2--5 is required')
        self.rng = np.random.default_rng(seed)
        self.training_spec = training
        self.shaping = shaping

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and 'scenario' in options:
            selected = copy.deepcopy(options)
        else:
            count = int(self.rng.integers(2, 6))
            case = copy.deepcopy(self.by_count[count][int(self.rng.integers(len(self.by_count[count])))])
            if self.shaping:
                # Swept endpoint/spacing guards are phase independent.
                for box, motion in zip(case['scene']['obstacles'], case['scene']['obstacle_motion']):
                    motion['phase'] = (np.asarray(motion['phase']) + self.rng.uniform(
                        -self.training_spec['phase_jitter_rad'], self.training_spec['phase_jitter_rad'], 3)).tolist()
                    box['center'] = (np.asarray(motion['anchor']) + np.asarray(motion['extent'])*np.sin(motion['phase'])).tolist()
            selected = {'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']}
        obs, info = super().reset(seed=seed, options=selected)
        self.crossings = set()
        self.crossing_sides = []
        self.stall = StallDetector(weight=.1)
        self.stall.observe(self.env.joint_state()[0], info['position_error'], info['orientation_error'], self._remaining())
        self.signed = self._signed_positions()
        return obs, info

    def _remaining(self):
        q, _ = self.env.joint_state()
        path = self.cursor.path
        return float(np.linalg.norm(self.ref-q) + np.linalg.norm(np.diff(path[self.cursor.index:], axis=0), axis=1).sum())

    def _signed_positions(self):
        position = np.asarray(self.env.panel_pose()[0])
        records = self.last_scene['motion_anchor_placement']['obstacles']
        return [float(np.dot(position[:2]-np.asarray(m['anchor'])[:2], r['tangent_xy']))
                for m, r in zip(self.last_scene['obstacle_motion'], records)]

    def step(self, action):
        obs, base, done, truncated, info = super().step(action)
        signed = self._signed_positions()
        pos, quat = self.env.panel_pose()
        half_z = float(np.abs(Rotation.from_quat(quat).as_matrix()[2]) @ (np.asarray(self.env.config['panel']['dimensions'])/2))
        events = 0
        records = self.last_scene['motion_anchor_placement']['obstacles']
        for slot in self.last_scene['motion_anchor_placement']['route_slot_order']:
            if slot in self.crossings or not self.signed[slot] < 0 <= signed[slot]:
                continue
            self.crossings.add(slot)  # One event per box; looping cannot farm rewards.
            center, dims = self.env.obstacles[slot]
            above = pos[2]-half_z-center[2]-dims[2]/2
            below = center[2]-dims[2]/2-pos[2]-half_z
            side = records[slot]['intended_side']
            near = np.linalg.norm(np.asarray(pos)[:2]-center[:2]) <= .24
            clear = (above if side == 'above' else below) >= .01
            if near and clear and not info['collision']:
                self.crossing_sides.append(side)
                events += 1
        self.signed = signed
        stall = self.stall.observe(self.env.joint_state()[0], info['position_error'], info['orientation_error'], self._remaining(), terminal=done or truncated)
        extra = events*self.training_spec['crossing_bonus']+stall if self.shaping else 0.
        reward = (base+extra)*self.training_spec['reward_scale']
        info.update(obstacle_count=len(self.env.obstacle_ids), raw_reward=float(base),
                    reward_vertical_crossing=float(events*self.training_spec['crossing_bonus']) if self.shaping else 0.,
                    reward_stall=float(stall) if self.shaping else 0., vertical_crossings=len(self.crossing_sides),
                    is_success=bool(info['success']))
        return obs, float(reward), done, truncated, info
