"""Variable-count known-route environment with 167 observations and 7 actions."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from wall_panel_rl.rl.stall_recovery import StallDetector


from wall_panel_rl.rl.known_route_env import KnownRouteEnv as VariableDynamicEnv
from wall_panel_rl.envs.known_route_motion import upgrade_case


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
        self.case_rng = np.random.default_rng(seed)
        self.training_spec = training
        self.shaping = shaping

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.case_rng = np.random.default_rng(seed)
        if options and 'scenario' in options:
            selected = copy.deepcopy(options)
        else:
            count = int(self.case_rng.integers(2, 6))
            case = copy.deepcopy(self.by_count[count][int(self.case_rng.integers(len(self.by_count[count])))])
            if self.shaping:
                # New route geometry and constant speed are sampled once per reset.
                speed_range=tuple(self.env.config['obstacles']['dynamic'].get('speed_range_m_s',(.04,.22)))
                case = upgrade_case(case, self.case_rng, speed_range, preserve_conflict=False)
            selected = {'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']}
        obs, info = super().reset(seed=seed, options=selected)
        self.crossings = set()
        self.crossing_sides = []
        self.stall = StallDetector(weight=.1)
        self.stall.observe(self.env.joint_state()[0], info['position_error'], info['orientation_error'], self._remaining())
        self.signed = self._signed_positions()
        self.vertical_potential=self._vertical_potential()
        self.last_crossing_rank=-1
        self.last_crossing_height=None
        self.alternating_crossings=0
        return obs, info

    def _vertical_potential(self):
        """Bounded potential for the next nearby intended whole-panel passage.

        Its discounted difference is used below, including a zero terminal
        potential, so waiting or oscillating cannot repeatedly collect it.
        """
        pos,quat=self.env.panel_pose()
        half=float(np.abs(Rotation.from_quat(quat).as_matrix()[2]) @
                   (np.asarray(self.env.config['panel']['dimensions'])/2))
        placement=self.last_scene['motion_anchor_placement']
        for slot in placement['route_slot_order']:
            if slot in self.crossings:continue
            center,dims=self.env.obstacles[slot]
            distance=float(np.linalg.norm(np.asarray(pos[:2])-center[:2]))
            side=placement['obstacles'][slot]['intended_side']
            clear=pos[2]-half-center[2]-dims[2]/2 if side=='above' else center[2]-dims[2]/2-pos[2]-half
            return -float(np.clip(.025-clear,0.,.3))*max(0.,1.-distance/.5)
        return 0.

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
        events = 0;alternating_events=0
        records = self.last_scene['motion_anchor_placement']['obstacles']
        for rank,slot in enumerate(self.last_scene['motion_anchor_placement']['route_slot_order']):
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
                if (self.crossing_sides and side!=self.crossing_sides[-1] and
                    rank>self.last_crossing_rank and
                    abs(pos[2]-self.last_crossing_height)>=.07):
                    alternating_events+=1
                self.crossing_sides.append(side)
                self.last_crossing_rank=rank;self.last_crossing_height=pos[2]
                events += 1
        self.signed = signed
        stall = self.stall.observe(self.env.joint_state()[0], info['position_error'], info['orientation_error'], self._remaining(), terminal=done or truncated)
        new_potential=0. if done or truncated else self._vertical_potential()
        potential_reward=self.training_spec.get('vertical_potential_weight',0.)*(
            .99*new_potential-self.vertical_potential)
        self.vertical_potential=new_potential
        self.alternating_crossings+=alternating_events
        extra = (events*self.training_spec['crossing_bonus']+stall+potential_reward+
                 alternating_events*self.training_spec.get('alternating_bonus',0.)) if self.shaping else 0.
        reward = (base+extra)*self.training_spec['reward_scale']
        info.update(obstacle_count=len(self.env.obstacle_ids), raw_reward=float(base),
                    reward_vertical_crossing=float(events*self.training_spec['crossing_bonus']) if self.shaping else 0.,
                    reward_stall=float(stall) if self.shaping else 0., vertical_crossings=len(self.crossing_sides),
                    alternating_vertical_crossings=self.alternating_crossings,
                    reward_vertical_potential=float(potential_reward) if self.shaping else 0.,
                    is_success=bool(info['success']))
        return obs, float(reward), done, truncated, info
