"""Canonical selection never averages conflicting routes or accepts unsafe links."""
import numpy as np
from wall_panel_rl.rl.consistent_teacher import (
    resample_route, route_length, path_diagnostics, select_canonical, canonical_suffix)


def route(side=0.):
    p = np.zeros((3, 7)); p[1, 0] = .5; p[1, 1] = side; p[-1, 0] = 1
    return p


def test_route_diagnostics_detect_opposite_actions_and_distinct_paths():
    a, b = route(.5), route(-.5)
    stats = path_diagnostics([a, b], [np.r_[0., 1., np.zeros(5)], np.r_[0., -1., np.zeros(5)]])
    assert stats['has_opposing_initial_directions']
    assert stats['max_pair_route_distance_rad'] > .1
    assert np.array_equal(resample_route(a)[[0, -1]], a[[0, -1]])
    assert route_length(a) > route_length(route())


def test_canonical_selection_is_order_independent_and_never_averages_routes():
    candidates = [{'path': route(.5), 'seed': 2, 'first_action': np.ones(7)},
                  {'path': route(-.2), 'seed': 1, 'first_action': np.zeros(7)}]
    i, _ = select_canonical(candidates, route(), np.zeros(7))
    j, _ = select_canonical(candidates[::-1], route(), np.zeros(7))
    assert candidates[i]['seed'] == candidates[::-1][j]['seed'] == 1
    assert np.array_equal(candidates[i]['path'], route(-.2))


def test_correction_reuses_saved_tail_and_rejects_unsafe_connector():
    p = route(.2); q = p[0].copy(); q[2] = .01
    path, index = canonical_suffix(q, p, lambda a, b: b[0] < .9)
    assert index == 1 and np.array_equal(path[1:], p[index:]) and np.array_equal(path[0], q)
    path, index = canonical_suffix(q, p, lambda a, b: False)
    assert path is None and index is None


def test_unsafe_expert_edge_is_a_failed_candidate_not_a_collection_crash():
    import pytest
    from types import SimpleNamespace
    from wall_panel_rl.rl.consistent_teacher import FixedRouteTeacher
    env=SimpleNamespace(obstacle_motion=[], max_increment=.1,
        collision=SimpleNamespace(edge_valid=lambda a,b: False))
    teacher=FixedRouteTeacher(env,route(.2))
    assert not teacher.planning_success and teacher.invalid_edge_index==0 and teacher.path is None
    with pytest.raises(RuntimeError): teacher.action()


def test_selection_rejects_timeout_collapse_and_old_validation_regression(monkeypatch):
    from wall_panel_rl.utils.config import ROOT
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from run_consistent_two_static import coverage_score, learned_gain
    def metrics(success, collision):
        return {'stage_target': {'success': success, 'collision': collision, 'joint_limit': 0, 'timeout': 64-success-collision}}
    baseline = {'two': metrics(55, 5), 'legacy_two': metrics(15, 4), 'ppo_ready': True, 'eligible': False}
    candidate = dict(baseline, two=metrics(30, 0))
    assert coverage_score(candidate, baseline) < coverage_score(baseline, baseline)
    candidate = dict(baseline, two=metrics(56, 4), legacy_two=metrics(14, 4))
    assert coverage_score(candidate, baseline) < coverage_score(baseline, baseline)
    candidate = dict(baseline, two=metrics(55, 4))
    assert coverage_score(candidate, baseline) > coverage_score(baseline, baseline)
    assert learned_gain(candidate['two'], baseline['two'])


def test_new_online_rrt_labels_are_excluded_from_canonical_imitation_pool(monkeypatch):
    import gymnasium as gym
    from gymnasium import spaces
    from types import SimpleNamespace
    from wall_panel_rl.utils.config import ROOT
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from train_consistent_two_static import CanonicalPoolEpisodes
    from wall_panel_rl.rl import episode_imitation
    class Toy(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)
        action_space = spaces.Box(-1, 1, (7,), np.float32)
        def __init__(self): self.env=SimpleNamespace(steps=0); self.task='static'; self.index=0
        def reset(self, *, seed=None, options=None):
            self.index+=1; self.env.steps=0; self.last_scene={'seed':self.index}
            return np.zeros(77, np.float32), {}
        def step(self, action):
            self.env.steps+=1; done=self.env.steps==2
            return np.zeros(77, np.float32), 1., done, False, {
                'termination_reason':'success' if done else 'running', 'position_error':0.,
                'orientation_error':0., 'executed_action':np.asarray(action)}
    class Expert:
        planning_success=True; planning_time=.01
        def action(self, obs): return np.zeros(7, np.float32)
    monkeypatch.setattr(episode_imitation, 'make_rrt', lambda *args: Expert())
    wrapper=CanonicalPoolEpisodes(Toy(), probability=.5, seed=2)
    try:
        wrapper.reset()
        assert len(wrapper.records)==2 and all(r['reason']=='success' for r in wrapper.records)
        assert all(not r['accepted_for_imitation'] and r['imitation_exclusion'] for r in wrapper.records)
        assert wrapper.drain_demonstrations()==[]
        assert wrapper.ppo_active
    finally: wrapper.close()
