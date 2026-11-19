"""Episode-level ownership and explicit collision-first model selection."""
import copy
import numpy as np
import pytest
from gymnasium import spaces
from wall_panel_rl.rl.episode_policy import EpisodePolicy,safety_first_score


class Model:
    observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
    action_space=spaces.Box(-1,1,(7,),np.float32)

    def __init__(self,value):self.value=value;self.calls=0
    def predict(self,observation,*,deterministic):
        self.calls+=1
        return np.full(7,self.value),None


def test_selection_is_once_per_episode_with_independent_reproducible_rng():
    bc,ppo=Model(.2),Model(-.3)
    router=EpisodePolicy(bc,ppo,seed=61)
    expected=np.random.default_rng(61).random(100)<.5
    for choice in expected:
        assert router.begin_episode()==('bc' if choice else 'ppo')
        with pytest.raises(RuntimeError,match='End the current'):router.begin_episode()
        for _ in range(5):
            action,_=router.predict(np.zeros(77))
            assert np.all(action==(.2 if choice else -.3))
        router.end_episode()
    assert bc.calls==5*expected.sum() and ppo.calls==5*(~expected).sum()
    assert router.episodes_started==100
    with pytest.raises(RuntimeError,match='begin_episode'):router.predict(np.zeros(77))


def test_probability_is_not_a_forced_even_split_and_ignores_global_rng():
    a=EpisodePolicy(Model(0),Model(1),seed=61)
    b=EpisodePolicy(Model(0),Model(1),seed=61)
    choices=[]
    for _ in range(20):
        np.random.random(20)
        x=a.begin_episode();y=b.begin_episode();assert x==y
        choices.append(x);a.end_episode();b.end_episode()
    assert 0<choices.count('bc')<20
    assert choices.count('bc')!=10  # Bernoulli draws, not an exact quota.


@pytest.mark.parametrize('probability,expected',[(0.,'ppo'),(1.,'bc')])
def test_single_policy_modes(probability,expected):
    router=EpisodePolicy(Model(0),Model(1),probability,seed=1)
    for _ in range(3):
        assert router.begin_episode()==expected;router.end_episode()


def test_reject_wrong_spaces_and_probability():
    with pytest.raises(ValueError):EpisodePolicy(Model(0),Model(1),float('nan'))
    bad=Model(0);bad.observation_space=spaces.Box(-1,1,(70,),np.float32)
    with pytest.raises(ValueError,match='77-feature'):EpisodePolicy(bad,Model(1))
    bad=Model(0);bad.action_space=spaces.Box(-2,2,(7,),np.float32)
    with pytest.raises(ValueError,match='normalized'):EpisodePolicy(bad,Model(1))


def result(success,collision,timeout):
    aligned={'episodes':10,'success':10,'collision':0,'timeout':0,'joint_limit':0}
    return {'checks':{'alignment_collision_free':True,'goal_convergence':True},
        'metrics':{'dynamic_ppo':{'stage_target':{'episodes':20,'success':success,
            'collision':collision,'timeout':timeout,'joint_limit':0},
            'alignment_far':copy.deepcopy(aligned),'alignment_near':copy.deepcopy(aligned)}}}


def test_safety_priority_prefers_professor_candidate_but_keeps_success_floor():
    thresholds={'minimum_episodes':20,'minimum_alignment_episodes':10,
        'minimum_success_rate':.8,'maximum_collision_rate':.1,'minimum_alignment_success_rate':.9}
    baseline=result(18,2,0);candidate=result(17,0,3)
    assert safety_first_score(candidate,thresholds)>safety_first_score(baseline,thresholds)
    assert safety_first_score(result(0,0,20),thresholds)<safety_first_score(baseline,thresholds)
    unsafe=copy.deepcopy(candidate);unsafe['checks']['goal_convergence']=False
    assert safety_first_score(unsafe,thresholds)<safety_first_score(baseline,thresholds)
