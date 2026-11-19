"""Nonzero-time takeover must reproduce the whole PPO prefix and timed suffix."""
import copy,json
import numpy as np,pytest,torch
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_corrections import trace_policy,restore_prefix,corrective_episode,replay_correction
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv

@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1)
    p=json.loads((ROOT/'results/diagnostics/four_staggered_dynamic_preparation_v11_seed91/protocol.json').read_text())
    cases=json.loads((ROOT/'data/four_staggered_dynamic_v11_seed91/train_cases.json').read_text());case=next(c for c in cases if c['scene']['seed']==1020002)
    model=PPO.load(ROOT/'results/diagnostics/four_dynamic_speed_v9_fast02_seed71/selected_model.zip',device='cpu')
    trace=trace_policy(p['configuration'],p['evaluation_settings'],case,model)
    return p,case,trace

@pytest.fixture(scope='module')
def positive(setup):
    p,case,trace=setup
    result=corrective_episode(p['configuration'],p['evaluation_settings'],case,trace,16,1121002)
    assert result['row']['accepted_for_imitation']
    return result

def test_prefix_restores_original_time_velocity_reference_and_immutable_case(setup):
    p,case,trace=setup;saved=copy.deepcopy(case);env=HybridFourEnv(copy.deepcopy(p['configuration']),[],copy.deepcopy(p['evaluation_settings']))
    try:
        obs,_,error=restore_prefix(env,case,[s['action'] for s in trace['states'][:16]],trace['states'][16])
        assert error<=1e-6 and env.env.steps==16 and np.isclose(env.env.sim_time,.8) and env.cursor.index==trace['states'][16]['reference_index']
        np.testing.assert_array_equal(obs,trace['states'][16]['observation'])
        assert case==saved
    finally:env.close()

def test_verified_corrective_labels_exclude_policy_prefix_and_start_at_nonzero_time(positive):
    r=positive['row'];assert r['online_avoidance_planner'] and r['handoff']==16 and r['handoff_time_s']>0
    assert r['pre_verified'] and r['post_time_verified'] and r['physical_replay_verified'] and r['reference_preserved']
    assert r['steps']==r['prefix_steps']+r['suffix_steps'] and len(positive['pairs'])==r['suffix_steps']==r['replay_steps']
    assert r['replay_max_error']<=1e-6 and all(x.shape==(77,) and y.shape==(7,) for x,y in positive['pairs'])

def test_corrupted_observation_is_rejected_by_independent_replay(setup,positive):
    p,case,_=setup;pairs=copy.deepcopy(positive['pairs']);pairs[0][0][0]+=.01
    ok,_,error=replay_correction(p['configuration'],p['evaluation_settings'],case,positive['prefix'],pairs,positive['q'],positive['times'])
    assert not ok and error>.001

def test_zeroed_takeover_clock_is_rejected(setup,positive):
    p,case,_=setup;times=np.asarray(positive['times'])-positive['times'][0]
    ok,steps,_=replay_correction(p['configuration'],p['evaluation_settings'],case,positive['prefix'],positive['pairs'],positive['q'],times)
    assert not ok and steps==0

def test_planning_failure_never_returns_supervised_pairs(setup,monkeypatch):
    from types import SimpleNamespace
    import wall_panel_rl.rl.dynamic_corrections as module
    p,case,trace=setup
    monkeypatch.setattr(module,'make_rrt',lambda *a:SimpleNamespace(timed=True,planning_success=False,planning_time=0.))
    result=module.corrective_episode(p['configuration'],p['evaluation_settings'],case,trace,16,1)
    assert result['row']['reason']=='planning_failure' and not result['row']['accepted_for_imitation'] and not result['pairs']
