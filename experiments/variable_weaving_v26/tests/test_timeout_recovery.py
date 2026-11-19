"""Persistent local recovery respects bounds and rolls back all trial state."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from run_timeout_study import initialize,runtime_for,baseline_from_cache
from wall_panel_rl.submission import load_json,load_agent
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv,planning_snapshot,assert_planning_paused
from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot
from wall_panel_rl.utils.config import ROOT


def test_persistent_and_lookahead_trials_restore_state():
    initialize();rt=runtime_for('detour');rt['controller']['recovery']['horizon']=4
    case=load_json(ROOT/'data/low_diagnostic_cases.json')[0]
    env=SlowDynamicFourEnv(rt['configuration'],[],rt['settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        controller=TimeoutRecoveryController(load_agent(),env,12,rt['controller']['gains'],rt['controller']['recovery'])
        before=planning_snapshot(env)
        for kind,value in [('persistent',([1,3],[.28,-.28])),('lookahead',(6,.7))]:
            with EpisodeSnapshot(env):
                _,_,_,actions=controller._candidate(kind,value,obs.copy())
                assert len(actions)>0
                assert all(np.isfinite(a).all() and np.max(np.abs(a))<=1 for a in actions)
            assert_planning_paused(env,before)
    finally:env.close()


def test_cached_baseline_is_exact_and_complete():
    initialize();result=baseline_from_cache()
    assert result['metrics']['all']['episodes']==48
    assert result['metrics']['high']['success']==16
    assert result['metrics']['low']['timeout']==18
