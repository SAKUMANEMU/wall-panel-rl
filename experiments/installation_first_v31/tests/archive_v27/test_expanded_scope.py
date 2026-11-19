"""Safety contracts for versioned scope and local recovery candidates."""
import copy
import numpy as np
import yaml
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_json,load_runtime,load_agent
from wall_panel_rl.rl.expanded_scope import workspace_offset
from wall_panel_rl.rl.expanded_recovery import DiverseRecoveryController
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv,planning_snapshot,assert_planning_paused


def test_swept_workspace_bound_is_applied_to_whole_box():
    scene=load_json(ROOT/'data/low_diagnostic_cases.json')[0]['scene']
    scope=yaml.safe_load((ROOT/'configs/expanded_scope.yaml').read_text())
    shift=workspace_offset(scene,np.random.default_rng(23),scope)
    for box,motion in zip(scene['obstacles'],scene['obstacle_motion']):
        z=motion['anchor'][2]+shift[2];radius=motion['extent'][2]+box['dimensions'][2]/2
        assert z-radius>=scope['swept_z_bounds_m'][0]-1e-12
        assert z+radius<=scope['swept_z_bounds_m'][1]+1e-12


def test_paired_candidate_rollout_restores_clock_and_state():
    runtime=load_runtime();case=load_json(ROOT/'data/selection_cases.json')[0]
    env=SlowDynamicFourEnv(runtime['configuration'],[],runtime['settings'])
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        controller=DiverseRecoveryController(load_agent(),env,12,runtime['controller']['gains'],dict(horizon=4,commit_steps=2))
        before=planning_snapshot(env)
        found=controller._find_plan(obs,float('inf'))
        assert_planning_paused(env,before)
        modes=controller.candidate_modes()
        assert any(k=='paired' for k,_ in modes) and any(k=='waypoint' for k,_ in modes)
        if found is not None:
            for action in found[3]:assert action.shape==(7,) and np.isfinite(action).all() and np.max(np.abs(action))<=1
    finally:env.close()


def test_high_layout_witness_uses_frozen_physical_guide_corridor():
    from wall_panel_rl.rl.expanded_scope import corridor_witnesses
    runtime=load_runtime();case=load_json(ROOT/'data/train_cases.json')[0]
    rows=corridor_witnesses(runtime['configuration'],case)
    assert len(rows)==4 and {x['slot'] for x in rows}==set(range(4))
    assert all(x['panel'] or x['arm'] for x in rows)
    assert all(x['route_name'] in ('nominal_motion','original_nominal_motion') for x in rows)
