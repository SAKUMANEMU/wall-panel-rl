"""Regression: densification cannot turn an unchecked route into an action plan."""
import json
from types import SimpleNamespace
import numpy as np
from wall_panel_rl.planning.rrt_connect import RRTConnect,interpolate
from wall_panel_rl.planning.static_validation import execution_edge_valid,dense_route_valid
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import ROOT


def test_refinement_catches_a_gap_between_original_coarse_samples():
    valid=lambda q:not .145<float(q[0])<.155
    checker=SimpleNamespace(is_configuration_valid=valid,
        edge_valid=lambda a,b:all(valid(q) for q in interpolate(a,b,.2)))
    a,b=np.array([0.]),np.array([1.])
    assert checker.edge_valid(a,b)
    assert not execution_edge_valid(checker,a,b,.4,.005)


def test_rejected_direct_route_searches_again_and_does_not_smooth_it_back():
    config={'timeout':1.,'max_iterations':100,'goal_bias':.15,'step_size':.2,'shortcut_attempts':10}
    planner=RRTConnect([0,0],[1,1],lambda q:True,lambda a,b:True,config,61,path_valid=lambda path:len(path)>2)
    path=planner.plan(np.array([.1,.1]),np.array([.9,.9]))
    assert path is not None and len(path)>2 and planner.rejected_paths>=1


def test_all_rejected_routes_return_failure_within_search_budget():
    config={'timeout':.05,'max_iterations':10,'goal_bias':.15,'step_size':.2,'shortcut_attempts':0}
    planner=RRTConnect([0,0],[1,1],lambda q:True,lambda a,b:True,config,61,path_valid=lambda path:False)
    assert planner.plan(np.array([.1,.1]),np.array([.9,.9])) is None
    assert planner.rejected_paths>0 and planner.iterations<=10


def test_four_archived_clearance_failures_are_rejected_without_state_changes():
    source=ROOT/'results/diagnostics/four_on_path_rrt_validation_v1_seed61'
    protocol=json.loads((source/'protocol.json').read_text())
    cases={c['scene']['seed']:c for c in json.loads((ROOT/'data/four_static_on_path_v1_seed61/cases.json').read_text())}
    env=WallPanelEnv(protocol['configuration'])
    try:
        for seed in (1018018,1018025,1018029,1018087):
            scene=cases[seed]['scene'];env.reset(seed=seed,options={'scenario':scene})
            before=env.joint_state();pose=env.panel_pose()
            path=json.loads((source/'traces/rrt_primary'/f'{seed}.json').read_text())['planned_joint_route']
            assert not dense_route_valid(env.collision,[np.asarray(q) for q in path],.005)
            after=env.joint_state()
            assert np.array_equal(before[0],after[0]) and np.array_equal(before[1],after[1])
            assert np.array_equal(pose[0],env.panel_pose()[0])
    finally:env.close()


def test_controller_rejects_unsafe_route_even_if_planner_bypasses_validator(monkeypatch):
    from wall_panel_rl.baselines.classical import ClassicalController
    source=ROOT/'results/diagnostics/four_on_path_rrt_validation_v1_seed61'
    protocol=json.loads((source/'protocol.json').read_text())
    case=next(c for c in json.loads((ROOT/'data/four_static_on_path_v1_seed61/cases.json').read_text()) if c['scene']['seed']==1018018)
    path=json.loads((source/'traces/rrt_primary/1018018.json').read_text())['planned_joint_route']
    monkeypatch.setattr(RRTConnect,'plan',lambda *a:[np.asarray(q) for q in path])
    env=WallPanelEnv(protocol['configuration'])
    try:
        env.reset(seed=1018018,options={'scenario':case['scene']})
        before=env.joint_state()[0].copy()
        controller=ClassicalController(env,42,goal_q=env.goal_q)
        assert not controller.planning_success and controller.static_path_verified is False
        assert controller.path is None and np.array_equal(before,env.joint_state()[0])
        assert np.array_equal(controller.action(),np.zeros(7))
    finally:env.close()
