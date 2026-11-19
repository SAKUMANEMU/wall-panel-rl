"""Diagnostic classification, matched scene accounting and inference-only demo."""
import copy
import sys
import numpy as np
import pytest
from wall_panel_rl.utils.config import ROOT


def modules():
    sys.path.insert(0,str(ROOT/'scripts'))
    import compare_static_candidates as m
    return m


def stationary_trace():
    row={'q':np.zeros(7).tolist(),'position_m':.2,'orientation_deg':10.,'cancellation_ratio':.001,
         'action_norm':.0001,'reference_index':3,'last_reference_index':5,'reference_distance_rad':.4,
         'route_distance_rad':.35,'reference_recoveries':0,'wall_clearance_capped_m':.025,'goal_max_joint_error_rad':.6}
    return {'seed':1,'position_m':.2,'orientation_deg':10.,'states':[copy.deepcopy(row) for _ in range(40)]}


def test_stall_classification_reports_cancellation_and_cursor_without_claiming_bug():
    m=modules();trace=stationary_trace();r=m.stall_summary(trace)
    assert r['kind']=='nearly_stationary' and r['strong_action_cancellation']
    assert r['reference_index_change_last40']==0 and not r['cursor_at_final_waypoint']
    for i,row in enumerate(trace['states']):
        row['q'][0]=i*.01;row['position_m']-=i*.001
    assert m.stall_summary(trace)['kind']=='slow_task_progress'


def test_paired_accounting_excludes_alignment_and_rejects_different_scenes():
    m=modules()
    a=[{'suite':'stage_target','seed':1,'reason':'success'},
       {'suite':'stage_target','seed':2,'reason':'collision'},
       {'suite':'alignment_far','seed':1,'reason':'success'}]
    b=[{'suite':'stage_target','seed':2,'reason':'timeout'},
       {'suite':'stage_target','seed':1,'reason':'success'}]
    result=m.paired_outcomes(a,b)
    assert result['episodes']==2 and result['transitions']=={'success -> success':1,'collision -> timeout':1}
    with pytest.raises(ValueError,match='same cases'):m.paired_outcomes(a,b[:1])


def test_autonomous_demo_replays_success_without_runtime_planner(monkeypatch,capsys):
    modules();import demo_autonomous_ppo as demo
    from wall_panel_rl.baselines.classical import ClassicalController
    from wall_panel_rl.planning.rrt_connect import RRTConnect
    def forbidden(*a,**kw):raise AssertionError('Runtime planner in autonomous demo')
    monkeypatch.setattr(ClassicalController,'__init__',forbidden)
    monkeypatch.setattr(RRTConnect,'plan',forbidden)
    # Pin this historical baseline even when the current demo release changes.
    report='/home/johnsonchen/wall-panel-rl/results/diagnostics/static_stability_v3_seed61'
    monkeypatch.setattr(sys,'argv',['demo_autonomous_ppo.py','--headless','--seed','1000000',
        '--model',report+'/selected_model.zip','--manifest',report+'/manifest.json',
        '--scene-bank',str(ROOT/'data/clearance_ppo_v3_seed61/independent_cases.json')])
    demo.main();text=capsys.readouterr().out
    assert '"reason": "success"' in text and '"steps": 65' in text
    assert '"online_avoidance_planner": false' in text
