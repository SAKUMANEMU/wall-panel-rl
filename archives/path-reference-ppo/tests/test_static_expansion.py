"""Static expansion changes count only and keeps prospective banks separate."""
import copy
import json
import sys
import numpy as np
import pytest
from wall_panel_rl.utils.config import ROOT


def module():
    sys.path.insert(0,str(ROOT/'scripts'))
    import run_static_expansion as m
    return m


def parent():return json.loads((ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/manifest.json').read_text())


def test_stage_bundle_changes_only_active_stage_not_weights_or_physics():
    m=module();p=parent();original=copy.deepcopy(p);b=m.stage_bundle(p)
    assert p==original and b['model_sha256']==p['model_sha256'] and b['configuration']==p['configuration']
    for key in ('training_settings','evaluation_settings'):
        expected=copy.deepcopy(p[key]);expected['active_stage'].update(count=3,name='three_static')
        assert b[key]==expected


def test_paired_static_projection_retains_same_start_goal_and_middle_box():
    m=module();p=parent();full=json.loads((ROOT/'data/staged_avoidance_v3_seed61/one_static/validation_full.json').read_text())
    original=copy.deepcopy(full)
    one=m.project_cases(full,m.static_stage(p['evaluation_settings'],1))
    three=m.project_cases(full,m.static_stage(p['evaluation_settings'],3))
    assert full==original and len(one)==len(three)==40
    for a,b in zip(one[:20],three[:20]):
        x,y=a['scene'],b['scene']
        assert len(x['obstacles'])==1 and len(y['obstacles'])==3
        assert not x['obstacle_motion'] and not y['obstacle_motion']
        assert x['obstacles'][0]==y['obstacles'][1]
        assert m.minimum_box_gap(y)>=p['configuration']['obstacles']['min_obstacle_gap']
        assert x['initial_q']==y['initial_q'] and x['reference_joint_path']==y['reference_joint_path']
        assert x['target_position']==y['target_position'] and x['target_orientation']==y['target_orientation']
    assert all(not c['scene']['obstacles'] for c in three[20:])


def test_nested_seed_audit_and_unsupported_dynamic_stage():
    m=module();assert m.collect_seeds([{'seed':1},{'scene':{'seed':2}},{'other':[{'seed':3}]}])=={1,2,3}
    settings=parent()['evaluation_settings'];settings['active_stage']['mode']='dynamic'
    with pytest.raises(ValueError,match='static'):m.static_stage(settings,3)


def test_scene_audit_includes_previous_fresh_and_test_banks(tmp_path):
    m=module()
    (tmp_path/'one_fresh.json').write_text(json.dumps([{'scene':{'seed':1006000}}]))
    (tmp_path/'two_test.json').write_text(json.dumps([{'scene':{'seed':1008000}}]))
    assert m.known_scene_seeds(tmp_path)=={1006000,1008000}


def test_default_only_checks_does_not_train(monkeypatch,capsys):
    m=module();monkeypatch.setattr(m,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Experiment started')))
    monkeypatch.setattr(sys,'argv',['run_static_expansion.py']);m.main()
    assert 'check_only' in capsys.readouterr().out
