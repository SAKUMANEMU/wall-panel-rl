"""Count-only four-box projection preserves initialization and legacy slots."""
import copy
import json
import numpy as np
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.avoidance_curriculum import stage_scene


def test_four_boxes_extend_two_slots_without_reordering_or_mutating_source():
    full=json.loads((ROOT/'data/consistent_two_static_v4_seed61/test_full.json').read_text())[0]
    saved=copy.deepcopy(full);stage={'count':2,'mode':'static','amplitude':.05,'speed':.02}
    two=stage_scene(full,stage);four=stage_scene(full,dict(stage,count=4))
    assert full==saved and len(four['obstacles'])==4 and four['obstacle_original_indices']==[2,3,1,4]
    assert four['obstacles'][:2]==two['obstacles'] and not four['obstacle_motion']
    for key in ('initial_q','target_position','target_orientation','reference_joint_path'):
        assert two[key]==four[key]


def test_four_profile_changes_only_stage_and_does_not_change_sensor_shape(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from run_four_static_initial import four_settings
    parent=json.loads((ROOT/'results/diagnostics/consistent_two_static_v4_seed61/candidate_manifest.json').read_text())
    settings=parent['evaluation_settings'];before=copy.deepcopy(settings);four=four_settings(settings)
    expected=copy.deepcopy(before);expected['active_stage'].update(count=4,name='four_static',mode='static')
    assert four==expected and settings==before and parent['configuration']['obstacles']['max_count']==5


def test_fresh_four_scene_seed_reproduces_start_and_boxes():
    from wall_panel_rl.rl.static_demo_scene import fresh_static_case
    parent=json.loads((ROOT/'results/diagnostics/consistent_two_static_v4_seed61/candidate_manifest.json').read_text())
    settings=copy.deepcopy(parent['evaluation_settings']);settings['active_stage'].update(count=4,name='four_static')
    a,rejected=fresh_static_case(parent['configuration'],settings,20261103)
    b,again=fresh_static_case(parent['configuration'],settings,20261103)
    assert not rejected and not again and a==b
    assert len(a['scene']['obstacles'])==4 and not a['scene']['obstacle_motion']
