"""Reject apparent weaving from center heights alone or a constant-height path."""
import sys
import numpy as np
from wall_panel_rl.utils.config import ROOT
sys.path.insert(0, str(ROOT/'scripts'))
from wall_panel_rl.visualization.passage_metrics import passage_measurements


def payload():
    centers = [[i, 0, .4 if i%2==0 else .8] for i in range(4)]
    heights=[.8,.4,.8,.4]
    frames = [dict(time_s=i*.05, panel=[i/2,0,float(np.interp(i/2,range(4),heights))],
                   panel_orientation=[0,0,0,1], obstacle_centers=centers) for i in range(7)]
    return dict(configuration={'panel':{'dimensions':[.3,.2,.03]}},
                case={'scene':{'obstacles':[{'dimensions':[.1,.1,.1]} for _ in range(4)],
                      'motion_anchor_placement':{'route_slot_order':[0,1,2,3]}}}, trajectory=frames)


def test_ordered_actual_rise_fall_with_whole_panel_clearance():
    result = passage_measurements(payload())
    assert result['passage_count']==4
    assert [x['side'] for x in result['ordered_alternating_passages']]==['above','below','above','below']
    assert all(x['xy_distance_m']==0 for x in result['ordered_alternating_passages'])


def test_large_panel_center_height_does_not_prove_whole_panel_clearance():
    data = payload()
    data['configuration']['panel']['dimensions'][2] = 1.
    assert passage_measurements(data)['passage_count']==0


def test_constant_height_is_not_counted_as_up_down_weaving():
    data = payload()
    for frame in data['trajectory']:
        frame['panel'][2] = .6
    assert passage_measurements(data)['passage_count']==0
