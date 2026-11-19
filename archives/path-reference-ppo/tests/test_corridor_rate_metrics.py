"""Installation and successful weaving share the complete valid-scene denominator."""
import sys
import pytest
from wall_panel_rl.utils.config import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_corridor_success_rate import summary_metrics


def test_rate_denominator_includes_failures_and_nonweaving_successes():
    rows=[]
    for reason,weaving in [('success',True),('success',False),('collision',False)]:
        rows.append(dict(reason=reason,steps=10,successful_with_verified_weaving=weaving,
                         zero_command_steps=0,stationary_steps=0,initial_planning_s=0.,
                         control_wall_s=1.,total_wall_s=1.))
    result=summary_metrics(rows)
    assert result['episodes']==3 and result['success']==2 and result['collision']==1
    assert result['success_rate']==pytest.approx(2/3)
    assert result['successful_with_verified_weaving']==1
    assert result['success_and_weaving_rate']==pytest.approx(1/3)
