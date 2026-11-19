"""Independent-verification denominators, uncertainty and paired comparisons."""
import sys
import pytest
from wall_panel_rl.utils.config import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from expanded_validation_metrics import metrics,paired,wilson,true_runs


def test_wilson_handles_extreme_samples_without_false_certainty():
    lo,hi=wilson(0,20)
    assert lo==pytest.approx(0.) and .15<hi<.17
    lo,hi=wilson(20,20)
    assert .83<lo<.85 and hi==pytest.approx(1.)
    assert wilson(0,0) is None


def test_all_failures_remain_in_denominator_and_count_groups():
    rows=[{'seed':i,'obstacle_count':2+i%4,'reason':reason}
          for i,reason in enumerate(['success','collision','timeout','joint_limit','planning_failure'])]
    result=metrics(rows)
    assert result['all']['episodes']==5 and result['all']['success_rate']==.2
    assert sum(result['by_count'][str(n)]['episodes'] for n in [2,3,4,5])==5
    assert result['all']['planning_failure']==1


def test_paired_comparison_excludes_unfinished_pairs_and_detects_duplicates():
    rows=[]
    for seed,a,b in [(1,True,True),(2,True,False),(3,False,False)]:
        rows.extend([{'seed':seed,'arm':'v31_selected','reason':'success' if a else 'timeout'},
                     {'seed':seed,'arm':'v29_baseline','reason':'success' if b else 'timeout'}])
    rows.append({'seed':4,'arm':'v31_selected','reason':'success'})
    result=paired(rows)
    assert result['matched_completed']==3 and result['v31_only_success']==1
    assert result['paired_success_rate_difference']==pytest.approx(1/3)
    with pytest.raises(ValueError,match='Duplicate'):paired(rows+[rows[0]])


def test_longest_wait_counts_control_steps_once():
    runs=true_runs([False,True,True,False,True],[1,2,3,4,5])
    assert runs==[{'first_step':2,'last_step':3,'steps':2},{'first_step':5,'last_step':5,'steps':1}]
