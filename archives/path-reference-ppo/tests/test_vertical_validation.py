"""No success substitution in episode routing; distinguish precision vs waiting."""
import sys
import pytest
from wall_panel_rl.utils.config import ROOT
sys.path.insert(0, str(ROOT/'scripts'))
from evaluate_vertical_layout import mixture_rows, metrics, timeout_category


def row(seed, reason, steps):
    return dict(seed=seed, reason=reason, steps=steps, zero_command_steps=0,
                stationary_steps=0, initial_planning_s=0., control_wall_s=0., total_wall_s=0.)


def test_mixture_preserves_selected_planning_failure_and_collision():
    results = {'rrt': [row(1, 'planning_failure', 0), row(2, 'success', 30)],
               'ppo_progress': [row(1, 'success', 40), row(2, 'collision', 5)]}
    assignments = [{'seed': 1, 'mode': 'rrt'}, {'seed': 2, 'mode': 'ppo_progress'}]
    mixed = mixture_rows(results, assignments)
    assert [r['reason'] for r in mixed] == ['planning_failure', 'collision']
    assert [r['mixture_branch'] for r in mixed] == ['rrt', 'ppo_progress']
    result = metrics(mixed)
    assert result['episodes'] == 2 and result['success_rate'] == 0
    assert result['planning_failure'] == result['collision'] == 1
    mixed[0]['reason'] = 'success'
    assert results['rrt'][0]['reason'] == 'planning_failure'


def test_precision_is_separated_from_waiting_and_actual_stagnation_counts():
    candidate = dict(reason='timeout', steps=240, position_m=.02, orientation_deg=2.,
                     zero_command_steps=80, stationary_steps=80)
    assert timeout_category(candidate, .01, 1.) == 'near_goal_precision'
    candidate.update(position_m=.5, zero_command_steps=0)
    assert timeout_category(candidate, .01, 1.) == 'waiting_or_stagnation'
    candidate['stationary_steps'] = 0
    assert timeout_category(candidate, .01, 1.) == 'slow_or_ineffective_progress'
    candidate['reason'] = 'collision'
    assert timeout_category(candidate, .01, 1.) is None
