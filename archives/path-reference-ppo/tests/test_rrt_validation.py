"""Physical collision attribution and honest separation of plan/execute/retries."""
import json
from types import SimpleNamespace
import torch
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl import rrt_validation as m


def source_case():
    root=ROOT/'results/diagnostics/four_static_on_path_v1_seed61'
    parent=json.loads((root/'manifest.json').read_text())
    cached=json.loads((root/'on_path.json').read_text())
    row=next(r for r in cached['rows'] if r['reason']=='collision')
    cases=json.loads((ROOT/'data/four_static_on_path_v1_seed61/cases.json').read_text())
    case=next(c for c in cases if c['scene']['seed']==row['seed'])
    return root,parent,case,row


def test_physical_initialization_replay_preserves_result_and_attributes_first_contact():
    torch.set_num_threads(1);root,parent,case,expected=source_case()
    model=PPO.load(root/'selected_model.zip',device='cpu')
    row,trace=m.run_episode(model,parent['configuration'],parent['evaluation_settings'],case,'ppo')
    assert row['reason']=='collision' and row['steps']==expected['steps']
    assert row['first_collision_step']==row['steps'] and row['first_collision_sim_seconds']>0
    assert row['contacts'] and row['dominant_contact'] in row['contacts']
    assert all(c['moving_body'] in ('arm','panel') for c in row['contacts'])
    assert all(c['obstacle_slot'] in (1,2,3,4) for c in row['contacts'] if c['kind']=='obstacle')
    assert len(trace['executed_joint_route'])==row['steps']+1 and not trace['planned_joint_route']


def test_planning_failure_records_zero_execution_without_fallback(monkeypatch):
    _,parent,case,_=source_case()
    monkeypatch.setattr(m,'make_rrt',lambda *a:SimpleNamespace(planning_success=False,ik_success=False,planning_time=.1))
    def forbidden(*a,**k):raise AssertionError('Executed an action after a failed plan')
    monkeypatch.setattr(m.ConditionalRecoveryEnv,'step',forbidden)
    row,trace=m.run_episode(None,parent['configuration'],parent['evaluation_settings'],case,'rrt',123)
    assert row['reason']=='planning_failure' and row['steps']==0
    assert not row['planning_success'] and row['planned_route_recheck'] is None
    assert len(trace['executed_joint_route'])==1 and row['first_collision_step'] is None


def test_retry_success_does_not_rewrite_primary_rate_or_prove_unresolved_infeasible():
    primary=[{'seed':1,'reason':'success','planning_success':True},
             {'seed':2,'reason':'collision','planning_success':True}]
    retry=[{'seed':2,'reason':'success','planning_success':True},
           {'seed':2,'reason':'planning_failure','planning_success':False}]
    result=m.feasibility(primary,retry)
    assert result['primary']['success']==1 and result['primary']['collision']==1
    assert result['retry_trials']['success']==1 and result['retry_trials']['planning_failure']==1
    assert result['scenes_with_any_successful_execution']==2
    assert not result['unresolved_scene_seeds']
    result=m.feasibility(primary,[])
    assert result['unresolved_scene_seeds']==[2] and result['failure_is_not_a_proof_of_infeasibility']
