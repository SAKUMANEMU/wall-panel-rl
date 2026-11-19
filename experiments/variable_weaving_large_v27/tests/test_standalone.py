"""Integration contracts for a self-contained, direct-inference policy."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from unittest.mock import patch
import numpy as np
import torch
from run_standalone import converted,weighted_bc_loss,score
from wall_panel_rl.submission import CaseBankEnv,load_agent,load_json,run_episode
from wall_panel_rl.utils.config import ROOT


def test_conversion_preserves_mean_and_exposes_avoidance_parameters():
    torch.set_num_threads(1);env=CaseBankEnv(ROOT/'data/train_cases.json')
    try:
        obs,_=env.reset(seed=22);x=torch.as_tensor(obs[None])
        old=load_agent();new=converted(ROOT/'models/current_ppo.zip')
        with torch.no_grad():assert torch.equal(old.policy.tracking_mean(x),new.policy.tracking_mean(x))
        new.policy.configure_learning(train_critic=False)
        trained=[n for n,p in new.policy.named_parameters() if p.requires_grad]
        assert any(n.startswith('prior_gate.') for n in trained)
        assert any(n.startswith('residual.') for n in trained)
        assert any(n.startswith('multi_adapter.') for n in trained)
        assert all(n.startswith(('multi_adapter.','residual.','prior_gate.')) for n in trained)
        assert not new.policy.log_std.requires_grad
    finally:env.close()


def test_preclamp_supervision_has_gradient_when_action_saturates():
    env=CaseBankEnv(ROOT/'data/train_cases.json')
    try:
        obs,_=env.reset(seed=22);x=torch.as_tensor(obs[None]);model=converted(ROOT/'models/current_ppo.zip')
        model.policy.configure_learning(train_critic=False)
        with torch.no_grad():model.policy.multi_adapter[-1].bias.fill_(2.)
        assert (model.policy.imitation_mean(x)>1).any()
        loss=weighted_bc_loss(model.policy,x,torch.zeros((1,7)),torch.ones(1))
        loss.backward()
        grad=model.policy.multi_adapter[-1].bias.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
    finally:env.close()


def test_direct_evaluation_has_no_predictive_teacher_calls():
    case=load_json(ROOT/'data/train_cases.json')[0]
    with patch('wall_panel_rl.rl.predictive_safety.PredictiveSafetyController.action',side_effect=AssertionError('Teacher called during direct inference')):
        row,_=run_episode(converted(ROOT/'models/current_ppo.zip'),case,'ppo')
    assert row['local_rescues']==0 and not row['uses_global_rrt']


def test_pure_success_controls_selection():
    def result(success,collision):
        return {'metrics':{'success':success,'joint_limit':0,'collision':collision},
                'rows':[{'position_m':.1,'orientation_deg':5}]}
    assert score(result(10,6))>score(result(9,0))
    assert score(result(9,2))>score(result(9,3))
