"""Functional checks for the isolated supervised/rollout integration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from run_adaptation import bc_step, metrics, selection_score
from wall_panel_rl.submission import CaseBankEnv, load_agent
from wall_panel_rl.utils.config import ROOT


def test_bc_improves_an_unsaturated_target_without_mutating_prior():
    torch.set_num_threads(1);torch.manual_seed(71)
    env=CaseBankEnv(ROOT/'data/train_cases.json')
    try:
        obs,_=env.reset(seed=71)
        model=load_agent();policy=model.policy
        policy.configure_learning(train_critic=False)
        x=torch.as_tensor(np.repeat(obs[None],16,axis=0))
        with torch.no_grad():
            initial=policy.tracking_mean(x).clone();y=initial.clone()
            movable=initial.abs()<.85
            assert movable.any()
            y[movable]+=.05
        frozen={n:p.detach().clone() for n,p in policy.named_parameters() if not p.requires_grad}
        before=float((initial-y).square().mean())
        optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-5)
        for _ in range(20):bc_step(policy,optimizer,x,y)
        after=float((policy.tracking_mean(x).detach()-y).square().mean())
        assert after<before
        assert all(torch.equal(p,frozen[n]) for n,p in policy.named_parameters() if n in frozen)
    finally:env.close()


def test_metrics_retains_failed_episodes_and_safety_selection():
    rows=[dict(reason=r,waiting_fraction=.5,stationary_fraction=.5,position_m=.4,orientation_deg=50)
          for r in ('success','collision','timeout','planning_failure')]
    m=metrics(rows)
    assert m['episodes']==4 and m['success_rate']==.25
    assert m['timeout_mean_position_m']==.4
    old={'ppo':{'metrics':dict(m)},'enhanced':{'metrics':dict(m)}}
    changed={'ppo':{'metrics':dict(m)},'enhanced':{'metrics':dict(m,success=3,collision=2)}}
    assert selection_score(old)>selection_score(changed)
