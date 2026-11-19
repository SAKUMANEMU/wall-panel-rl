"""Wall shaping does not redefine installation or choose unsafe regressions."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from wall_panel_rl.rl import wall_feedback as w
from wall_panel_rl.utils.config import ROOT


def metric(success=14,collision=3,timeout=3,retained=True):
    return {'passed':True,'checks':{k:retained for k in
        ('alignment_samples','alignment_retained','alignment_collision_free','goal_convergence')},
        'metrics':{'dynamic_ppo':{'stage_target':{'episodes':20,'success':success,'collision':collision,
                                              'timeout':timeout,'joint_limit':0}}}}


def test_collision_selector_requires_no_success_or_timeout_regression():
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_clearance_ppo import collision_score
    base=metric();target=base['metrics']['dynamic_ppo']['stage_target']
    baseline=collision_score(base,target)
    assert collision_score(metric(15,2,3),target)>baseline
    assert collision_score(metric(17,3,0),target)>baseline
    assert collision_score(metric(17,4,0),target)<baseline
    assert collision_score(metric(13,1,6),target)<baseline
    assert collision_score(metric(14,2,4),target)<baseline
    assert collision_score(metric(16,2,2,retained=False),target)<baseline


def test_wall_penalty_is_bounded_and_monotone(monkeypatch):
    env=type('Physical',(),{'robot':1,'panel':2,'wall':3,'client':0})()
    result=[]
    for distance in (.03,.015,.005,-.002):
        contact=[0.]*9;contact[8]=distance
        monkeypatch.setattr(w.p,'getClosestPoints',lambda *a,**kw:[contact])
        penalty,_=w.wall_feedback(env,.02,1.)
        assert -1<=penalty<=0;result.append(penalty)
    assert result==sorted(result,reverse=True) and result[0]==0 and result[-1]==-1
    assert w.wall_feedback(env,.02,0.)[0]==0


def physical_settings():
    path=ROOT.parent.parent/'results/diagnostics/clearance_ppo_v3_seed61/collision_diagnosis.json'
    meta=json.loads(path.read_text())
    s=copy.deepcopy(meta['settings']);s['wall_reward']={'distance':.02,'weight':1.}
    return meta,s


def test_nominal_installation_has_no_wall_penalty():
    meta,s=physical_settings();case=meta['traces']['train'][0]['case']
    env=w.WallAwareEnv(meta['configuration'],[],s)
    try:
        env.reset(options={'scenario':case['scene'],'task':'static','strength':[.05,.02]})
        with env.env.collision.at_configuration(env.env.goal_q):
            penalty,distance=w.wall_feedback(env.env,.02,1.)
            assert penalty==0 and distance==.02 and env.env.info()['success']
    finally:env.close()


def test_wall_weight_changes_only_reward_for_identical_action_replay():
    torch.set_num_threads(1);meta,s=physical_settings()
    t=next(t for t in meta['traces']['train'] if t['reason']=='collision')
    control=copy.deepcopy(s);control['wall_reward']['weight']=0.
    a=w.WallAwareEnv(meta['configuration'],[],control);b=w.WallAwareEnv(meta['configuration'],[],s)
    options={'scenario':t['case']['scene'],'task':'static','strength':[.05,.02]}
    penalties=[]
    try:
        x,_=a.reset(options=options);y,_=b.reset(options=options);assert np.array_equal(x,y)
        for row in t['states']:
            x,r,done,truncated,info=a.step(row['executed_action'])
            y,rr,dd,tt,ii=b.step(row['executed_action'])
            assert np.array_equal(x,y) and (done,truncated)==(dd,tt)
            assert info['executed_action']==ii['executed_action']
            assert np.isclose(rr-r,ii['reward_wall_clearance'])
            penalties.append(ii['reward_wall_clearance'])
            if done or truncated:break
        assert info['termination_reason']=='collision' and min(penalties)<0
    finally:a.close();b.close()


def test_clearance_study_defaults_to_check_only_with_one_changed_weight(monkeypatch,capsys):
    import run_clearance_ppo as r
    study=yaml.safe_load((ROOT/'configs/clearance_ppo_study.yaml').read_text())
    r.check(study)
    assert study['variants']=={'control':0.,'wall_feedback':1.} and study['ppo_steps']==10000
    monkeypatch.setattr(r,'run',lambda *a:(_ for _ in ()).throw(AssertionError('training in check mode')))
    monkeypatch.setattr(sys,'argv',['run_clearance_ppo.py','--config',str(ROOT/'configs/clearance_ppo_study.yaml')])
    r.main();assert 'check_only' in capsys.readouterr().out
