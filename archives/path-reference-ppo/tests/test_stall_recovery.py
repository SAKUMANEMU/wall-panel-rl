"""Stagnation reward is bounded feedback, never a change to collision/control."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from wall_panel_rl.rl.stall_recovery import StallDetector,StallAwareEnv
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv,stage_scene
from wall_panel_rl.utils.config import ROOT


def test_stall_penalty_requires_complete_window_and_is_bounded():
    d=StallDetector(window=4,weight=.1)
    for i in range(4):assert d.observe(np.zeros(7),.2,.4,1.)==0
    assert d.observe(np.zeros(7),.2,.4,1.)==-.1
    assert d.observe(np.zeros(7),.2,.4,1.,terminal=True)==0


@pytest.mark.parametrize('progress',['position','orientation','reference','detour'])
def test_real_progress_or_detour_does_not_trigger_stall(progress):
    d=StallDetector(window=4)
    values=[]
    for i in range(5):
        q=np.zeros(7);position=.2;angle=.4;path=1.
        if progress=='position':position-=i*.001
        if progress=='orientation':angle-=i*.001
        if progress=='reference':path-=i*.01
        if progress=='detour':q[0]=i*.03;position+=i*.01
        values.append(d.observe(q,position,angle,path))
    assert values==[0.]*5


def test_stall_environment_preserves_observations_actuator_and_termination():
    torch.set_num_threads(1)
    report=ROOT.parent.parent/'results/diagnostics/targeted_bc_v3_seed61'
    # Actual frozen scenario exercises physical execution without an optimizer.
    s=json.loads((report/'targeted_mix.json').read_text())['settings']
    config=json.loads((report/'manifest.json').read_text())['configuration']
    s=copy.deepcopy(s);s['stall_reward']={'window':4,'weight':.1}
    scene=json.loads((ROOT/'data/staged_avoidance_v3_seed61/one_static/train.json').read_text())[0]
    scene=stage_scene(scene,s['active_stage'])
    options={'scenario':scene,'task':'static','strength':[.05,.02]}
    a=StagedAvoidanceEnv(config,[],s);b=StallAwareEnv(config,[],s)
    try:
        x,_=a.reset(options=options);y,_=b.reset(options=options)
        assert np.array_equal(x,y)
        penalties=[]
        for i in range(20):
            x,r,done,truncated,info=a.step(np.zeros(7))
            y,sr,sd,st,si=b.step(np.zeros(7))
            assert np.array_equal(x,y) and (done,truncated)==(sd,st)
            assert info['executed_action']==si['executed_action']
            assert np.isclose(sr-r,si['reward_stall']);penalties.append(si['reward_stall'])
            if done or truncated:break
        assert min(penalties)==-.1
        b.reset(options=options);assert len(b.stall_detector.history)==1
    finally:a.close();b.close()


def test_timeout_handoffs_leave_recovery_budget_and_skip_other_outcomes():
    sys.path.insert(0,str(ROOT/'scripts'))
    from run_stall_recovery import stall_handoffs
    t={'reason':'timeout','states':[{'q':np.zeros(7).tolist()} for _ in range(240)]}
    points=stall_handoffs(t,160)
    assert len(points)>10 and max(points)==160 and min(points)>=0
    t['reason']='collision';assert stall_handoffs(t)==[]


def test_new_study_checks_only_and_bounds_both_variants(monkeypatch,capsys):
    import run_stall_recovery as r
    s=yaml.safe_load((ROOT/'configs/stall_recovery_study.yaml').read_text())
    assert s['bc_updates']==s['ppo_steps']==10000 and s['bc_learning_rate']==1e-4
    monkeypatch.setattr(r,'check',lambda s:None)
    monkeypatch.setattr(r,'run',lambda s:(_ for _ in ()).throw(AssertionError('training in check mode')))
    monkeypatch.setattr(sys,'argv',['run_stall_recovery.py','--config',str(ROOT/'configs/stall_recovery_study.yaml')])
    r.main();assert 'check_only' in capsys.readouterr().out


def test_ppo_rejects_failed_baseline_before_workers_or_training(tmp_path,monkeypatch):
    import run_stall_recovery as r
    study={'output_root':str(tmp_path/'models'),'report_root':str(tmp_path/'report'),'initial_model':'base.zip'}
    monkeypatch.setattr(r.PPO,'load',lambda *a,**kw:object())
    monkeypatch.setattr(r.stage,'score_bc_model',lambda *a:({'passed':False},[]))
    def forbidden(*a,**kw):raise AssertionError('Workers created after failed BC gate')
    monkeypatch.setattr(r,'SubprocVecEnv',forbidden)
    with pytest.raises(RuntimeError,match='Initial BC gate failed'):
        r.run_ppo(study,{}, {},[],[],None,lambda *a,**kw:None)
    assert not (tmp_path/'models/ppo').exists()


def test_ppo_validation_preserves_training_rng_and_saves_initial_candidate(tmp_path,monkeypatch):
    import random
    import run_stall_recovery as r
    output=tmp_path/'ppo';output.mkdir()
    class Model:
        num_timesteps=5000
        def save(self,path):Path(path).write_bytes(b'checkpoint')
    model=Model()
    result={'passed':True,'checks':{k:True for k in
        ('alignment_samples','alignment_retained','alignment_collision_free','goal_convergence')},
        'metrics':{'dynamic_ppo':{'stage_target':{'success':13,'collision':2,'joint_limit':0,'timeout':5}}}}
    def evaluate(*a):
        torch.manual_seed(999);np.random.seed(999);random.seed(999)
        return result,[]
    monkeypatch.setattr(r.stage,'score_bc_model',evaluate)
    callback=r.PilotValidation(output,None,{}, {},[],lambda *a,**kw:None);callback.model=model
    states=torch.get_rng_state(),np.random.get_state(),random.getstate()
    callback._on_training_start()
    assert torch.equal(states[0],torch.get_rng_state())
    assert np.array_equal(states[1][1],np.random.get_state()[1]) and states[2]==random.getstate()
    selection=json.loads((output/'selection.json').read_text())
    assert selection['additional_steps']==0 and selection['gate']['passed']
    assert (output/'best_model.zip').read_bytes()==b'checkpoint'
