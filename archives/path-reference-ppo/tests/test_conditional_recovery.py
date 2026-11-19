"""Conditional recovery acts only after a gate and never rewrites robot actions."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import yaml
from wall_panel_rl.rl.conditional_recovery import RecoveryGate,ConditionalRecoveryEnv
from wall_panel_rl.rl.avoidance_curriculum import StagedAvoidanceEnv,RecoveringReferenceCursor
from wall_panel_rl.utils.config import ROOT


def observe(gate,step,**kwargs):
    defaults={'q':np.zeros(7),'position':.3,'orientation':.4,'index':1,'last_index':5,'reference_distance':1.}
    defaults.update(kwargs);return gate.observe(step,**defaults)


def test_gate_requires_history_minimum_time_and_bounds_attempts_and_successes():
    gate=RecoveryGate()
    for step in range(60):assert not observe(gate,step)
    assert observe(gate,60)
    gate.record_attempt(60,False);assert not observe(gate,61)
    assert observe(gate,80);gate.record_attempt(80,False)
    assert observe(gate,100);gate.record_attempt(100,False)
    assert not observe(gate,140)
    gate=RecoveryGate()
    for step in range(61):allowed=observe(gate,step)
    assert allowed;gate.record_attempt(60,True)
    assert not observe(gate,120)


@pytest.mark.parametrize('reason',['position','orientation','motion','index','final','near_reference','alignment','terminal'])
def test_progress_alignment_and_terminal_states_do_not_trigger(reason):
    gate=RecoveryGate()
    for step in range(61):
        kwargs={}
        if reason=='position':kwargs['position']=.3-step*.001
        if reason=='orientation':kwargs['orientation']=.4-step*.001
        if reason=='motion':kwargs['q']=np.full(7,step*.002)
        if reason=='index':kwargs['index']=1 if step<40 else 2
        if reason=='final':kwargs['index']=5
        if reason=='near_reference':kwargs['reference_distance']=.1
        if reason=='alignment':kwargs['obstacles_present']=False
        if reason=='terminal':kwargs['terminal']=True
        assert not observe(gate,step,**kwargs)


def test_conditional_step_only_changes_reference_after_gate_and_restores_radius(monkeypatch):
    path=np.zeros((5,7));path[:,:2]=[[0,0],[0,.5],[.5,.5],[.5,0],[1,0]]
    q=np.zeros(7);q[:2]=[.75,-.4]
    cursor=RecoveringReferenceCursor(path,radius=.3);ref=cursor.reference(q)
    physical=type('Physical',(),{'steps':59,'lower':np.full(7,-3.),'upper':np.full(7,3.),'obstacle_ids':[7],
        'joint_state':lambda self:(q.copy(),np.zeros(7))})()
    env=ConditionalRecoveryEnv.__new__(ConditionalRecoveryEnv)
    env.env=physical;env.cursor=cursor;env.ref=ref;env.recovery_gate=RecoveryGate();env.conditional_events=[]
    for step in range(30,60):env.recovery_gate.observe(step,q,.3,.4,cursor.index,len(cursor.path)-1,1.)
    supplied=[]
    def parent_step(self,action):
        supplied.append(np.asarray(action).copy());self.env.steps+=1
        return np.zeros(77,np.float32),2.,False,False,{'position_error':.3,'orientation_error':.4,'executed_action':list(action)}
    monkeypatch.setattr(StagedAvoidanceEnv,'step',parent_step)
    action=np.full(7,.1);obs,reward,done,truncated,info=env.step(action)
    assert np.array_equal(supplied[0],action) and reward==2. and not done and not truncated
    assert cursor.radius==.3 and cursor.index==len(cursor.path)-1
    assert info['conditional_recovery_triggered'] and env.recovery_gate.recoveries==1
    assert np.array_equal(obs[:70],np.zeros(70)) and np.allclose(obs[70:],path[-1]/3.)
    assert np.array_equal(physical.joint_state()[0],q)
    env.step(action);assert env.recovery_gate.recoveries==1 and len(env.conditional_events)==1


def test_comparison_rejects_untriggered_episode_changes():
    sys.path.insert(0,str(ROOT/'scripts'))
    from evaluate_conditional_recovery import compare_unchanged_untriggered
    a=[{'suite':'stage_target','seed':1,'reason':'success','steps':60,'position_m':.001}]
    b=[{**a[0],'conditional_recoveries':0}]
    assert compare_unchanged_untriggered(a,b)['identical_untriggered_episodes']==1
    b[0]['reason']='collision'
    with pytest.raises(AssertionError,match='Untriggered episode changed'):compare_unchanged_untriggered(a,b)
    b[0]['conditional_recoveries']=1
    assert compare_unchanged_untriggered(a,b)['triggered_episodes']==1


def test_conditional_study_check_only_does_not_start_evaluation(monkeypatch,capsys):
    import evaluate_conditional_recovery as r
    study=yaml.safe_load((ROOT/'configs/conditional_recovery_study.yaml').read_text());r.check(study)
    monkeypatch.setattr(r,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Evaluation in check mode')))
    monkeypatch.setattr(sys,'argv',['evaluate_conditional_recovery.py','--config',str(ROOT/'configs/conditional_recovery_study.yaml')])
    r.main();assert 'check_only' in capsys.readouterr().out
