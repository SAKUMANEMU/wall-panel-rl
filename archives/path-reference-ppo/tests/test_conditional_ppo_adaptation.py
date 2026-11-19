"""Preserve training shaping and execute the validated reference rule consistently."""
import copy
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl.conditional_recovery import ConditionalTrainingEnv,ConditionalRecoveryEnv
from wall_panel_rl.rl.wall_feedback import WallAwareEnv
from wall_panel_rl.utils.config import ROOT


def setup():
    sys.path.insert(0,str(ROOT/'scripts'))
    import train_conditional_ppo as entry
    study=yaml.safe_load((ROOT/'configs/conditional_ppo_adaptation.yaml').read_text())
    return entry,study,entry.inputs(study)


def test_training_wrapper_retains_wall_and_stall_feedback_without_action_change():
    torch.set_num_threads(1);entry,study,(settings,_,config)=setup()
    cases=entry.read(Path(study['base_scenes'])/'validation_staged.json')
    case=next(c for c in cases if c['suite']=='stage_target')
    a=WallAwareEnv(config,[],settings);b=ConditionalTrainingEnv(config,[],settings)
    options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']}
    try:
        x,_=a.reset(options=options);y,_=b.reset(options=options);assert np.array_equal(x,y)
        penalties=[]
        for _ in range(35):
            x,r,d,t,info=a.step(np.zeros(7));y,rr,dd,tt,ii=b.step(np.zeros(7))
            assert np.array_equal(x,y) and r==rr and (d,t)==(dd,tt)
            assert info['executed_action']==ii['executed_action']
            assert info['reward_stall']==ii['reward_stall'] and info['reward_wall_clearance']==ii['reward_wall_clearance']
            assert ii['conditional_recovery_count']==0;penalties.append(ii['reward_stall'])
            if d or t:break
        assert min(penalties)<0
        b.reset(options=options);assert b.recovery_gate.attempts==0 and len(b.stall_detector.history)==1
    finally:a.close();b.close()


def test_conditional_scorer_reproduces_recovery_baseline_and_goal_convergence():
    torch.set_num_threads(1);entry,study,(_,settings,config)=setup()
    cases=[c for c in entry.read(Path(study['base_scenes'])/'validation_staged.json') if c['suite']!='five_full_diagnostic']
    model=PPO.load(study['initial_model'],device='cpu')
    result,rows=entry.score_model(model,None,config,settings,cases)
    assert result['metrics']['dynamic_ppo']['stage_target']=={'episodes':20,'success':18,'collision':2,'timeout':0,'joint_limit':0}
    assert result['passed'] and result['checks']['goal_convergence']
    recovered=[r for r in rows if r['conditional_recoveries']]
    assert {r['seed'] for r in recovered}=={900002,900003} and all(r['reason']=='success' for r in recovered)


def test_training_and_evaluation_match_through_actual_recovery():
    torch.set_num_threads(1);entry,study,(training,evaluation,config)=setup()
    case=next(c for c in entry.read(Path(study['base_scenes'])/'validation_staged.json')
              if c['suite']=='stage_target' and c['scene']['seed']==900002)
    a=ConditionalRecoveryEnv(config,[],evaluation);b=ConditionalTrainingEnv(config,[],training)
    model=PPO.load(study['initial_model'],device='cpu')
    options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']}
    try:
        x,_=a.reset(options=options);y,_=b.reset(options=options)
        while True:
            assert np.array_equal(x,y);action=model.predict(x,deterministic=True)[0]
            x,r,d,t,info=a.step(action);y,rr,dd,tt,ii=b.step(action)
            assert np.array_equal(x,y) and (d,t)==(dd,tt)
            assert info['executed_action']==ii['executed_action']
            assert info['conditional_recovery_triggered']==ii['conditional_recovery_triggered']
            assert np.isclose(rr-r,ii['reward_stall']+ii['reward_wall_clearance'])
            if d or t:break
        assert info['termination_reason']=='success' and info['conditional_recovery_count']==ii['conditional_recovery_count']==1
    finally:a.close();b.close()


def test_actual_sampling_counters_include_partial_episodes(tmp_path):
    entry,study,_=setup()
    callback=entry.ConditionalValidation(tmp_path,None,{}, {},[],lambda *a,**kw:None)
    callback.locals={'infos':[{'conditional_recovery_attempted':True,'conditional_recovery_triggered':True,
                              'reward_stall':-.1,'reward_wall_clearance':-.25},{}]}
    assert callback._on_step()
    assert callback.sampling['observed_transitions']==2 and callback.sampling['recovery_success_events']==1
    assert callback.sampling['stall_penalty_sum']==-.1 and callback.sampling['wall_penalty_sum']==-.25


def test_new_scorer_can_block_before_ppo_resources(tmp_path,monkeypatch):
    entry,study,_=setup()
    import run_stall_recovery as old
    trial={'output_root':str(tmp_path/'models'),'report_root':str(tmp_path/'report'),'initial_model':'input.zip'}
    monkeypatch.setattr(old.PPO,'load',lambda *a,**kw:object())
    def forbidden(*a,**kw):raise AssertionError('Old scorer or workers used')
    monkeypatch.setattr(old.stage,'score_bc_model',forbidden);monkeypatch.setattr(old,'SubprocVecEnv',forbidden)
    with pytest.raises(RuntimeError,match='Initial BC gate failed'):
        old.run_ppo(trial,{}, {},[],[],None,lambda *a,**kw:None,initial_scorer=lambda *a:({'passed':False},[]))
    assert not (tmp_path/'models/ppo').exists()


def test_demo_reads_conditional_manifest_and_uses_same_runtime(monkeypatch,capsys):
    entry,study,_=setup();import demo_autonomous_ppo as demo
    from wall_panel_rl.planning.rrt_connect import RRTConnect
    monkeypatch.setattr(RRTConnect,'plan',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Runtime planner')))
    monkeypatch.setattr(sys,'argv',['demo_autonomous_ppo.py','--headless','--seed','900002',
        '--model',study['initial_model'],'--manifest',str(Path(study['conditional_report'])/'manifest.json'),
        '--scene-bank',str(ROOT/'data/staged_avoidance_v3_seed61/one_static/validation_staged.json')])
    demo.main();out=capsys.readouterr().out
    assert 'ConditionalRecoveryEnv' in out and '"reason": "success"' in out


def test_adaptation_cli_is_explicit_and_bounded(monkeypatch,capsys):
    entry,study,_=setup();changed=copy.deepcopy(study);changed['ppo_steps']=20000
    with pytest.raises(ValueError,match='Exactly one 10k'):entry.inputs(changed)
    monkeypatch.setattr(entry,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Training in check mode')))
    monkeypatch.setattr(sys,'argv',['train_conditional_ppo.py','--config',str(ROOT/'configs/conditional_ppo_adaptation.yaml')])
    entry.main();assert 'check_only' in capsys.readouterr().out
