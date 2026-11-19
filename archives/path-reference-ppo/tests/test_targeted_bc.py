"""Continuation, baseline retention, late handoffs and leakage regressions."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl import avoidance_imitation as b
from wall_panel_rl.rl import avoidance_curriculum as c
from wall_panel_rl.utils.config import ROOT,load_config


def fixture_data(tmp_path):
    sys.path.insert(0,str(ROOT/'scripts'))
    from staged_avoidance import resolve
    s=resolve(yaml.safe_load((ROOT/'configs/staged_avoidance.yaml').read_text()),'one_static')
    s['source_model']=str(ROOT/s['source_model'])
    s['teacher'].update(updates=200,batch_size=8,minimum_unique_scenes=2,closed_loop_updates=[100,200])
    model=c.initialize_gated(s['source_model'],s);teacher=tmp_path/'teacher';teacher.mkdir()
    obs=np.zeros((8,77),np.float32);obs[:,70:]=.02;obs[2:4,29]=1;obs[6:8,29]=1
    with torch.no_grad():actions=model.policy.reference_mean(torch.as_tensor(obs)).numpy()
    actions[2:4]=.1;actions[6:8]=.1;terminal=obs[[0,2,4,6]].copy()
    with torch.no_grad():ta=model.policy.reference_mean(torch.as_tensor(terminal)).numpy()
    b.save_teacher(teacher/'teacher.npz',obs,actions,actions,[0,2,4,6,8],[1,1,2,2],
                   ['alignment','avoidance','alignment','avoidance'],terminal,ta)
    config=load_config(ROOT/s['environment'])
    meta={'completed':True,'interface_contract':c.INTERFACE_CONTRACT,'settings':copy.deepcopy(s),
          'environment_config':config,'unique_scene_seeds':2,'source_sha256':hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest(),
          'teacher_sha256':hashlib.sha256((teacher/'teacher.npz').read_bytes()).hexdigest()}
    (teacher/'metadata.json').write_text(json.dumps(meta))
    return s,teacher,config


def metric(n):
    return {'passed':False,'checks':{k:True for k in
        ('alignment_samples','alignment_retained','alignment_collision_free','goal_convergence')},
        'metrics':{'dynamic_ppo':{'stage_target':{'success':n,'collision':20-n,'joint_limit':0,'timeout':0}}}}


def test_bc_continuation_matches_uninterrupted_optimizer_and_sampling(tmp_path):
    torch.set_num_threads(1);s,data,config=fixture_data(tmp_path)
    def evaluate(*a):return metric(1)
    whole=copy.deepcopy(s);whole['teacher'].update(updates=400,closed_loop_updates=[200,400])
    b.pretrain_actor(c.initialize_gated(s['source_model'],s),data,whole,tmp_path/'whole',
                     reuse_environment=config,evaluator=evaluate)
    b.pretrain_actor(c.initialize_gated(s['source_model'],s),data,s,tmp_path/'first',
                     reuse_environment=config,evaluator=evaluate)
    first=tmp_path/'first/checkpoints/bc_200.zip'
    b.pretrain_actor(PPO.load(first),data,s,tmp_path/'second',reuse_environment=config,
        evaluator=evaluate,resume_state=first.with_suffix('.bc_state.pt'))
    a=PPO.load(tmp_path/'whole/checkpoints/bc_400.zip').policy.state_dict()
    z=PPO.load(tmp_path/'second/checkpoints/bc_200.zip').policy.state_dict()
    assert all(torch.equal(v,z[k]) for k,v in a.items())
    report=json.loads((tmp_path/'second/metadata.json').read_text())
    assert report['bc_optimizer_resumed'] and report['previous_bc_updates']==200
    assert report['cumulative_bc_updates']==400 and report['ppo_environment_steps']==0


def test_initial_baseline_retained_when_new_updates_regress_and_split_fixed(tmp_path):
    torch.set_num_threads(1);s,data,config=fixture_data(tmp_path)
    model=c.initialize_gated(s['source_model'],s);before=copy.deepcopy(model.policy.state_dict())
    b.pretrain_actor(model,data,s,tmp_path/'run',reuse_environment=config,
        evaluator=lambda candidate,step,loss:metric(10 if step==0 else 0),
        heldout_seeds=[2],include_initial=True)
    report=json.loads((tmp_path/'run/metadata.json').read_text())
    assert report['selected_update']==0 and report['heldout_scene_seeds']==[2]
    assert report['train_scene_seeds']==[1]
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items())


def test_late_handoffs_precede_failure_and_cover_distant_self_collision():
    import run_targeted_bc as r
    t={'reason':'collision','states':[{'step':i,'clearance_m':.30,'route_distance_rad':0.}
                                     for i in range(80)]}
    points=r.handoffs(t,[2,6,12,20,32])
    assert max(points)==77 and min(points)==47 and all(12<k<80 for k in points)
    t['reason']='success';assert r.handoffs(t,[2])==[]


def test_new_study_is_explicit_bounded_and_has_no_ppo_entry(monkeypatch,capsys):
    import run_targeted_bc as r
    study=yaml.safe_load((ROOT/'configs/targeted_bc_study.yaml').read_text())
    assert study['additional_updates']==10000 and study['independent_scenes']==100
    monkeypatch.setattr(r,'check',lambda s:None)
    monkeypatch.setattr(r,'run',lambda s:(_ for _ in ()).throw(AssertionError('run in check mode')))
    monkeypatch.setattr(sys,'argv',['run_targeted_bc.py','--config',str(ROOT/'configs/targeted_bc_study.yaml')])
    r.main();assert 'check_only' in capsys.readouterr().out
