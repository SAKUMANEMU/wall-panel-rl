"""Data-reuse contracts, checkpoint selection, and correction-label regressions."""
import copy
import hashlib
import json
import random
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl import avoidance_imitation as b
from wall_panel_rl.rl import avoidance_curriculum as c
from wall_panel_rl.utils.config import ROOT,load_config


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from staged_avoidance import resolve
    return resolve(yaml.safe_load((ROOT/'configs/staged_avoidance.yaml').read_text()),'one_static')


def reuse_meta(s):
    return {'completed':True,'interface_contract':c.INTERFACE_CONTRACT,'settings':copy.deepcopy(s),
        'environment_config':load_config(ROOT/s['environment']),
        'teacher_sha256':'test-data','source_sha256':hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest()}


def test_reuse_allows_budget_output_changes_but_rejects_semantic_changes(settings):
    meta=reuse_meta(settings);s=copy.deepcopy(settings)
    s['teacher']['updates']=10000;s['stage_output']='models/different_run';s['data']='data/fixed_copy'
    binding=b.teacher_reuse_contract(meta,s,meta['environment_config'])
    assert binding['teacher_sha256']==meta['teacher_sha256']
    s['reference_recovery']['radius']+=.01
    with pytest.raises(ValueError,match='reference/control'):
        b.teacher_reuse_contract(meta,s,meta['environment_config'])
    config=copy.deepcopy(meta['environment_config']);config['simulation']['substeps']+=1
    with pytest.raises(ValueError,match='physical'):
        b.teacher_reuse_contract(meta,settings,config)


def test_reuse_checks_source_hash_and_legacy_attestation(settings,tmp_path):
    meta=reuse_meta(settings);meta.pop('source_sha256')
    proof=tmp_path/'original/bc';proof.mkdir(parents=True)
    meta['settings']['stage_output']=str(proof.parent)
    (proof/'metadata.json').write_text(json.dumps({'teacher_sha256':'different','source_sha256':'wrong'}))
    with pytest.raises(ValueError,match='attest'):
        b.teacher_reuse_contract(meta,settings,meta['environment_config'])
    (proof/'metadata.json').write_text(json.dumps({'teacher_sha256':meta['teacher_sha256'],'source_sha256':'wrong'}))
    with pytest.raises(ValueError,match='weights'):
        b.teacher_reuse_contract(meta,settings,meta['environment_config'])


def score_metrics(success,collision,passed=False,retained=True):
    return {'passed':passed,'checks':{key:retained for key in
        ('alignment_samples','alignment_retained','alignment_collision_free','goal_convergence')},
        'metrics':{'dynamic_ppo':{'stage_target':{'success':success,'collision':collision,
            'joint_limit':0,'timeout':20-success-collision}}}}


def test_checkpoint_selection_prefers_success_and_retention_over_low_mse():
    assert b.closed_loop_score(score_metrics(10,4),.1)>b.closed_loop_score(score_metrics(0,20),.001)
    assert b.closed_loop_score(score_metrics(5,5),.1)>b.closed_loop_score(score_metrics(19,1,retained=False),.001)
    assert b.closed_loop_score(score_metrics(16,1,passed=True),.1)>b.closed_loop_score(score_metrics(18,2),.001)


def test_small_bc_run_reloads_checkpoints_and_selects_closed_loop(settings,tmp_path,monkeypatch):
    torch.set_num_threads(1);s=copy.deepcopy(settings)
    s['source_model']=str(ROOT/s['source_model'])
    s['teacher'].update(updates=200,batch_size=8,minimum_unique_scenes=2,closed_loop_updates=[100,200])
    model=c.initialize_gated(s['source_model'],s)
    teacher=tmp_path/'teacher';teacher.mkdir()
    obs=np.zeros((8,77),np.float32);obs[:,70:]=.02
    obs[2:4,29]=1;obs[6:8,29]=1
    with torch.no_grad():actions=model.policy.reference_mean(torch.as_tensor(obs)).numpy()
    actions[2:4]=.1;actions[6:8]=.1
    terminal=obs[[0,2,4,6]].copy()
    with torch.no_grad():terminal_actions=model.policy.reference_mean(torch.as_tensor(terminal)).numpy()
    b.save_teacher(teacher/'teacher.npz',obs,actions,actions,[0,2,4,6,8],[1,1,2,2],
                   ['alignment','avoidance','alignment','avoidance'],terminal,terminal_actions)
    meta=reuse_meta(s);meta.update(unique_scene_seeds=2,
        teacher_sha256=hashlib.sha256((teacher/'teacher.npz').read_bytes()).hexdigest())
    (teacher/'metadata.json').write_text(json.dumps(meta))
    calls=[]
    def evaluator(candidate,step,loss):
        calls.append((step,candidate.num_timesteps))
        # Evaluation/loading must not change the optimizer's future RNG stream.
        torch.manual_seed(999);np.random.seed(999);random.seed(999)
        return score_metrics(4 if step==100 else 0,16 if step==100 else 20)
    output=tmp_path/'bc'
    b.pretrain_actor(model,teacher,s,output,reuse_environment=meta['environment_config'],evaluator=evaluator)
    report=json.loads((output/'metadata.json').read_text())
    assert calls==[(100,5000),(200,5000)]
    assert report['selected_update']==100 and report['bc_updates']==200
    assert report['selection']=='closed_loop_then_loss' and report['critic_and_prior_unchanged']
    state=torch.load(output/'checkpoints/bc_200.bc_state.pt',weights_only=False)
    assert torch.equal(torch.get_rng_state(),state['torch_rng'])
    selected=PPO.load(output/'model.zip',device='cpu')
    checkpoint=PPO.load(output/'checkpoints/bc_100.zip',device='cpu')
    assert np.array_equal(selected.predict(obs,deterministic=True)[0],checkpoint.predict(obs,deterministic=True)[0])
    assert state['updates']==200 and state['optimizer']['state']
    assert len(json.loads((output/'loss_history.json').read_text()))==2


def test_correction_prefix_is_replayable_but_not_imitation_labels(settings,tmp_path,monkeypatch):
    scene=json.loads((ROOT/'data/dynamic_autonomous_pilot_seed61/train.json').read_text())[0]
    s=copy.deepcopy(settings)
    s['teacher'].update(alignment_episodes=1,avoidance_episodes=2,correction_episodes=1,
        correction_model=str(tmp_path/'prefix.zip'),correction_prefix_min=2,correction_prefix_max=2,
        correction_handoff_clearance=.12,minimum_unique_avoidance_scenes=1,maximum_attempts=3)
    Path(s['teacher']['correction_model']).write_bytes(b'prefix')
    class Policy:
        def reference_mean(self,x):return torch.zeros((len(x),7))
    class Model:
        policy=Policy()
        def predict(self,obs,**kw):return np.full(7,.125),None
    class Env:
        def __init__(self,*a):
            self.env=type('Physical',(),{'steps':0})();self.cursor=type('Cursor',(),{'recoveries':0})()
        def reset(self,**kw):self.env.steps=0;return np.zeros(77,np.float32),{}
        def step(self,action):
            self.env.steps+=1;done=bool(np.isclose(action[0],.25))
            return np.full(77,self.env.steps*.01,np.float32),0.,done,False,{
                'executed_action':np.asarray(action).tolist(),'termination_reason':'success' if done else 'running'}
        def close(self):pass
    class Expert:
        planning_success=True;planning_time=0.;timed=False;replans=0
        def action(self,*a):return np.full(7,.25)
    monkeypatch.setattr(b,'ReferencePreservingPolicy',Policy)
    monkeypatch.setattr(b,'initialize_gated',lambda *a:Model())
    monkeypatch.setattr(PPO,'load',lambda *a,**k:Model())
    monkeypatch.setattr(b,'StagedAvoidanceEnv',Env)
    monkeypatch.setattr(b,'make_teacher',lambda *a:Expert())
    monkeypatch.setattr(c,'clearance_feedback',lambda *a:(0.,.12))
    output=tmp_path/'teacher';b.collect_teacher({},[scene],s,output)
    attempts=json.loads((output/'attempts.json').read_text())
    correction=next(x for x in attempts if x['group']=='correction')
    assert correction['warmup_steps']==2 and np.allclose(correction['warmup_actions'],.125)
    with np.load(output/'teacher.npz',allow_pickle=False) as d:
        assert len(d['obs'])==3 and np.allclose(d['actions'],.25)
        i=next(i for i,x in enumerate(attempts) if x['group']=='correction')
        assert np.allclose(d['obs'][d['offsets'][i]],.02)
    meta=json.loads((output/'metadata.json').read_text())
    assert meta['counts']=={'alignment':1,'avoidance':2}
    assert meta['group_counts']['correction']==1 and meta['completed']


def test_study_defaults_are_bounded_and_check_only(settings,monkeypatch,capsys):
    import run_bc_budget as study
    spec=yaml.safe_load((ROOT/'configs/bc_budget_study.yaml').read_text())
    study.check_study(spec)
    first=study.settings_for(spec,'same_data_10k');second=study.settings_for(spec,'expanded_10k')
    assert first['data']==spec['existing_data'] and first['teacher']['updates']==10000
    assert second['teacher']['avoidance_episodes']==800 and second['teacher']['correction_episodes']==200
    assert second['additional_timesteps']==10000 and second['train_scenes']==256
    def forbidden(*a):raise AssertionError('Training in check mode')
    monkeypatch.setattr(study,'run',forbidden);monkeypatch.setattr(sys,'argv',['run_bc_budget.py'])
    study.main();assert 'check_only' in capsys.readouterr().out
