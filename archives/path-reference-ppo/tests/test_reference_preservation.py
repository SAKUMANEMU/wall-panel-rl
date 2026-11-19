"""Regression checks for the diagnosed BC failure. No optimizer updates."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.rl import avoidance_curriculum as c
from wall_panel_rl.rl import avoidance_imitation as imitation
from wall_panel_rl.rl.autonomous_dynamic_ppo import without_obstacles
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import ROOT,load_config


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from staged_avoidance import resolve,check
    s=resolve(yaml.safe_load((ROOT/'configs/staged_avoidance.yaml').read_text()),'one_static')
    check(s);return s


@pytest.fixture
def model(settings,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Optimizer update in a code-only regression')
    monkeypatch.setattr(torch.optim.Adam,'step',forbidden)
    monkeypatch.setattr(PPO,'learn',forbidden)
    torch.set_num_threads(1)
    return c.initialize_gated(ROOT/settings['source_model'],settings)


def observation_at(model,q):
    obs=np.zeros((1,77),np.float32)
    lower=np.asarray(model.policy.joint_lower);span=np.asarray(model.policy.joint_upper)-lower
    obs[0,:7]=2*(q-lower)/span-1
    obs[0,70:77]=obs[0,:7]
    return obs


def test_alignment_preserved_even_if_correction_weights_are_bad(model):
    with torch.no_grad():
        model.policy.prior_gate[-1].bias.fill_(-10)
        model.policy.residual[-1].bias.fill_(2)
    obs=observation_at(model,np.array(model.policy.goal_angles)+.5)
    obs[:,70:77]+=.03
    with torch.no_grad():
        expected=model.policy.reference_mean(torch.as_tensor(obs)).numpy()
    assert np.array_equal(model.predict(obs,deterministic=True)[0],expected)
    obs[:,29]=1  # A real obstacle slot must enable trainable avoidance.
    assert not np.allclose(model.predict(obs,deterministic=True)[0],expected)


@pytest.mark.parametrize('present',[False,True])
def test_goal_action_zero_with_stale_reference_and_arbitrary_residual(model,present):
    with torch.no_grad():
        model.policy.prior_gate[-1].bias.fill_(-10)
        model.policy.residual[-1].bias.fill_(2)
    obs=observation_at(model,np.array(model.policy.goal_angles));obs[:,70:77]+=.5
    if present:obs[:,29]=1
    assert np.max(np.abs(model.predict(obs,deterministic=True)[0]))<1e-5


def test_correction_taper_is_second_order_near_goal(model):
    strengths=[]
    for displacement in (.01,.02):
        obs=observation_at(model,np.array(model.policy.goal_angles)+displacement);obs[:,29]=1
        strength,_,_=model.policy.correction_components(torch.as_tensor(obs))
        strengths.append(float(strength.item()))
    assert np.isclose(strengths[1]/strengths[0],4,rtol=1e-4)


def test_policy_training_predict_reload_and_obstacle_gradient_agree(model,tmp_path):
    obs=observation_at(model,np.array(model.policy.goal_angles)+.5)
    obs[:,70:77]+=.03;obs[:,29]=1;obs[:,26]=.1
    with torch.no_grad():model.policy.residual[-1].weight.fill_(.01)
    x=torch.tensor(obs,requires_grad=True)
    mean=model.policy.tracking_mean(x)
    mean.sum().backward()
    assert torch.any(x.grad[:,20:70]!=0)
    assert model.policy.tracking_gain_raw.grad is None
    with torch.no_grad():
        action,_,logp=model.policy(x,deterministic=True)
        _,same_logp,_=model.policy.evaluate_actions(x,action)
    assert np.array_equal(action.numpy(),model.predict(obs,deterministic=True)[0])
    assert torch.equal(logp,same_logp)
    path=tmp_path/'model.zip';model.save(path);loaded=PPO.load(path,device='cpu')
    assert np.array_equal(loaded.predict(obs,deterministic=True)[0],action.numpy())
    assert loaded.policy.goal_angles==model.policy.goal_angles
    assert not loaded.policy.tracking_bias.requires_grad and not loaded.policy.log_std.requires_grad


def test_cursor_recovers_forward_after_detour_without_rewriting_reference():
    path=np.zeros((5,7));path[:,:2]=[[0,0],[0,.5],[.5,.5],[.5,0],[1,0]]
    original=path.copy();cursor=c.RecoveringReferenceCursor(path)
    q=np.zeros(7);q[:2]=[.75,.02]
    assert np.array_equal(cursor.reference(q),path[-1])
    assert cursor.recoveries==1 and cursor.index==4
    cursor.reference(np.zeros(7))
    assert cursor.index==4 and np.array_equal(path,original)
    far=c.RecoveringReferenceCursor(path)
    far.reference(np.full(7,5.))
    assert far.index==1 and far.recoveries==0


def test_fixed_endpoint_expert_does_not_run_ik_or_mutate_state(monkeypatch):
    from wall_panel_rl.baselines import classical
    config=load_config(ROOT/'configs/ppo_dynamic_clean.yaml');env=WallPanelEnv(config)
    scene=json.loads((ROOT/'data/dynamic_autonomous_pilot_seed61/train.json').read_text())[0]
    try:
        env.reset(options={'scenario':without_obstacles(scene)})
        before=env.joint_state()[0].copy();observed=[]
        def forbidden(*a,**k):raise AssertionError('Independent IK branch selected')
        original_plan=classical.RRTConnect.plan
        def plan(self,start,goal):
            observed.append(goal.copy())
            return original_plan(self,start,goal)
        monkeypatch.setattr(classical,'solve_goal',forbidden)
        monkeypatch.setattr(classical.RRTConnect,'plan',plan)
        expert=classical.ClassicalController(env,42,goal_q=env.goal_q)
        assert expert.planning_success and np.array_equal(observed[0],env.goal_q)
        assert np.array_equal(before,env.joint_state()[0])
        invalid=classical.ClassicalController(env,42,goal_q=np.full(7,100.))
        assert not invalid.planning_success and len(observed)==1
    finally:env.close()


def test_alignment_teacher_uses_current_reference_without_expert_planning(model,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Planner used for alignment labels')
    monkeypatch.setattr(imitation,'ClassicalController',forbidden)
    teacher=imitation.make_teacher(None,model,'alignment',{},42)
    obs=observation_at(model,np.array(model.policy.goal_angles)+.4);obs[:,70:77]+=.05
    with torch.no_grad():expected=model.policy.reference_mean(torch.as_tensor(obs)).numpy()[0]
    assert np.array_equal(teacher.action(obs[0]),expected)


def test_avoidance_teacher_receives_reference_endpoint_without_replacing_path(model,monkeypatch):
    seen=[];scene={'reference_joint_path':np.arange(21).reshape(3,7).tolist()}
    before=copy.deepcopy(scene)
    def expert(env,seed,**kwargs):seen.append((env,seed,kwargs));return 'offline teacher'
    monkeypatch.setattr(imitation,'ClassicalController',expert)
    assert imitation.make_teacher('physical',model,'avoidance',scene,42)=='offline teacher'
    assert seen==[('physical',42,{'goal_q':scene['reference_joint_path'][-1]})]
    assert scene==before


def test_old_teacher_rejected_before_any_optimizer(settings,monkeypatch,tmp_path):
    path=tmp_path/'teacher';path.mkdir();data=b'legacy';(path/'teacher.npz').write_bytes(data)
    (path/'metadata.json').write_text(json.dumps({'completed':True,'teacher_sha256':hashlib.sha256(data).hexdigest()}))
    def forbidden(*a,**k):raise AssertionError('Optimizer constructed for obsolete teacher')
    monkeypatch.setattr(torch.optim,'Adam',forbidden)
    with pytest.raises(ValueError,match='obsolete'):
        imitation.pretrain_actor(None,path,settings,tmp_path/'bc')


def test_failed_bc_gate_blocks_before_ppo_loading_workers_or_output(settings,monkeypatch,tmp_path):
    import staged_avoidance as entry
    data=tmp_path/'data';teacher=data/'teacher';teacher.mkdir(parents=True)
    (teacher/'teacher.npz').write_bytes(b'demonstrations')
    (teacher/'metadata.json').write_text(json.dumps({'environment_config':{},'interface_contract':c.INTERFACE_CONTRACT}))
    bc=tmp_path/'bc';bc.mkdir();s=copy.deepcopy(settings);s['source_model']=str(ROOT/s['source_model'])
    (bc/'metadata.json').write_text(json.dumps({'settings':s,
        'source_sha256':hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest(),
        'teacher_sha256':hashlib.sha256((teacher/'teacher.npz').read_bytes()).hexdigest()}))
    seen=[]
    def reject(*a):seen.append(True);return {'passed':False}
    def forbidden(*a,**k):raise AssertionError('PPO resources created after failed BC gate')
    monkeypatch.setattr(entry,'validate_bc',reject);monkeypatch.setattr(PPO,'load',forbidden)
    monkeypatch.setattr(entry,'SubprocVecEnv',forbidden);monkeypatch.setattr(PPO,'learn',forbidden)
    output=tmp_path/'ppo'
    with pytest.raises(RuntimeError,match='BC closed-loop gate failed'):
        entry.train({},s,data,bc,output)
    assert seen==[True] and not output.exists()


def test_v2_bc_rejected_before_model_load(settings,monkeypatch,tmp_path):
    import staged_avoidance as entry
    (tmp_path/'metadata.json').write_text(json.dumps({'settings':settings}))
    def forbidden(*a,**k):raise AssertionError('Obsolete BC was loaded')
    monkeypatch.setattr(PPO,'load',forbidden)
    with pytest.raises(ValueError,match='v2 BC'):
        entry.validate_bc({},settings,tmp_path,tmp_path)


def test_bc_validation_rechecks_and_records_failed_alignment(settings,model,monkeypatch,tmp_path):
    import staged_avoidance as entry
    data=tmp_path/'data';teacher=data/'teacher';teacher.mkdir(parents=True)
    (teacher/'teacher.npz').write_bytes(b'demo')
    config=load_config(ROOT/settings['environment'])
    (teacher/'metadata.json').write_text(json.dumps({'interface_contract':c.INTERFACE_CONTRACT,
        'environment_config':config}))
    bc=tmp_path/'bc';bc.mkdir();model.save(bc/'model.zip')
    s=copy.deepcopy(settings);s['source_model']=str(ROOT/settings['source_model'])
    (bc/'metadata.json').write_text(json.dumps({'settings':s,'interface_contract':c.INTERFACE_CONTRACT,
        'source_sha256':hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest(),
        'teacher_sha256':hashlib.sha256((teacher/'teacher.npz').read_bytes()).hexdigest()}))
    scene=json.loads((ROOT/'data/dynamic_autonomous_pilot_seed61/train.json').read_text())[0]
    case={'suite':'alignment_far','scene':without_obstacles(scene),'task':'alignment','strength':[.05,.02]}
    (data/'validation_staged.json').write_text(json.dumps([case]))
    metrics={'stage_target':{'episodes':20,'success':18,'collision':1,'joint_limit':0,'timeout':1},
        'alignment_far':{'episodes':10,'success':10,'collision':0,'joint_limit':0,'timeout':0},
        'alignment_near':{'episodes':10,'success':10,'collision':0,'joint_limit':0,'timeout':0}}
    calls=[]
    def evaluate(*a,**kw):calls.append(kw['env_factory']);return {'dynamic_ppo':copy.deepcopy(metrics)},[]
    monkeypatch.setattr(entry,'evaluate',evaluate)
    assert entry.validate_bc(config,s,data,bc)['passed']
    metrics['alignment_near'].update(success=9,collision=1)
    assert not entry.validate_bc(config,s,data,bc)['passed']
    assert calls==[c.StagedAvoidanceEnv,c.StagedAvoidanceEnv]
    report=json.loads((bc/'closed_loop_gate.json').read_text())
    assert not report['checks']['alignment_collision_free']
    assert report['checks']['goal_convergence'] and not report['training_performed']
    metrics['alignment_near'].update(success=10,collision=0)
    monkeypatch.setattr(c.ReferencePreservingPolicy,'tracking_mean',
                        lambda self,x:x.new_full((len(x),7),.1))
    result=entry.validate_bc(config,s,data,bc)
    assert not result['passed'] and not result['checks']['goal_convergence']
    assert result['checks']['alignment_collision_free'] and len(calls)==3
