"""Two-box projection, label-preserving augmentation, and actor-only held-out BC."""
import copy
import json
import sys
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.avoidance_curriculum import stage_scene
from wall_panel_rl.rl.multi_obstacle_warmup import swap_two_slots,teacher_arrays,warmup_actor,GROUPS
from wall_panel_rl.rl.avoidance_imitation import save_teacher


def test_two_box_bridge_keeps_middle_obstacle_in_first_slot():
    original=json.loads((ROOT/'data/staged_avoidance_v3_seed61/one_static/train.json').read_text())[0]
    saved=copy.deepcopy(original)
    one=stage_scene(original,{'count':1,'mode':'static','amplitude':.05,'speed':.02})
    two=stage_scene(original,{'count':2,'mode':'static','amplitude':.05,'speed':.02})
    assert original==saved and one['obstacles'][0]==two['obstacles'][0]
    assert len(two['obstacles'])==2 and two['obstacle_original_indices']==[2,3]
    assert not two['obstacle_motion'] and one['reference_joint_path']==two['reference_joint_path']


def test_swap_changes_only_two_active_blocks_without_changing_inputs_or_labels():
    x=torch.zeros(3,77);x[:,:20]=.3;x[:,70:]=.4
    x[0,20:30]=.1;x[0,30:40]=.2;x[0,29]=x[0,39]=1
    x[1,20:30]=.1;x[1,29]=1
    before=x.clone();y=swap_two_slots(x,np.random.default_rng(61),1.)
    assert torch.equal(x,before) and torch.equal(y[:,:20],x[:,:20]) and torch.equal(y[:,70:],x[:,70:])
    assert torch.equal(y[0,20:30],x[0,30:40]) and torch.equal(y[0,30:40],x[0,20:30])
    assert torch.equal(y[1:],x[1:]) and torch.equal(y[:,40:70],x[:,40:70])
    assert torch.equal(swap_two_slots(x,np.random.default_rng(61),0),x)


def dataset(path,model,held_value):
    observations=[];actions=[];offsets=[0];seeds=[];tasks=[];terminals=[];anchors=[]
    for seed in (1,2):
        for count,group in [(2,'two_static'),(1,'one_static'),(0,'alignment')]:
            obs=np.zeros((2,77),np.float32);obs[:,70:]=.02
            for k in range(count):obs[:,20+k*10:29+k*10]=.1*(k+1);obs[:,29+k*10]=1
            with torch.no_grad():labels=model.policy.tracking_mean(torch.as_tensor(obs)).numpy()
            labels=np.clip(labels+.01,-1,1) if seed==1 else np.full_like(labels,held_value)
            observations.extend(obs);actions.extend(labels);offsets.append(len(observations));seeds.append(seed);tasks.append(group)
            terminals.append(np.zeros(77,np.float32));anchors.append(np.zeros(7,np.float32))
    path.mkdir()
    save_teacher(path/'teacher.npz',observations,actions,actions,offsets,seeds,tasks,terminals,anchors)


def test_bc_keeps_prior_critic_fixed_and_heldout_labels_out_of_gradients(tmp_path):
    torch.set_num_threads(1)
    source=ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/selected_model.zip'
    spec={'updates':2,'checkpoints':[1,2],'batch_size':8,'learning_rate':1e-4,
        'task_weights':{'two_static':.6,'one_static':.3,'alignment':.1},'seed':61,
        'swap_two_obstacles_probability':.5,'terminal_weight':2.,'gate_retention_weight':.01,'residual_weight':.001,
        'correction_weight':3.}
    states=[]
    for i,held_value in enumerate((-.8,.8)):
        model=PPO.load(source,device='cpu');before=copy.deepcopy(model.policy.state_dict());counter=model.num_timesteps
        dataset(tmp_path/f'data{i}',model,held_value)
        data,parts=teacher_arrays(tmp_path/f'data{i}/teacher.npz',{2})
        assert all(np.all(data['seeds'][ids]==1) for ids in parts['train'].values())
        path,meta=warmup_actor(model,tmp_path/f'data{i}/teacher.npz',{2},spec,tmp_path/f'run{i}',
            lambda candidate,update:{'eligible':True,'score':[1,update]})
        after=PPO.load(path,device='cpu').policy.state_dict()
        assert model.num_timesteps==counter and meta['train_scene_seeds']==[1] and meta['heldout_scene_seeds']==[2]
        for name,value in before.items():
            if not name.startswith(('residual.','prior_gate.')):assert torch.equal(value,after[name])
        states.append(after)
    assert all(torch.equal(v,states[1][k]) for k,v in states[0].items())


def test_cli_no_flag_does_not_start_experiment(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import run_two_static_warmup as m
    monkeypatch.setattr(m,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Training launched')))
    monkeypatch.setattr(sys,'argv',['run_two_static_warmup.py']);m.main()
    assert 'check_only' in capsys.readouterr().out


def test_ppo_start_gate_is_distinct_from_deployment_gate_and_requires_one_retention(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import run_two_static_warmup as m
    parent=json.loads((ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/manifest.json').read_text())
    one=m.stage(parent['evaluation_settings'],1);two=m.stage(one,2)
    counts={1:(18,0,2),2:(12,2,6)}
    def fake_score(candidate,source,config,settings,cases):
        n,c,t=counts[settings['active_stage']['count']]
        aligned={'episodes':10,'success':10,'collision':0,'timeout':0,'joint_limit':0}
        metrics={'stage_target':{'episodes':20,'success':n,'collision':c,'timeout':t,'joint_limit':0},
                 'alignment_far':aligned,'alignment_near':aligned}
        return {'metrics':{'dynamic_ppo':metrics},'checks':{'goal_convergence':True,'alignment_collision_free':True}},[]
    monkeypatch.setattr(m,'score_model',fake_score)
    result,_,_=m.validation(None,{},two,one,[],[])
    assert result['ppo_ready'] and not result['eligible']
    counts[1]=(17,1,2)
    result,_,_=m.validation(None,{},two,one,[],[])
    assert not result['ppo_ready']
    counts[1]=(18,0,2);counts[2]=(16,2,2)
    result,_,_=m.validation(None,{},two,one,[],[])
    assert result['ppo_ready'] and result['eligible']


def test_auxiliary_pool_seeding_excludes_bc_heldout_layouts(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from train_two_static_ppo import SeededImitation
    torch.set_num_threads(1)
    model=PPO.load(ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/selected_model.zip',device='cpu')
    dataset(tmp_path/'teacher',model,.8)
    spec={'capacity':100,'batch_size':2,'checkpoint_every':5000,'updates_per_rollout':1,
          'weight':.2,'learning_rate':1e-5,'maximum_kl':.005}
    callback=SeededImitation(tmp_path/'aux',spec,teacher_path=tmp_path/'teacher/teacher.npz',heldout={2})
    callback.init_callback(SimpleNamespace(policy=model.policy,num_timesteps=40000))
    callback._on_training_start()
    data,parts=teacher_arrays(tmp_path/'teacher/teacher.npz',{2})
    assert len(callback.pool)==6
    expected=np.concatenate([np.arange(data['offsets'][i],data['offsets'][i+1])
                             for i in sorted(int(j) for ids in parts['train'].values() for j in ids)])
    np.testing.assert_array_equal(np.stack([p[0] for p in callback.pool]),data['obs'][expected])
