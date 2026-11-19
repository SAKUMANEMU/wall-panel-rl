"""Obstacles must intersect the physical nominal motion, not merely its vicinity."""
import copy
import json
import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.static_demo_scene import fresh_static_case
import yaml


def make():
    torch.set_num_threads(1)
    parent=json.loads((ROOT/'results/diagnostics/four_static_initial_v1_seed61/manifest.json').read_text())
    settings=copy.deepcopy(parent['evaluation_settings'])
    settings['on_path_obstacles']=yaml.safe_load((ROOT/'configs/four_static_on_path.yaml').read_text())['on_path_obstacles']
    model=PPO.load(ROOT/'results/diagnostics/four_static_initial_v1_seed61/selected_model.zip',device='cpu')
    return parent['configuration'],settings,model


def test_all_four_physically_intersect_nominal_but_endpoints_are_clear():
    config,settings,model=make();before=copy.deepcopy(model.policy.state_dict())
    case,rejects=fresh_static_case(config,settings,20261103,model=model);scene=case['scene']
    assert not rejects and scene['on_path_placement']['all_obstacles_intersect_nominal']
    assert len(scene['obstacles'])==4 and not scene['obstacle_motion']
    env=WallPanelEnv(config)
    try:
        env.reset(seed=scene['seed'],options={'scenario':scene})
        assert env.collision.is_configuration_valid(scene['initial_q'])
        assert env.collision.is_configuration_valid(scene['reference_joint_path'][-1])
        for body,record in zip(env.obstacle_ids,scene['on_path_placement']['obstacles']):
            with env.collision.at_configuration(record['nominal_joint_position']):
                hits=list(p.getClosestPoints(env.robot,body,0.,physicsClientId=env.client))
                hits+=list(p.getClosestPoints(env.panel,body,0.,physicsClientId=env.client))
                assert hits and min(c[8] for c in hits)<0
        for i,a in enumerate(scene['obstacles']):
            for b in scene['obstacles'][i+1:]:
                gap=np.maximum(np.abs(np.asarray(a['center'])-b['center'])-(np.asarray(a['dimensions'])+b['dimensions'])/2,0)
                assert np.linalg.norm(gap)>=config['obstacles']['min_obstacle_gap']
    finally:env.close()
    assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items())


def test_on_path_seed_reproduces_all_positions_and_contact_witnesses():
    config,settings,model=make()
    a,_=fresh_static_case(config,settings,20261103,model=model)
    b,_=fresh_static_case(config,settings,20261103,model=model)
    assert a==b
    assert a['scene']['nominal_motion']['termination_reason']=='success'
