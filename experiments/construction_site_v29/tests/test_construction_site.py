"""Physical boundaries must affect queries, physics and complete motion bounds."""
import copy
import numpy as np
import pybullet as p
import pytest
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.submission import load_runtime,load_json
from wall_panel_rl.rl.slow_dynamic_four import validate_placement
from wall_panel_rl.utils.config import ROOT


def test_endpoints_valid_and_only_mounting_base_floor_contact_ignored():
    env=WallPanelEnv(load_runtime()['configuration'])
    try:
        raw=p.getClosestPoints(env.robot,env.floor,0,physicsClientId=env.client)
        assert raw and all(c[3]==-1 for c in raw)
        assert env.collision.is_configuration_valid(env.goal_q)
        assert not env.collision.site_contact_types()
        low=np.zeros(7);low[1]=1.9
        env.set_configuration(low)
        moving=[c for c in p.getClosestPoints(env.robot,env.floor,0,physicsClientId=env.client) if c[3]!=-1]
        assert moving and 'floor' in env.collision.site_contact_types()
        assert not env.collision.is_configuration_valid(low)
    finally:env.close()


def test_full_cutaway_ceiling_still_blocks_robot_and_payload():
    env=WallPanelEnv(load_runtime()['configuration'])
    try:
        assert env.config['construction_site']['cutaway_ceiling']
        assert p.getCollisionShapeData(env.ceiling,-1,physicsClientId=env.client)
        upright=np.zeros(7);env.set_configuration(upright)
        assert 'ceiling' in env.collision.site_contact_types()
        assert not env.collision.is_configuration_valid(upright)
        env.set_configuration(env.goal_q)
        env.reset(seed=7,options={'scenario':{'initial_q':env.goal_q.tolist(),
            'target_position':env.target_pose[0].tolist(),'obstacles':[],'obstacle_motion':[]}})
        pos,quat=env.panel_pose()
        p.resetBasePositionAndOrientation(env.panel,[pos[0],pos[1],1.10],quat,physicsClientId=env.client)
        assert env.info()['ceiling_collision']
    finally:env.close()


def test_real_physics_episode_terminates_on_floor_contact():
    config=load_runtime()['configuration'];config['obstacles']['stage']='A'
    env=WallPanelEnv(config)
    try:
        _,info=env.reset(seed=3440000)
        assert not info['collision']
        for _ in range(60):
            action=np.zeros(7);action[1]=.5
            _,_,done,truncated,info=env.step(action)
            if done or truncated:break
        assert done and info['collision'] and info['floor_collision']
        assert not info['joint_limit_violation']
    finally:env.close()


@pytest.mark.parametrize('count',[2,3,4,5])
def test_site_scenes_keep_complete_sweeps_inside_bay(count):
    runtime=load_runtime();case=load_json(ROOT/f'checks_site/count{count}.json')
    validate_placement(runtime['configuration'],case)
    site=runtime['configuration']['construction_site']
    for box,motion in zip(case['scene']['obstacles'],case['scene']['obstacle_motion']):
        half=box['dimensions'][2]/2+motion['extent'][2]
        assert motion['anchor'][2]-half>=site['floor_z']+.015-1e-12
        assert motion['anchor'][2]+half<=site['ceiling_z']-.015+1e-12
        assert .04<=motion['speed_m_s']<=.22


def test_placement_rejects_floor_overlap_of_swept_box():
    runtime=load_runtime();case=copy.deepcopy(load_json(ROOT/'checks_site/count2.json'))
    box=case['scene']['obstacles'][0];motion=case['scene']['obstacle_motion'][0]
    dz=.11-motion['anchor'][2]
    motion['anchor'][2]+=dz;motion['reference'][2]+=dz;box['center'][2]+=dz
    with pytest.raises((ValueError,RuntimeError)):
        validate_placement(runtime['configuration'],case)


def test_hidden_visual_details_do_not_remove_physical_fixtures():
    config=load_runtime()['configuration'];config['construction_site']['visual_details']=False
    env=WallPanelEnv(config)
    try:
        assert len(env.site_collision_ids)==4
        assert env.collision.is_configuration_valid(env.goal_q)
        env.set_configuration(np.zeros(7))
        assert 'ceiling' in env.collision.site_contact_types()
    finally:env.close()
