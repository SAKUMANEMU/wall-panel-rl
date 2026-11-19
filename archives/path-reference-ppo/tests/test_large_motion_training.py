"""Physical bounds and invariant reference for the enhanced-motion curriculum."""
import copy,json
import numpy as np
import pybullet as pb
import pytest
import yaml
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors
from wall_panel_rl.rl.slow_dynamic_four import dynamic_configuration,validate_placement
from wall_panel_rl.rl.dynamic_four_training import case_at_speed,DynamicFourBankEnv
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv

@pytest.fixture
def setup():
    spec=yaml.safe_load((ROOT/'configs/four_large_motion_training.yaml').read_text())
    source=json.loads((ROOT/spec['source_report']/'candidate_manifest.json').read_text())
    profile=yaml.safe_load((ROOT/spec['motion_profile']).read_text())
    config,settings=dynamic_configuration(source['configuration'],source['evaluation_settings'],profile)
    old=json.loads((ROOT/spec['source_train_cases']).read_text())[0]
    return spec,profile,config,settings,old

def test_reanchor_preserves_reference_and_has_actual_physical_witnesses(setup):
    spec,profile,config,settings,old=setup;saved=copy.deepcopy(old)
    new=fit_motion_anchors(config,settings,profile,old)
    assert old==saved and new==fit_motion_anchors(config,settings,profile,old)
    for key in ['initial_q','reference_joint_path','nominal_motion','obstacle_original_indices']:
        assert new['scene'][key]==old['scene'][key]
    assert [b['dimensions'] for b in new['scene']['obstacles']]==[b['dimensions'] for b in old['scene']['obstacles']]
    validate_placement(config,new)
    env=WallPanelEnv(config)
    try:
        empty=copy.deepcopy(new['scene']);empty.update(obstacles=[],obstacle_motion=[])
        env.reset(options={'scenario':empty})
        for box,motion,w in zip(new['scene']['obstacles'],new['scene']['obstacle_motion'],new['scene']['motion_anchor_placement']['obstacles']):
            body=env.box(box['dimensions'],motion['anchor'],[1,0,0,0])
            with env.collision.at_configuration(w['nominal_joint_position']):
                assert pb.getClosestPoints(env.panel,body,0,physicsClientId=env.client) or pb.getClosestPoints(env.robot,body,0,physicsClientId=env.client)
            pb.removeBody(body,physicsClientId=env.client)
    finally:env.close()

def test_motion_norms_are_exactly_scaled_and_episode_change_is_deferred(setup):
    spec,profile,config,settings,old=setup;new=fit_motion_anchors(config,settings,profile,old)
    for a,v in zip(spec['curriculum_amplitudes'],spec['curriculum_speeds']):
        case=case_at_speed(new,v,amplitude=a)
        for box,m in zip(case['scene']['obstacles'],case['scene']['obstacle_motion']):
            assert np.isclose(np.linalg.norm(m['extent']),a)
            assert np.linalg.norm(np.array(m['extent'])*m['omega'])<=v+1e-12
            np.testing.assert_allclose(box['center'],np.array(m['anchor'])+np.array(m['extent'])*np.sin(m['phase']))
        validate_placement(config,case)
    spec=copy.deepcopy(spec);spec['static_probability']=0
    env=DynamicFourBankEnv(config,settings,[new],spec,seed=81)
    try:
        env.reset();before=copy.deepcopy(env.last_scene);env.set_curriculum(2)
        assert env.amplitude==.1 and env.speed==.05 and env.last_scene==before
        env.reset();assert env.amplitude==.15 and env.speed==.1
    finally:env.close()
    with pytest.raises(ValueError):case_at_speed(new,.1,amplitude=.16)

def test_impossible_geometry_is_rejected_without_shrinking(setup):
    _,profile,config,settings,old=setup;bad=copy.deepcopy(old)
    for box in bad['scene']['obstacles']:box['dimensions']=[20.,20.,20.]
    saved=copy.deepcopy(bad)
    with pytest.raises(RuntimeError,match='Could not fit'):
        fit_motion_anchors(config,settings,dict(profile,anchor_restarts=1),bad)
    assert bad==saved
