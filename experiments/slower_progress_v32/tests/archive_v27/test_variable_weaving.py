"""Variable-count masks, dynamic physics and genuine alternating passage checks."""
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from stable_baselines3.common.env_checker import check_env
from wall_panel_rl.submission import load_json, load_runtime, load_agent
from wall_panel_rl.rl.variable_weaving import VariableCaseBankEnv, VariableDynamicEnv
from wall_panel_rl.visualization.passage_metrics import passage_measurements
from wall_panel_rl.utils.config import ROOT


@pytest.mark.parametrize('count', [2, 3, 4, 5])
def test_count_mask_and_prescribed_motion(count):
    rt = load_runtime()
    case = load_json(ROOT/f'checks_variable/count{count}.json')
    env = VariableDynamicEnv(rt['configuration'], [], rt['settings'])
    try:
        obs, _ = env.reset(options={'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']})
        slots = obs[20:70].reshape(5, 10)
        assert np.all(slots[:count, 9] == 1)
        assert np.all(slots[count:] == 0)
        before = np.array([c for c, _ in env.env.obstacles])
        action = load_agent().predict(obs, deterministic=True)[0]
        assert action.shape == (7,) and np.isfinite(action).all()
        env.step(np.zeros(7, np.float32))
        after = np.array([c for c, _ in env.env.obstacles])
        assert not np.allclose(before, after)
        assert env.env.sim_time > 0
    finally:
        env.close()


def test_bank_gym_interface_and_seeded_count_sampling():
    rt = load_runtime()
    cases = [load_json(ROOT/f'checks_variable/count{n}.json') for n in (2, 3, 4, 5)]
    spec = yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())['training']
    env = VariableCaseBankEnv(rt, cases, spec, seed=61)
    try:
        check_env(env, warn=True)
        def sequence():
            result = [env.reset(seed=918)[1]['obstacle_count']]
            result += [env.reset()[1]['obstacle_count'] for _ in range(31)]
            return result
        first = sequence()
        assert sequence() == first
        assert set(first) == {2, 3, 4, 5}
    finally:
        env.close()


@pytest.mark.parametrize('count', [2, 3, 4, 5])
def test_measured_passages_support_all_counts(count):
    boxes = [{'center': [i, 0, (.5 if i % 2 else .8)], 'dimensions': [.1, .1, .1]} for i in range(count)]
    frames = []
    for i in range(count):
        for _ in range(3):
            frames.append({'time_s': len(frames)*.05, 'panel': [i, 0, (.8 if i % 2 else .5)],
                           'panel_orientation': [0, 0, 0, 1], 'obstacle_centers': [b['center'] for b in boxes]})
    payload = {'trajectory': frames, 'configuration': {'panel': {'dimensions': [.3, .2, .03]}},
               'case': {'scene': {'obstacles': boxes, 'motion_anchor_placement': {'route_slot_order': list(range(count))}}}}
    result = passage_measurements(payload)
    assert result['passage_count'] == count
    assert result['ordered_alternating_passages'][0]['side'] == 'below'


def test_generation_has_real_anchor_and_timed_witnesses():
    """Reject a proxy-only interpretation of 'on the route'."""
    import pybullet as p
    from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
    rt = load_runtime()
    for count in (2, 3, 4, 5):
        case = load_json(ROOT/f'checks_variable/count{count}.json')
        scene = case['scene']
        env = WallPanelEnv(rt['configuration'])
        try:
            env.reset(seed=scene['seed'], options={'scenario': scene})
            timed_count = 0
            for slot, record in enumerate(scene['motion_anchor_placement']['obstacles']):
                motion = scene['obstacle_motion'][slot]
                center = np.array(motion['anchor'])
                if record['timed_conflict']:
                    center += np.array(motion['extent'])*np.sin(np.array(motion['phase'])+np.array(motion['omega'])*record['guide_time_s'])
                    timed_count += 1
                body = env.obstacle_ids[slot]
                p.resetBasePositionAndOrientation(body, center.tolist(), [0,0,0,1], physicsClientId=env.client)
                with env.collision.at_configuration(record['nominal_joint_position']):
                    assert (p.getClosestPoints(env.panel, body, 0, physicsClientId=env.client) or
                            p.getClosestPoints(env.robot, body, 0, physicsClientId=env.client))
            assert timed_count == 1
        finally:
            env.close()
