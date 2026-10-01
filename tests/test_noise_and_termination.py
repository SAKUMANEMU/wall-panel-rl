import numpy as np
import pytest
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.envs.noise import observe_target
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.envs.geometry import rotation_error


def test_noise_statistics_and_unchanged_truth():
    pose = (np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))
    rng = np.random.default_rng(12)
    samples = [
        observe_target(pose, rng, {"position_std": 0.005, "orientation_std_deg": 0.5})
        for _ in range(10000)
    ]
    pos = np.array([a for a, b in samples])
    ori = np.array([rotation_error(b, pose[1]) for a, b in samples])
    assert np.allclose(pos.std(axis=0), 0.005, rtol=0.04)
    assert np.allclose(ori.std(axis=0), np.deg2rad(0.5), rtol=0.04)
    assert np.array_equal(pose[0], np.zeros(3))


def test_timeout_not_terminal_and_bad_action():
    c = load_config()
    c["simulation"]["max_steps"] = 1
    e = WallPanelEnv(c)
    try:
        e.reset(seed=42)
        _, _, done, truncated, info = e.step(np.zeros(7))
        assert truncated and not done
        with pytest.raises(ValueError):
            e.step(np.full(7, np.nan))
        with pytest.raises(ValueError):
            e.step(np.zeros(6))
    finally:
        e.close()


def test_collision_penalty_and_obstacle_padding():
    e = WallPanelEnv()
    try:
        obs, _ = e.reset(seed=42)
        assert np.array_equal(obs[-21:], np.zeros(21))
        e.obstacle_ids.append(e.box([0.3, 0.3, 0.3], e.panel_pose()[0], [1, 0, 0, 1]))
        _, _, done, truncated, info = e.step(np.zeros(7))
        assert done and not truncated and info["collision"] and not info["success"]
        assert info["reward_collision"] == -50
    finally:
        e.close()


def test_unseeded_resets_continue_seeded_noise_stream():
    c = load_config()
    c['noise']['position_std'] = .005
    c['noise']['orientation_std_deg'] = .5
    a, b = WallPanelEnv(c), WallPanelEnv(c)
    try:
        a.reset(seed=123)
        b.reset(seed=123)
        for _ in range(4):
            oa, _ = a.reset()
            ob, _ = b.reset()
            assert np.array_equal(oa, ob)
    finally:
        a.close()
        b.close()


def test_evaluation_disables_training_reset_mixture():
    from wall_panel_rl.utils.config import ROOT
    for condition in ['clean', 'noise']:
        training = load_config(ROOT / f'configs/train_{condition}.yaml')
        evaluation = load_config(ROOT / f'configs/eval_{condition}.yaml')
        assert training['robot']['near_start_probability'] == .25
        assert evaluation['robot']['near_start_probability'] == 0
