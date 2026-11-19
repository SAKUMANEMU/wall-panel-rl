import numpy as np
import pytest
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.config import load_config
from wall_panel_rl.envs.geometry import rotation_error


@pytest.fixture
def env():
    e = WallPanelEnv()
    yield e
    e.close()


def test_shapes_and_physical_success(env):
    obs, _ = env.reset(seed=42)
    assert obs.shape == (41,) and obs.dtype == np.float32
    assert env.action_space.shape == (7,)
    for _ in range(60):
        action = np.clip((env.goal_q - env.joint_state()[0]) / env.max_increment, -1, 1)
        obs, reward, done, truncated, info = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if done or truncated:
            break
    assert info["success"] and not info["collision"]
    assert (
        np.linalg.norm(np.array(env.panel_pose()[0]) - env.kinematic_panel_pose()[0])
        < 0.001
    )


def test_limit_penalty(env):
    env.reset(seed=42)
    q = env.goal_q.copy()
    q[0] = env.upper[0] - 0.001
    env.set_configuration(q)
    action = np.zeros(7)
    action[0] = 1
    _, _, done, _, info = env.step(action)
    assert done and info["joint_limit_violation"] and info["reward_joint_limit"] < 0


def test_reward_progress(env):
    env.reset(seed=42)
    a = np.clip((env.goal_q - env.joint_state()[0]) / env.max_increment, -1, 1)
    _, toward, _, _, ti = env.step(a)
    env.reset(seed=42)
    _, away, _, _, ai = env.step(-a)
    assert ti["reward_position_progress"] > ai["reward_position_progress"]
    assert toward > away


def test_deterministic_noise_separate_from_scene():
    clean = load_config()
    clean["obstacles"]["stage"] = "C"
    noisy = load_config()
    noisy["obstacles"]["stage"] = "C"
    noisy["noise"]["position_std"] = 0.005
    a, b = WallPanelEnv(clean), WallPanelEnv(noisy)
    try:
        oa, _ = a.reset(seed=23)
        ob, _ = b.reset(seed=23)
        assert np.allclose(a.joint_state()[0], b.joint_state()[0])
        assert np.allclose(np.array(a.obstacles), np.array(b.obstacles))
        assert not np.allclose(oa, ob)
        ob2, _ = b.reset(seed=23)
        assert np.array_equal(ob, ob2)
    finally:
        a.close()
        b.close()


def test_quaternion_sign():
    q = np.array([0, 0.3, 0, np.sqrt(0.91)])
    assert np.linalg.norm(rotation_error(q, -q)) < 1e-12
