"""Regression checks without calling PPO.learn or performing optimizer updates."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl import mixed_curriculum as mixed
from wall_panel_rl.utils.config import load_config, ROOT
from wall_panel_rl.utils.metrics import capture_scenario


@pytest.fixture(scope='module')
def scene():
    env = WallPanelEnv(load_config(ROOT/'configs/ppo_dynamic_clean.yaml'))
    try:
        env.reset(seed=10002)
        return capture_scenario(env, 10002)
    finally:
        env.close()


def test_curriculum_switches_static_dynamic_and_alignment(scene):
    env = mixed.MixedCurriculumEnv(load_config(ROOT/'configs/ppo_dynamic_clean.yaml'),
        [scene], {'alignment': .3, 'static': .3, 'dynamic': .4}, [.02, .15], 42)
    try:
        for task in ('static', 'dynamic', 'alignment', 'dynamic'):
            env.probabilities = np.array([float(k == task) for k in mixed.TASKS])
            obs, info = env.reset(seed=42)
            assert obs.shape == (70,) and info['curriculum_task'] == task
            before = np.array([center.copy() for center, _ in env.unwrapped.obstacles])
            obs, _, _, _, info = env.step(np.zeros(7))
            after = np.array([center for center, _ in env.unwrapped.obstacles])
            slots = obs[20:].reshape(5,10)
            if task == 'static':
                assert np.array_equal(before, after)
                assert np.all(slots[:,6:9] == 0)
            elif task == 'dynamic':
                assert len(after) == 5 and not np.array_equal(before, after)
                assert np.any(slots[:,6:9] != 0)
            else:
                assert len(after) == 0 and np.all(slots == 0)
                assert not info['collision']
        # Selecting curriculum tasks and noise streams is reproducible across reset(seed=...).
        env.probabilities = np.array([.3,.3,.4])
        a, ia = env.reset(seed=123)
        b, ib = env.reset(seed=123)
        assert ia['curriculum_task'] == ib['curriculum_task'] and np.array_equal(a,b)
        assert len(scene['obstacles']) == 5  # Shared frozen bank is never mutated.
    finally:
        env.close()


def test_timeout_metrics_exclude_collisions_and_successes():
    def row(success=False, collision=False, limit=False, pos=0., angle=0.):
        return dict(success=success, collision=collision, joint_limit_violation=limit,
                    position_error_m=pos, orientation_error_deg=angle, cumulative_reward=0.)
    result = mixed.summarize([row(success=True), row(collision=True,pos=1.,angle=90),
                              row(pos=.04,angle=8), row(limit=True,pos=2.,angle=100)])
    assert result['success_rate'] == result['collision_rate'] == result['timeout_rate'] == .25
    assert result['joint_limit_rate'] == .25
    assert result['timeout_mean_position_error_m'] == .04
    assert result['timeout_mean_orientation_error_deg'] == 8
    assert mixed.summarize([row(success=True)])['timeout_mean_position_error_m'] is None


def test_validation_uses_post_update_and_reloads_final_weights(tmp_path, monkeypatch):
    # Simulate SB3's callback ordering only; no neural network or optimizer is trained.
    class SavedModel:
        def __init__(self, steps, version):
            self.num_timesteps, self.version = steps, version

        def save(self, path):
            Path(path).write_text(json.dumps({'steps': self.num_timesteps, 'version': self.version}))

    observed = []

    def evaluate(model, *_):
        observed.append((model.num_timesteps, model.version))
        metrics = {'success_rate': 0., 'collision_rate': 0., 'mean_reward': float(model.version)}
        return {key: metrics for key in ('dynamic','static','alignment_far','alignment_near')}, []

    def reload(path, **_):
        data = json.loads(Path(path).read_text())
        return SavedModel(data['steps'], data['version'])

    monkeypatch.setattr(mixed, 'validate_policy', evaluate)
    monkeypatch.setattr(mixed.PPO, 'load', reload)
    callback = mixed.PostUpdateValidation(tmp_path, {}, [], 200000, 10000, .1)
    callback.model = SavedModel(200000, 0)
    callback._on_training_start()
    callback.model.num_timesteps = 210000
    callback._on_step()
    callback._on_rollout_end()
    assert observed == [(200000, 0)]  # No validation of pre-update rollout weights.
    callback.model.version = 1
    callback._on_rollout_start()
    callback._on_rollout_start()
    assert observed == [(200000, 0), (210000, 1)]
    callback.model.num_timesteps = 220000
    callback.model.version = 2
    callback._on_training_end()
    assert observed[-1] == (220000, 2)
    selection = json.loads((tmp_path/'selection.json').read_text())
    assert selection['additional_steps'] == 20000
    assert selection['kind'] == 'final_saved_reload'
    assert json.loads((tmp_path/'best_model.zip').read_text())['version'] == 2


def test_budget_and_probability_validation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from train_mixed_pilot import check_settings
    import yaml
    study = yaml.safe_load((ROOT/'configs/ppo_mixed_pilot.yaml').read_text())
    check_settings(study)
    with pytest.raises(ValueError, match='multiples'):
        check_settings(study | {'additional_timesteps': 50001})
    with pytest.raises(ValueError, match='sum to one'):
        check_settings(study | {'mixture': {'alignment': .3, 'static': .3, 'dynamic': .5}})


def test_legacy_checkpoint_timing(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from train_dynamic_study import Progress
    saved = []
    model = SimpleNamespace(num_timesteps=0, version=0,
                            save=lambda path: saved.append((model.num_timesteps, model.version)))
    callback = Progress(tmp_path, [], {}, 'static', 100000, {})
    callback.model = model
    callback._on_training_start()
    model.num_timesteps = 50000
    callback._on_step()
    assert not saved
    model.version = 1
    callback._on_rollout_start()
    assert saved == [(50000,1)]
    model.num_timesteps = 100000
    model.version = 2
    callback._on_training_end()
    assert saved[-1] == (100000,2)
