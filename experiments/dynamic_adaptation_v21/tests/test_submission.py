"""Public-interface contracts and dataset separation for the portable submission."""
import numpy as np
from stable_baselines3.common.env_checker import check_env
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import CaseBankEnv, load_json, load_agent, run_episode


def test_dataset_separation_and_saved_checkpoint():
    split=load_json(ROOT/'data/split.json')
    assert not set(split['train']) & set(split['validation'])
    assert not set(split['demo']) & (set(split['train']) | set(split['validation']))
    model=load_agent()
    assert model.observation_space.shape==(77,) and model.action_space.shape==(7,)


def test_gymnasium_interface_and_finite_actions():
    env=CaseBankEnv(ROOT/'data/train_cases.json')
    try:
        check_env(env,warn=True)
        observation,_=env.reset(seed=2026)
        assert observation.shape==(77,) and observation.dtype==np.float32
        action,_=load_agent().predict(observation,deterministic=True)
        observation,reward,done,truncated,info=env.step(action)
        assert action.shape==(7,) and np.isfinite(action).all()
        assert observation.shape==(77,) and np.isfinite(reward)
    finally:
        env.close()


def test_recorded_demo_is_real_full_episode_and_excerpt_is_labeled():
    full=load_json(ROOT/'demo/vertical_corridor_1166026.json')
    excerpt=load_json(ROOT/'demo/vertical_corridor_1166026_motion_excerpt.json')
    assert full['result']['reason']=='success' and full['result']['steps']==239
    assert len(full['trajectory'])==240
    assert excerpt['excerpt']['first_real_step']==155
    assert excerpt['result']==full['result']
    assert excerpt['trajectory']==full['trajectory'][155:]
