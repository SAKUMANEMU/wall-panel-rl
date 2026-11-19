"""PPO-only reset bank, unchanged actions, reward units and real optimizer smoke."""
import copy
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl.four_priority_ppo import FourPolicyTrainingEnv
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES
from wall_panel_rl.utils.config import ROOT


class Toy(gym.Env):
    observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)
    action_space = spaces.Box(-1, 1, (7,), np.float32)
    def reset(self, *, seed=None, options=None):
        self.steps = 0; self.options = options; x = np.zeros(77, np.float32)
        x[:7] = -.1; x[70:] = -.05
        for i in range(4): x[20+i*10:29+i*10] = .2; x[29+i*10] = 1
        self.x = x; return x.copy(), {}
    def step(self, action):
        self.action = np.asarray(action).copy(); self.steps += 1
        return self.x.copy(), 10., self.steps == 2, False, {'termination_reason': 'success'}


def cases():
    return [{'scene': {'seed': i, 'obstacles': [{}, {}, {}, {}], 'obstacle_motion': []}, 'strength': [.05, .02]} for i in (1, 2)]


def test_frozen_bank_action_identity_reward_scale_and_journal(tmp_path):
    raw = Toy(); bank = cases(); saved = copy.deepcopy(bank)
    with pytest.raises(ValueError): FourPolicyTrainingEnv(raw, bank, excluded=[1])
    wrapped = FourPolicyTrainingEnv(raw, bank, excluded=[99], reward_scale=.01, journal=tmp_path/'episodes.jsonl')
    wrapped.reset(seed=61); assert raw.options['scenario']['seed'] in (1, 2) and raw.options['task'] == 'static'
    action = np.full(7, .1, np.float32)
    _, reward, done, _, _ = wrapped.step(action)
    assert reward == .1 and not done and np.array_equal(raw.action, action)
    _, reward, done, _, info = wrapped.step(action)
    assert done and info['episode']['r'] == .2 and wrapped.records[0]['raw_return'] == 20.
    assert wrapped.records[0]['controller'] == 'ppo' and bank == saved
    with pytest.raises(ValueError): wrapped.reset(options={'scenario': {}})


def test_actual_four_ppo_update_keeps_base_actor_and_std(tmp_path):
    torch.set_num_threads(1); env = DummyVecEnv([lambda: FourPolicyTrainingEnv(Toy(), cases())])
    try:
        m = PPO.load(ROOT/'results/diagnostics/four_obstacle_bc_v1_seed61/final_candidate.zip', env=env,
            device='cpu', n_steps=4, batch_size=4, n_epochs=1, learning_rate=.001)
        before = copy.deepcopy(m.policy.state_dict()); start = m.num_timesteps
        m.policy.configure_learning(train_critic=True)
        m.policy.optimizer = torch.optim.Adam(m.policy.parameters(), lr=.001)
        m.learn(8, reset_num_timesteps=False)
        after = m.policy.state_dict()
        assert m.num_timesteps-start == 8
        assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith(('multi_adapter.', *CRITIC_PREFIXES)))
        assert any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith('multi_adapter.'))
        assert all(r['seed'] in (1, 2) and r['controller'] == 'ppo' for r in env.envs[0].records)
    finally: env.close()


def test_cli_default_cannot_launch_ppo(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts')); import train_four_priority_ppo as runner
    monkeypatch.setattr(runner, 'run', lambda *_: (_ for _ in ()).throw(AssertionError('Unexpected learning')))
    monkeypatch.setattr('sys.argv', ['train_four_priority_ppo.py']); runner.main()
    assert 'check_only' in capsys.readouterr().out
