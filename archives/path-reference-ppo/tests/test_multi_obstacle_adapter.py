"""Independent adapter learns without changing the original one-box actor."""
import copy
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.multi_obstacle_adapter import initialize_adapter, frozen_base_preserved
from wall_panel_rl.rl.avoidance_imitation import save_teacher
from wall_panel_rl.rl.multi_obstacle_warmup import warmup_actor


def model():
    torch.set_num_threads(1)
    return initialize_adapter(ROOT/'results/diagnostics/episode_rrt_ppo_v1_seed61/selected_model.zip', {}, 61)


def observations():
    x = torch.zeros(3, 77)
    x[:, :7] = -.1; x[:, 70:] = -.05
    x[1:, 20:29] = .2; x[1:, 29] = 1
    x[2, 30:39] = .4; x[2, 39] = 1
    return x


def test_independent_adapter_updates_second_obstacle_response_preserves_other_actions_and_reload(tmp_path):
    m = model(); before = copy.deepcopy(m.policy); x = observations()
    with torch.no_grad(): initial = m.policy.tracking_mean(x).clone()
    assert torch.equal(initial, before.base_mean(x))
    params = [p for p in m.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=.001)
    # Supervise the TWO-obstacle row only; other rows must remain exact anyway.
    for _ in range(6):
        loss = (m.policy.tracking_mean(x)[2] - (initial[2] + .1).clamp(-1, 1)).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        after = m.policy.tracking_mean(x)
        shifted = x.clone(); shifted[2, 30:33] += .15
        response = (m.policy.tracking_mean(shifted)[2]-after[2]).abs().max()
    assert torch.equal(initial[:2], after[:2])
    assert not torch.equal(initial[2], after[2]) and response > 0
    assert frozen_base_preserved(before, m.policy)
    m.save(tmp_path/'model.zip'); loaded = PPO.load(tmp_path/'model.zip', device='cpu')
    with torch.no_grad(): assert torch.equal(after, loaded.policy.tracking_mean(x))
    assert all(not p.requires_grad for name, p in loaded.policy.named_parameters()
               if not name.startswith(('multi_adapter.', 'mlp_extractor.value_net.', 'value_net.')))


def test_actual_ppo_update_preserves_frozen_actor_and_one_box_mean(tmp_path):
    class Toy(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)
        action_space = spaces.Box(-1, 1, (7,), np.float32)
        def reset(self, *, seed=None, options=None):
            self.steps = 0; return observations()[2].numpy(), {}
        def step(self, action):
            self.steps += 1
            return observations()[2].numpy(), float(-np.square(action).mean()), self.steps == 4, False, {}
    m = model(); m.save(tmp_path/'initial.zip'); env = DummyVecEnv([Toy])
    try:
        candidate = PPO.load(tmp_path/'initial.zip', env=env, device='cpu', n_steps=4, batch_size=4,
                             n_epochs=1, learning_rate=.001)
        before = copy.deepcopy(candidate.policy)
        candidate.policy.configure_learning(train_critic=True)
        candidate.policy.optimizer = torch.optim.Adam(candidate.policy.parameters(), lr=.001)
        candidate.learn(8, reset_num_timesteps=False)
        assert frozen_base_preserved(before, candidate.policy)
        assert any(not torch.equal(p, dict(candidate.policy.named_parameters())[n])
                   for n, p in before.named_parameters() if n.startswith('multi_adapter.'))
        with torch.no_grad():
            assert torch.equal(before.tracking_mean(observations()[:2]), candidate.policy.tracking_mean(observations()[:2]))
    finally: env.close()


def test_ppo_gate_requires_learned_validation_gain(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from run_two_static_adapter import ppo_learning_gate, correction_handoffs
    initial = {'two': {'stage_target': {'success': 15, 'collision': 4}}}
    selected = dict(initial, update=0, ppo_ready=True,
                    second_slot_sensitivity={'max_action_change_15cm': .01})
    assert not ppo_learning_gate(selected, initial)
    selected['update'] = 100
    assert not ppo_learning_gate(selected, initial)
    selected['two'] = {'stage_target': {'success': 15, 'collision': 3}}
    assert ppo_learning_gate(selected, initial)
    selected['second_slot_sensitivity']['max_action_change_15cm'] = 0.
    assert not ppo_learning_gate(selected, initial)
    assert correction_handoffs(60, [12, 24]) == [35, 47]
    assert correction_handoffs(3, [12, 24]) == [0]


def test_bc_and_auxiliary_use_only_adapter_and_exclude_heldout_labels(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    from train_two_static_ppo import SeededImitation
    from types import SimpleNamespace
    states = []
    for index, held_label in enumerate((-.8, .8)):
        candidate = model(); before = copy.deepcopy(candidate.policy)
        x = observations(); obs, labels, seeds, tasks, offsets = [], [], [], [], [0]
        with torch.no_grad(): initial = before.tracking_mean(x).numpy()
        for seed in (1, 2):
            for row, group in ((2, 'two_static'), (1, 'one_static'), (0, 'alignment')):
                obs.append(x[row].numpy()); seeds.append(seed); tasks.append(group); offsets.append(len(obs))
                labels.append((initial[row]+.03).clip(-1, 1) if seed == 1 else np.full(7, held_label))
        directory = tmp_path/f'teacher{index}'; directory.mkdir()
        save_teacher(directory/'teacher.npz', obs, labels, labels, offsets, seeds, tasks, obs, labels)
        spec = {'updates': 3, 'checkpoints': [3], 'batch_size': 32, 'learning_rate': .001,
            'task_weights': {'two_static': .8, 'one_static': .1, 'alignment': .1}, 'seed': 61,
            'swap_two_obstacles_probability': 0., 'terminal_weight': 0., 'gate_retention_weight': 0.,
            'residual_weight': 0., 'correction_weight': 1.}
        path, _ = warmup_actor(candidate, directory/'teacher.npz', {2}, spec, tmp_path/f'run{index}',
                              lambda m, update: {'score': [update]})
        trained = PPO.load(path, device='cpu')
        assert frozen_base_preserved(before, trained.policy)
        with torch.no_grad():
            assert torch.equal(before.tracking_mean(x[:2]), trained.policy.tracking_mean(x[:2]))
            assert not torch.equal(before.tracking_mean(x[2:]), trained.policy.tracking_mean(x[2:]))
        states.append(copy.deepcopy(trained.policy.state_dict()))
        callback = SeededImitation(tmp_path/f'aux{index}',
            {'capacity': 100, 'batch_size': 2, 'checkpoint_every': 5000, 'updates_per_rollout': 1,
             'weight': .2, 'learning_rate': 1e-5, 'maximum_kl': .005},
            teacher_path=directory/'teacher.npz', heldout={2})
        callback.init_callback(SimpleNamespace(policy=trained.policy, num_timesteps=40000))
        callback._on_training_start()
        assert len(callback.pool) == 3
        expected = {id(p) for n, p in trained.policy.named_parameters() if n.startswith('multi_adapter.')}
        assert {id(p) for p in callback.params} == expected
    assert all(torch.equal(v, states[1][name]) for name, v in states[0].items())
