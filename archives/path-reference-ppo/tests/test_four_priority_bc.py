"""Four-only gradients, held-out isolation and success-based selection."""
import copy
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from wall_panel_rl.rl.avoidance_imitation import save_teacher
from wall_panel_rl.rl.four_priority_bc import load_teacher, fit, score, sample_rows
from wall_panel_rl.utils.config import ROOT


def dataset(tmp_path, model):
    x = np.zeros((2, 77), np.float32); x[:, :7] = -.1; x[:, 70:] = -.05
    for i in range(4): x[:, 20+i*10:29+i*10] = .1*(i+1); x[:, 29+i*10] = 1
    with torch.no_grad(): y = (model.policy.tracking_mean(torch.as_tensor(x)).numpy() + .1).clip(-1, 1)
    path = tmp_path / 'teacher.npz'
    save_teacher(path, x, y, y, [0, 1, 2], [1, 2], ['four_static']*2, x, y)
    return path


def test_four_only_real_update_keeps_frozen_state_counter_and_roundtrip(tmp_path):
    torch.set_num_threads(1)
    model = PPO.load(ROOT / 'results/diagnostics/four_obstacle_bc_v1_seed61/final_candidate.zip', device='cpu')
    path = dataset(tmp_path, model)
    with pytest.raises(ValueError, match='split'): load_teacher(path, [1, 2], [2])
    with pytest.raises(ValueError, match='split'): load_teacher(path, [1], [])
    data = load_teacher(path, [1, 2], [3, 4])
    before = copy.deepcopy(model.policy.state_dict()); counter = model.num_timesteps
    spec = {'updates': 3, 'checkpoints': [3], 'batch_size': 16, 'learning_rate': .001, 'seed': 61,
            'terminal_weight': 0., 'adapter_weight': 0., 'correction_weight': 0.}
    fit(model, data, spec, tmp_path / 'fit', lambda m, n: {'score': [n]})
    reloaded = PPO.load(tmp_path / 'fit/selected_model.zip', device='cpu')
    assert reloaded.num_timesteps == counter
    after = reloaded.policy.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith('multi_adapter.'))
    assert any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith('multi_adapter.'))


def test_priority_ranks_four_success_without_old_two_dependency():
    def m(s, c, t=0, j=0): return {'success': s, 'collision': c, 'timeout': t, 'joint_limit': j}
    assert score(m(20, 12), 15) > score(m(17, 15), 15)
    assert score(m(20, 11, 1), 15) > score(m(20, 12), 15)
    assert score(m(21, 10, 0, 1), 15) < score(m(17, 15), 15)
    assert score(m(10, 16, 6), 15)[0] == 0


def test_check_only_never_runs_training(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import run_four_priority_bc as runner
    monkeypatch.setattr(runner, 'run', lambda *_: (_ for _ in ()).throw(AssertionError('Unexpected training')))
    monkeypatch.setattr('sys.argv', ['run_four_priority_bc.py'])
    runner.main()
    assert 'check_only' in capsys.readouterr().out


def test_corrective_sampling_keeps_episode_boundaries_and_full_route_coverage():
    data = {'offsets': np.array([0, 4, 20, 30])}
    ids = np.tile([0, 1, 2], 1000)
    focused = sample_rows(data, ids, np.random.default_rng(61), 1., 8)
    assert np.all(focused >= data['offsets'][ids])
    assert np.all(focused < data['offsets'][ids] + np.minimum(8, np.diff(data['offsets'])[ids]))
    mixed = sample_rows(data, ids, np.random.default_rng(61), .6, 8)
    assert np.all(mixed < data['offsets'][ids+1]) and np.all(mixed >= data['offsets'][ids])
    assert np.any(mixed[ids == 1] >= 12)
    with pytest.raises(ValueError): sample_rows(data, ids, np.random.default_rng(61), 1.1)


def test_corrective_duplicates_require_explicit_opt_in_and_still_exclude_heldout(tmp_path):
    model = PPO.load(ROOT / 'results/diagnostics/four_obstacle_bc_v1_seed61/final_candidate.zip', device='cpu')
    path = dataset(tmp_path, model)
    with np.load(path) as z: d = {k: z[k].copy() for k in z.files}
    d['seeds'][:] = 1
    np.savez_compressed(path, **d)
    with pytest.raises(ValueError): load_teacher(path, [1], [])
    assert len(load_teacher(path, [1], [], allow_corrections=True)['seeds']) == 2
    with pytest.raises(ValueError): load_teacher(path, [1], [1], allow_corrections=True)
