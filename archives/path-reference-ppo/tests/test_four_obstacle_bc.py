"""Four-box BC isolation, frozen skills, serialization and safety selection."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from wall_panel_rl.rl.avoidance_imitation import save_teacher
from wall_panel_rl.rl.four_obstacle_bc import make_pool, fit_actor, validation_score, read_teacher
from wall_panel_rl.utils.config import ROOT


def load_model():
    torch.set_num_threads(1)
    return PPO.load(ROOT / 'results/diagnostics/four_static_on_path_v1_seed61/selected_model.zip', device='cpu')


def observation(count):
    x = np.zeros(77, np.float32); x[:7] = -.1; x[70:] = -.05
    for i in range(count): x[20+i*10:29+i*10] = .1 * (i+1); x[29+i*10] = 1
    return x


def teachers(root, model, excluded_label):
    root.mkdir()
    x = observation(4)
    with torch.no_grad(): y = (model.policy.tracking_mean(torch.as_tensor(x[None])).numpy()[0] + .1).clip(-1, 1)
    save_teacher(root / 'new.npz', [x, x], [y, y], [y, y], [0, 2], [1], ['four_static'], [x], [y])
    specs = [(2, 'two_static', 2), (3, 'one_static', 1), (4, 'alignment', 0), (9, 'two_static', 2)]
    xs = [observation(n) for _, _, n in specs]
    ys = [np.full(7, excluded_label if seed == 9 else -.7, np.float32) for seed, _, _ in specs]
    save_teacher(root / 'old.npz', xs, ys, ys, [0, 1, 2, 3, 4], [s[0] for s in specs], [s[1] for s in specs], xs, ys)
    return root / 'new.npz', root / 'old.npz'


def test_four_bc_excludes_heldout_labels_and_keeps_base_critic_std_and_one_actions(tmp_path):
    trained = []
    for j, poison in enumerate((-.8, .8)):
        model = load_model(); before = copy.deepcopy(model.policy.state_dict()); steps = model.num_timesteps
        paths = teachers(tmp_path / f'data{j}', model, poison)
        data, parts, included = make_pool(*paths, {9, 99}, model.policy)
        assert set(data['seeds']) == {1, 2, 3, 4}
        assert not any(r['seed'] == 9 for r in included)
        with torch.no_grad():
            torch.testing.assert_close(torch.as_tensor(data['actions'][2:]),
                model.policy.tracking_mean(torch.as_tensor(data['obs'][2:])), atol=1e-6, rtol=0)
            protected = model.policy.tracking_mean(torch.as_tensor(np.stack([observation(0), observation(1)])))
        spec = {'updates': 3, 'checkpoints': [3], 'batch_size': 32, 'seed': 61, 'learning_rate': .001,
                'task_weights': {'four_static': .8, 'two_static': .1, 'one_static': .05, 'alignment': .05},
                'correction_weight': 0., 'terminal_weight': 0., 'adapter_weight': 0.}
        out = tmp_path / f'fit{j}'
        fit_actor(model, data, parts, spec, out, lambda m, update: {'score': [update]})
        reloaded = PPO.load(out / 'selected_model.zip', device='cpu')
        assert reloaded.num_timesteps == steps
        after = reloaded.policy.state_dict()
        assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith('multi_adapter.'))
        assert any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith('multi_adapter.'))
        with torch.no_grad():
            assert torch.equal(protected, reloaded.policy.tracking_mean(torch.as_tensor(np.stack([observation(0), observation(1)]))))
        trained.append(after)
        with pytest.raises(ValueError, match='Held-out'):
            make_pool(*paths, {1, 9}, model.policy)
    assert all(torch.equal(v, trained[1][k]) for k, v in trained[0].items())


def test_task_presence_mismatch_is_rejected(tmp_path):
    model = load_model(); new, _ = teachers(tmp_path / 'data', model, .8)
    with np.load(new) as z: d = {k: z[k].copy() for k in z.files}
    d['obs'][:, 59] = 0
    np.savez_compressed(tmp_path / 'wrong.npz', **d)
    with pytest.raises(ValueError, match='presence'): read_teacher(tmp_path / 'wrong.npz')


def test_safety_tradeoff_is_allowed_but_timeout_collapse_and_old_skill_loss_are_rejected():
    def counts(s, c, t): return {'episodes': 32, 'success': s, 'collision': c, 'timeout': t, 'joint_limit': 0}
    baseline = {'four': counts(18, 2, 12), 'two': counts(25, 4, 3), 'one_rows_identical': True}
    safer = dict(baseline, four=counts(17, 0, 15))
    # One fewer success and two prevented collisions produce three new timeouts.
    assert validation_score(safer, baseline) > validation_score(baseline, baseline)
    safer['four'] = counts(18, 0, 14)
    assert validation_score(safer, baseline) > validation_score(baseline, baseline)
    safer['four'] = counts(17, 1, 14)
    assert validation_score(safer, baseline) > validation_score(baseline, baseline)
    assert validation_score(dict(safer, two=counts(24, 4, 4)), baseline)[0] == 0
    assert validation_score(dict(safer, four=counts(5, 0, 27)), baseline)[0] == 0
    assert validation_score(dict(safer, one_rows_identical=False), baseline)[0] == 0


def test_cli_without_run_cannot_start_fit(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    import run_four_obstacle_bc as runner
    monkeypatch.setattr(runner, 'run', lambda *_: (_ for _ in ()).throw(AssertionError('Training launched')))
    monkeypatch.setattr('sys.argv', ['run_four_obstacle_bc.py'])
    runner.main()
    assert 'check_only' in capsys.readouterr().out
