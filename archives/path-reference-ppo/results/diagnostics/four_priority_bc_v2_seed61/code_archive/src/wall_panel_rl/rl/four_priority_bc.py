"""Four-obstacle-only continuation; earlier tasks do not gate selection."""
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
from wall_panel_rl.rl.four_obstacle_bc import read_teacher


def load_teacher(path, train_seeds, excluded):
    data = read_teacher(path)
    seeds = data['seeds'].tolist()
    if set(data['tasks'].tolist()) != {'four_static'} or len(set(seeds)) != len(seeds):
        raise ValueError('Expected one four-obstacle demonstration per training layout')
    if set(seeds) != set(train_seeds) or set(seeds) & set(excluded):
        raise ValueError('Training/held-out split mismatch')
    return data


def score(metrics, maximum_collisions):
    """Four-box success first; collision/joint checks remain physical verdicts."""
    return [int(metrics['joint_limit'] == 0 and metrics['collision'] <= maximum_collisions),
            metrics['success'], -metrics['collision'], -metrics['timeout']]


def fit(model, data, spec, output, evaluator):
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    model.policy.configure_learning(train_critic=False)
    if model.policy.imitation_prefixes != ('multi_adapter.',): raise ValueError('Wrong actor type')
    parameters = [p for name, p in model.policy.named_parameters() if name.startswith('multi_adapter.')]
    fixed = {name: p.detach().clone() for name, p in model.policy.named_parameters() if not name.startswith('multi_adapter.')}
    # A fresh optimizer deliberately discards momentum from old-task retention.
    optimizer = torch.optim.Adam(parameters, lr=spec['learning_rate'])
    rng = np.random.default_rng(spec['seed']); counter = model.num_timesteps
    obs, targets = torch.as_tensor(data['obs']), torch.as_tensor(data['actions'])
    terminal, anchors = torch.as_tensor(data['terminal_obs']), torch.as_tensor(data['terminal_actions'])
    history, best, selected = [], None, None
    def checkpoint(update):
        nonlocal best, selected
        with torch.no_grad(): mse = float((model.policy.tracking_mean(obs)-targets).square().mean())
        state = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        try: result = evaluator(model, update)
        finally: torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])
        for name, p in model.policy.named_parameters():
            if name in fixed and not torch.equal(p.detach(), fixed[name]): raise AssertionError(f'Changed frozen {name}')
        event = {'additional_bc_updates': update, 'training_mse': mse, **result}
        path = output / 'checkpoints' / f'bc_plus_{update}.zip'; path.parent.mkdir(exist_ok=True)
        model.save(path)
        torch.save({'update': update, 'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state()}, path.with_suffix('.bc_state.pt'))
        if best is None or tuple(event['score']) > best:
            best = tuple(event['score']); selected = copy.deepcopy(event); model.save(output / 'selected_model.zip')
        history.append(event); (output / 'history.json').write_text(json.dumps(history, indent=2))
        print('FOUR_PRIORITY_CHECK', json.dumps(event), flush=True)
    checkpoint(0)
    for update in range(1, spec['updates'] + 1):
        ids = rng.integers(0, len(data['seeds']), spec['batch_size'])
        lo, hi = data['offsets'][ids], data['offsets'][ids + 1]
        rows = lo + (rng.random(len(ids)) * (hi-lo)).astype(int)
        x, target = obs[rows], targets[rows]
        with torch.no_grad():
            importance = 1 + spec['correction_weight'] * (model.policy.base_mean(x)-target).square().mean(1).clamp(0, 1)
        imitation = (importance * (model.policy.tracking_mean(x)-target).square().mean(1)).sum() / importance.sum()
        anchor_loss = (model.policy.tracking_mean(terminal[ids])-anchors[ids]).square().mean()
        regularizer = model.policy.adapter_correction(x).square().mean()
        loss = imitation + spec['terminal_weight'] * anchor_loss + spec['adapter_weight'] * regularizer
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 1.); optimizer.step()
        with (output / 'loss.jsonl').open('a') as f:
            f.write(json.dumps({'update': update, 'loss': float(loss.detach()), 'imitation': float(imitation.detach())}) + '\n')
        if update in spec['checkpoints']: checkpoint(update)
    if model.num_timesteps != counter: raise AssertionError('PPO counter changed during BC')
    model.save(output / 'final_model.zip')
    result = {'updates': spec['updates'], 'selected': selected, 'history': history,
              'four_task_sampling_probability': 1., 'old_two_used_for_selection': False,
              'frozen_base_critic_and_std_unchanged': True, 'ppo_steps': 0, 'fresh_bc_optimizer': True}
    (output / 'metadata.json').write_text(json.dumps(result, indent=2))
    return result
