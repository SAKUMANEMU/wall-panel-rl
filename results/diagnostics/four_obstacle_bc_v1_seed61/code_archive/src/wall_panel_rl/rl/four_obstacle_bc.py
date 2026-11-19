"""Offline four-box BC with frozen-base distillation for earlier tasks.

New labels are verified RRT actions. Earlier observations are replayed against
the INITIAL policy, avoiding contradictory old RRT routes as retention targets.
Validation/test observations and actions never enter this optimization pool.
"""
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch

GROUPS = ('four_static', 'two_static', 'one_static', 'alignment')


def read_teacher(path):
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    n, episodes = len(d['obs']), len(d['seeds'])
    if (not episodes or d['obs'].shape != (n, 77) or d['actions'].shape != (n, 7)
        or d['offsets'].shape != (episodes + 1,) or d['offsets'][0] != 0
        or d['offsets'][-1] != n or np.any(np.diff(d['offsets']) <= 0)
        or d['tasks'].shape != (episodes,) or d['terminal_obs'].shape != (episodes, 77)
        or d['terminal_actions'].shape != (episodes, 7)):
        raise ValueError('Malformed teacher episode arrays')
    for key in ('obs', 'actions', 'terminal_obs', 'terminal_actions'):
        if not np.isfinite(d[key]).all(): raise ValueError('Non-finite teacher')
    if np.max(np.abs(d['actions'])) > 1.000001: raise ValueError('Action labels exceed bounds')
    for i, group in enumerate(d['tasks']):
        if group not in GROUPS: raise ValueError('Unknown BC task')
        lo, hi = d['offsets'][i:i+2]
        present = (d['obs'][lo:hi, 20:70].reshape(-1, 5, 10)[:, :, 9] > .5).sum(1)
        if not np.all(present == {'four_static': 4, 'two_static': 2, 'one_static': 1, 'alignment': 0}[group]):
            raise ValueError('Task does not match obstacle presence slots')
    return d


def make_pool(new_path, retention_path, excluded_seeds, source_policy):
    new, old = read_teacher(new_path), read_teacher(retention_path)
    excluded = set(map(int, excluded_seeds))
    if set(new['tasks'].tolist()) != {'four_static'} or len(set(new['seeds'])) != len(new['seeds']):
        raise ValueError('Expected one four-obstacle route per new training layout')
    if set(new['seeds'].tolist()) & excluded: raise ValueError('Held-out scene in new BC labels')
    if set(new['seeds'].tolist()) & set(old['seeds'].tolist()): raise ValueError('New/retained layout overlap')
    obs, actions, terminal, anchors, seeds, groups, offsets = [], [], [], [], [], [], [0]
    included = []
    for name, dataset in [('verified_four', new), ('retention', old)]:
        for i, seed in enumerate(dataset['seeds']):
            if int(seed) in excluded: continue
            group = str(dataset['tasks'][i])
            if name == 'retention' and group == 'four_static': raise ValueError('Unexpected retained task')
            lo, hi = dataset['offsets'][i:i+2]
            x = dataset['obs'][lo:hi].copy(); tx = dataset['terminal_obs'][i].copy()
            if name == 'retention':
                with torch.no_grad():
                    y = source_policy.tracking_mean(torch.as_tensor(x)).cpu().numpy()
                    ty = source_policy.tracking_mean(torch.as_tensor(tx[None])).cpu().numpy()[0]
            else: y, ty = dataset['actions'][lo:hi].copy(), dataset['terminal_actions'][i].copy()
            obs.extend(x); actions.extend(y); terminal.append(tx); anchors.append(ty)
            seeds.append(int(seed)); groups.append(group); offsets.append(len(obs))
            included.append({'source': name, 'source_episode': i, 'seed': int(seed), 'task': group})
    data = {'obs': np.asarray(obs, np.float32), 'actions': np.asarray(actions, np.float32),
            'terminal_obs': np.asarray(terminal, np.float32), 'terminal_actions': np.asarray(anchors, np.float32),
            'seeds': np.asarray(seeds), 'tasks': np.asarray(groups), 'offsets': np.asarray(offsets)}
    parts = {group: np.flatnonzero(data['tasks'] == group) for group in GROUPS}
    if any(not len(ids) for ids in parts.values()): raise ValueError('Missing training task')
    return data, parts, included


def validation_score(candidate, baseline, maximum_success_drop=2):
    """Prefer fewer collisions, allowing only a bounded success/timeout tradeoff."""
    a, b = candidate['four'], baseline['four']
    two, old_two = candidate['two'], baseline['two']
    retained = (candidate['one_rows_identical'] and two['success'] >= old_two['success']
                and two['collision'] <= old_two['collision'] and two['joint_limit'] == 0)
    bounded = (a['success'] >= b['success'] - maximum_success_drop
               and a['timeout'] <= b['timeout'] + maximum_success_drop + max(0, b['collision'] - a['collision'])
               and a['collision'] <= b['collision'] and a['joint_limit'] == 0)
    return [int(retained and bounded), -a['collision'], a['success'], -a['timeout']]


def fit_actor(model, data, parts, spec, output, evaluator):
    """Exactly spec.updates supervised optimizer steps; never model.learn()."""
    output = Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    model.policy.configure_learning(train_critic=False)
    prefixes = model.policy.imitation_prefixes
    if prefixes != ('multi_adapter.',): raise ValueError('Expected independent multi-obstacle adapter')
    params = [p for name, p in model.policy.named_parameters() if name.startswith(prefixes) and p.requires_grad]
    fixed = {name: p.detach().clone() for name, p in model.policy.named_parameters() if not name.startswith(prefixes)}
    optimizer = torch.optim.Adam(params, lr=spec['learning_rate'])
    rng = np.random.default_rng(spec['seed']); counter = model.num_timesteps
    weights = np.asarray([spec['task_weights'][g] for g in GROUPS])
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 1): raise ValueError('Invalid task weights')
    tensors = {key: torch.as_tensor(data[key]) for key in ('obs', 'actions', 'terminal_obs', 'terminal_actions')}
    diagnostic_rng = np.random.default_rng(spec['seed'] + 10000)
    diagnostic = {}
    for group, episodes in parts.items():
        rows = np.concatenate([np.arange(data['offsets'][i], data['offsets'][i+1]) for i in episodes])
        diagnostic[group] = diagnostic_rng.choice(rows, min(1024, len(rows)), replace=False)
    history, selected, best = [], None, None
    def check(update):
        nonlocal selected, best
        with torch.no_grad():
            mse = {g: float((model.policy.tracking_mean(tensors['obs'][idx]) - tensors['actions'][idx]).square().mean())
                   for g, idx in diagnostic.items()}
        state = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        try: result = evaluator(model, update)
        finally: torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])
        for name, p in model.policy.named_parameters():
            if name in fixed and not torch.equal(p.detach(), fixed[name]): raise AssertionError(f'BC modified frozen {name}')
        event = {'update': update, 'training_mse': mse, **result}
        path = output / 'checkpoints' / f'bc_{update}.zip'; path.parent.mkdir(exist_ok=True)
        model.save(path)
        torch.save({'update': update, 'optimizer': optimizer.state_dict(), 'rng': rng.bit_generator.state,
                    'torch_rng': torch.get_rng_state()}, path.with_suffix('.bc_state.pt'))
        if best is None or tuple(event['score']) > best:
            best = tuple(event['score']); selected = copy.deepcopy(event); model.save(output / 'selected_model.zip')
        history.append(event); (output / 'history.json').write_text(json.dumps(history, indent=2))
        print('FOUR_BC_CHECK', json.dumps(event), flush=True)
    check(0)
    for update in range(1, spec['updates'] + 1):
        tasks = rng.choice(len(GROUPS), size=spec['batch_size'], p=weights)
        ids = np.empty(len(tasks), dtype=int)
        for k, group in enumerate(GROUPS):
            mask = tasks == k; ids[mask] = rng.choice(parts[group], int(mask.sum()))
        lo, hi = data['offsets'][ids], data['offsets'][ids + 1]
        rows = lo + (rng.random(len(ids)) * (hi - lo)).astype(int)
        x, target = tensors['obs'][rows], tensors['actions'][rows]
        prediction = model.policy.tracking_mean(x)
        with torch.no_grad():
            deviation = (model.policy.base_mean(x) - target).square().mean(1).clamp(0, 1)
            importance = 1 + spec['correction_weight'] * torch.as_tensor(tasks == 0) * deviation
        imitation = (importance * (prediction - target).square().mean(1)).sum() / importance.sum()
        terminal = (model.policy.tracking_mean(tensors['terminal_obs'][ids]) - tensors['terminal_actions'][ids]).square().mean()
        regularizer = model.policy.adapter_correction(x).square().mean()
        loss = imitation + spec['terminal_weight'] * terminal + spec['adapter_weight'] * regularizer
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite BC loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
        with (output / 'loss.jsonl').open('a') as f:
            f.write(json.dumps({'update': update, 'loss': float(loss.detach()),
                               'imitation': float(imitation.detach()), 'terminal': float(terminal.detach())}) + '\n')
        if update in spec['checkpoints']: check(update)
    if model.num_timesteps != counter: raise AssertionError('PPO steps advanced during BC')
    model.save(output / 'final_model.zip')
    metadata = {'updates': spec['updates'], 'selected': selected, 'history': history,
                'frozen_base_critic_and_std_unchanged': True, 'ppo_steps': 0,
                'task_weights': spec['task_weights'], 'training_scene_seeds': sorted(set(data['seeds'].tolist())),
                'retention_labels': 'initial frozen policy actions, not old RRT actions'}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    return metadata
