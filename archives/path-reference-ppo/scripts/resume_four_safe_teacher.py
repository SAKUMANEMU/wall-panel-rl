"""Resume an interrupted teacher collection without regenerating frozen banks.

Archive the original partial package, collect a separate suffix, and verify
exact prefix preservation before replacing the working teacher package. No fit.
"""
import _bootstrap
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from prepare_four_safe_teacher import collect, audit_replay
from run_targeted_bc import read, dump, sha
from wall_panel_rl.rl.avoidance_imitation import save_teacher


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def main():
    torch.set_num_threads(1)
    report = Path('results/diagnostics/four_safe_teacher_v1_seed61')
    protocol = read(report / 'protocol.json')
    spec = protocol['spec']; data = Path(spec['data_root']); teacher = data / 'teacher'
    state = read(report / 'status.json')
    if state['status'] != 'running' or state['phase'] != 'collect_verified_teacher':
        raise RuntimeError('Expected an interrupted collection checkpoint')
    if (report / 'resume_protocol.json').exists():
        raise FileExistsError('Refusing to overwrite an earlier resume attempt')
    for path, expected in protocol['code_hashes'].items():
        if sha(path) != expected: raise AssertionError(f'Original source changed: {path}')
    for path, expected in read(report / 'data_hashes.json').items():
        if sha(path) != expected: raise AssertionError(f'Frozen bank changed: {path}')
    source = Path(spec['source_report']) / 'selected_model.zip'
    if sha(source) != protocol['source_model_sha256']: raise AssertionError('Wrong policy weights')
    cases = read(data / 'train_cases.json'); attempts = read(teacher / 'attempts.json')
    prefix = arrays(teacher / 'teacher.npz')
    completed = int(state['layouts_completed']); old_episodes = len(prefix['seeds'])
    assert 0 < completed < len(cases)
    assert old_episodes == state['episodes'] and len(attempts) == state['attempts']
    expected = [c['scene']['seed'] for c in cases[:completed]]
    assert list(dict.fromkeys(a['seed'] for a in attempts)) == expected
    accepted = [a for a in attempts if a['accepted']]
    assert prefix['seeds'].tolist() == [a['seed'] for a in accepted]
    assert all(a['reason'] == 'success' and a['pre_execution_verified'] and a['post_execution_verified'] for a in accepted)
    assert len(prefix['offsets']) == old_episodes + 1 and prefix['offsets'][-1] == len(prefix['obs'])
    for i, case in enumerate(cases[:completed]):
        tried = [a for a in attempts if a['seed'] == case['scene']['seed']]
        assert tried[-1]['accepted'] or len(tried) == len(spec['planner_seed_offsets'])
    archive = report / 'interrupted_checkpoint'
    archive.mkdir(); shutil.copytree(teacher, archive / 'teacher')
    for name in ('status.json', 'protocol.json', 'data_hashes.json', 'split.json'):
        shutil.copy2(report / name, archive / name)
    original_hash = sha(teacher / 'teacher.npz')
    dump(report / 'resume_protocol.json', {
        'resumed_at': datetime.now().astimezone().isoformat(), 'completed_layouts': completed,
        'preserved_episodes': old_episodes, 'preserved_transitions': len(prefix['obs']),
        'partial_teacher_sha256': original_hash, 'partial_attempts_sha256': sha(teacher / 'attempts.json'),
        'original_protocol_sha256': sha(report / 'protocol.json'),
        'remaining_train_seeds': [c['scene']['seed'] for c in cases[completed:]],
        'code_hashes': {'scripts/resume_four_safe_teacher.py': sha(__file__)},
        'same_configuration_and_seed_rule': True, 'training_steps': 0})
    state['resumed_at'] = datetime.now().astimezone().isoformat()
    def status(phase, **extra):
        if phase == 'collect_verified_teacher':
            extra['layouts_completed'] += completed
            extra['episodes'] += old_episodes
            extra['attempts'] += len(attempts)
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report / 'status.json', state)
        print('SAFE_TEACHER_RESUME', json.dumps(state), flush=True)
    config = protocol['configuration']; settings = protocol['evaluation_settings']
    model = PPO.load(source, device='cpu')
    segment = data / 'teacher_resume_segment'
    try:
        collect(model, config, settings, cases[completed:], spec['planner_seed_offsets'], segment, status)
        suffix = arrays(segment / 'teacher.npz'); added_attempts = read(segment / 'attempts.json')
        merged = {key: np.concatenate([prefix[key], suffix[key]]) for key in prefix if key != 'offsets'}
        merged['offsets'] = np.concatenate([prefix['offsets'], suffix['offsets'][1:] + len(prefix['obs'])])
        save_teacher(teacher / 'teacher.npz', merged['obs'], merged['actions'], merged['raw_actions'],
                     merged['offsets'], merged['seeds'], merged['tasks'], merged['terminal_obs'], merged['terminal_actions'])
        saved = arrays(teacher / 'teacher.npz')
        assert all(np.array_equal(saved[k][:len(v)], v) for k, v in prefix.items())
        assert sha(archive / 'teacher/teacher.npz') == original_hash
        for attempt in added_attempts:
            if attempt['saved_episode'] is not None: attempt['saved_episode'] += old_episodes
        attempts.extend(added_attempts); dump(teacher / 'attempts.json', attempts)
        seeds = saved['seeds'].tolist()
        metadata = {'requested_layouts': len(cases), 'episodes': len(seeds), 'transitions': len(saved['obs']),
            'attempts': len(attempts), 'accepted_seeds': seeds,
            'unresolved_train_seeds': [c['scene']['seed'] for c in cases if c['scene']['seed'] not in set(seeds)],
            'all_accepted_pre_and_post_verified': all(a['pre_execution_verified'] and a['post_execution_verified'] for a in attempts if a['accepted']),
            'all_accepted_physically_successful': all(a['reason'] == 'success' for a in attempts if a['accepted']),
            'original_reference_preserved': True, 'one_route_per_layout': len(seeds) == len(set(seeds)),
            'teacher_sha256': sha(teacher / 'teacher.npz'), 'ready_for_bc': len(seeds) == len(cases),
            'resumed_prefix_arrays_exactly_preserved': True, 'training_steps': 0}
        dump(teacher / 'metadata.json', metadata)
        dump(report / 'teacher_replay.json', audit_replay(config, settings, cases, teacher))
        dump(report / 'resumed_segment_replay.json', audit_replay(config, settings, cases[completed:], segment))
        assert sha(source) == protocol['source_model_sha256']
        for path, expected_hash in read(report / 'data_hashes.json').items(): assert sha(path) == expected_hash
        banks = {name: read(data / f'{name}_cases.json') for name in ('train', 'validation', 'test')}
        sets = {name: {c['scene']['seed'] for c in bank} for name, bank in banks.items()}
        old = {c['scene']['seed'] for c in read(spec['regression_cases'])}
        assert all(not sets[a] & sets[b] for a in sets for b in sets if a != b)
        assert all(not values & old for values in sets.values()) and set(seeds).issubset(sets['train'])
        dump(report / 'summary.json', {'teacher': metadata, 'bank_counts': {k: len(v) for k, v in banks.items()},
            'split_disjoint': True, 'regression_excluded_from_training': True, 'source_weights_unchanged': True,
            'validation_and_test_not_used_for_expert_collection': True, 'resumed_prefix_arrays_exactly_preserved': True,
            'training_steps': 0})
        for path in [*protocol['code_hashes'], 'scripts/resume_four_safe_teacher.py']:
            dest = report / 'code_archive' / path
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        status('complete', status='completed', ready_for_bc=metadata['ready_for_bc'], episodes=len(seeds))
    except Exception as exc:
        status('failed', status='failed', error=repr(exc)); raise


if __name__ == '__main__': main()
