"""Bounded four-static BC pilot; pure-policy validation, no PPO or new test use."""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from wall_panel_rl.rl.episode_imitation import evaluate_controller
from wall_panel_rl.rl.four_obstacle_bc import make_pool, fit_actor, validation_score, GROUPS


def setting(settings, count):
    result = copy.deepcopy(settings)
    result['active_stage'].update(count=count, mode='static', name=f'{count}_static')
    return result


def run(study):
    torch.set_num_threads(1); torch.manual_seed(study['seed'])
    models, report = Path(study['model_root']), Path(study['report_root'])
    if models.exists() or report.exists(): raise FileExistsError('Refusing pilot overwrite')
    teacher_report = Path(study['teacher_report'])
    if read(teacher_report / 'status.json')['status'] != 'completed': raise ValueError('Teacher collection incomplete')
    teacher_summary = read(teacher_report / 'summary.json')
    if not teacher_summary['teacher']['ready_for_bc']: raise ValueError('Unresolved teacher layouts')
    teacher_protocol = read(teacher_report / 'protocol.json')
    for path, expected in read(teacher_report / 'data_hashes.json').items():
        if sha(path) != expected: raise ValueError('Frozen scene bank changed')
    if sha(study['teacher']) != teacher_summary['teacher']['teacher_sha256']: raise ValueError('Teacher labels changed')
    source = Path(teacher_protocol['spec']['source_report']) / 'selected_model.zip'
    source_hash = sha(source)
    if source_hash != teacher_protocol['source_model_sha256']: raise ValueError('Wrong initialization')
    config = teacher_protocol['configuration']; settings = teacher_protocol['evaluation_settings']
    data_root = Path(teacher_protocol['spec']['data_root'])
    four_cases = read(data_root / 'validation_cases.json')
    one_cases = read(study['one_validation'])
    two_cases = [c for c in read(study['two_validation']) if c['suite'] == 'stage_target']
    splits = read(teacher_report / 'split.json')
    excluded = set(read(study['old_heldout_metadata'])['heldout_scene_seeds']) | set(splits['validation']) | set(splits['test'])
    excluded.update(c['scene']['seed'] for c in [*one_cases, *two_cases])
    model = PPO.load(source, device='cpu'); original = copy.deepcopy(model.policy.state_dict())
    source_steps = model.num_timesteps
    pool, parts, included = make_pool(study['teacher'], study['retention_teacher'], excluded, model.policy)
    if not set(pool['seeds'][parts['four_static']].tolist()) == set(splits['train']): raise ValueError('Four-task training layouts differ')
    assert not set(pool['seeds'].tolist()) & excluded
    models.mkdir(parents=True); report.mkdir(parents=True)
    state = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'ppo_steps': 0}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report / 'status.json', state); print('FOUR_BC', json.dumps(state), flush=True)
    code = ['scripts/run_four_obstacle_bc.py', 'configs/four_obstacle_bc.yaml',
            'src/wall_panel_rl/rl/four_obstacle_bc.py', 'src/wall_panel_rl/rl/multi_obstacle_adapter.py',
            'src/wall_panel_rl/rl/episode_imitation.py', 'src/wall_panel_rl/rl/conditional_recovery.py']
    hashes = {str(p): sha(p) for p in [source, Path(study['teacher']), Path(study['retention_teacher']),
              data_root / 'validation_cases.json', Path(study['one_validation']), Path(study['two_validation']),
              Path(study['old_heldout_metadata'])]}
    dump(report / 'protocol.json', {'study': study, 'source_model': str(source), 'source_model_sha256': source_hash,
        'source_ppo_num_timesteps': source_steps, 'configuration': config, 'evaluation_settings': settings,
        'code_hashes': {p: sha(p) for p in code}, 'input_hashes': hashes,
        'test_bank_sha256': sha(data_root / 'test_cases.json'), 'new_test_used_for_selection': False,
        'new_test_evaluated': False, 'excluded_scene_seeds': sorted(excluded),
        'train_scene_seeds': sorted(set(pool['seeds'].tolist())), 'included_episodes': included,
        'group_episode_counts': {g: len(ids) for g, ids in parts.items()},
        'retention_labels': 'deterministic INITIAL policy mean on old training observations, not old RRT labels',
        'validation_controller': 'pure deterministic policy; no RRT branch or action rejection',
        'selection': 'old one/alignment trajectories identical, old two success/collision non-regression; '
                     'four success decrease at most2 episodes; allow prevented collisions to become timeouts; '
                     'fewer collisions, then more success; ties keep earlier checkpoint',
        'ppo_steps': 0, 'default_demo_unchanged': True})
    np.savez_compressed(models / 'training_pool.npz', **pool)
    baseline = None; baseline_one_rows = None
    def evaluator(candidate, update):
        nonlocal baseline, baseline_one_rows
        four = evaluate_controller(candidate, config, settings, four_cases, mode='ppo')
        one = evaluate_controller(candidate, config, setting(settings, 1), one_cases, mode='ppo')
        two = evaluate_controller(candidate, config, setting(settings, 2), two_cases, mode='ppo')
        if baseline_one_rows is None: baseline_one_rows = one['rows']
        value = {'four': four['metrics']['stage_target'], 'two': two['metrics']['stage_target'],
                 'one': one['metrics'], 'one_rows_identical': one['rows'] == baseline_one_rows}
        if baseline is None: baseline = copy.deepcopy(value)
        value['score'] = validation_score(value, baseline, study['maximum_success_drop'])
        value['eligible'] = bool(value['score'][0])
        dump(report / 'validation' / f'update_{update}.json', {'summary': value, 'four': four, 'one': one, 'two': two})
        status('bc_validation', bc_updates=update, four=value['four'], two=value['two'], eligible=value['eligible'])
        return value
    try:
        status('training_interface_ready', updates_budget=study['bc']['updates'],
               group_episodes={g: len(ids) for g, ids in parts.items()})
        metadata = fit_actor(model, pool, parts, dict(study['bc'], seed=study['seed']), models / 'bc', evaluator)
        selected = metadata['selected']; selected_path = models / 'bc/selected_model.zip'
        chosen = PPO.load(selected_path, device='cpu')
        fixed_exact = all(torch.equal(value, chosen.policy.state_dict()[name]) for name, value in original.items()
                          if not name.startswith('multi_adapter.'))
        if not fixed_exact or chosen.num_timesteps != source_steps: raise AssertionError('Frozen state or PPO counter changed')
        # Exact action equality on ALL retained zero/one-box training states.
        probe = torch.as_tensor(pool['obs'])
        mask = (probe[:, 20:70].reshape(-1, 5, 10)[:, :, 9] > .5).sum(1) <= 1
        reference = PPO.load(source, device='cpu')
        with torch.no_grad(): protected_actions = torch.equal(reference.policy.tracking_mean(probe[mask]), chosen.policy.tracking_mean(probe[mask]))
        if not protected_actions or not selected['one_rows_identical']: raise AssertionError('Protected skills changed')
        a, b = selected['four'], baseline['four']
        gain = a['collision'] < b['collision'] or (a['collision'] == b['collision'] and a['success'] > b['success'])
        ready = (selected['update'] > 0 and selected['eligible'] and gain
                 and a['success'] / a['episodes'] >= .5 and a['collision'] / a['episodes'] <= .25)
        shutil.copy2(selected_path, report / 'selected_model.zip')
        shutil.copy2(models / 'bc/final_model.zip', report / 'final_candidate.zip')
        dump(report / 'candidate_manifest.json', {'configuration': config, 'evaluation_settings': settings,
             'model_sha256': sha(selected_path), 'selected_bc_update': selected['update'],
             'validation_only': True, 'default_model_preserved': True})
        summary = {'baseline': baseline, 'selected_bc': selected, 'bc_updates_run': study['bc']['updates'],
                   'ppo_started': False, 'ppo_steps': 0, 'ready_for_ppo_pilot': ready,
                   'learned_validation_gain': gain and selected['update'] > 0,
                   'frozen_state_unchanged': fixed_exact, 'zero_one_actions_identical': protected_actions,
                   'test_used_for_selection': False, 'new_test_evaluated': False,
                   'source_unchanged': sha(source) == source_hash, 'default_model_preserved': True,
                   'selected_model_sha256': sha(selected_path), 'final_candidate_sha256': sha(models / 'bc/final_model.zip'),
                   'training_pool_sha256': sha(models / 'training_pool.npz')}
        for path, expected in hashes.items():
            if sha(path) != expected: raise AssertionError('An input changed during training')
        if sha(data_root / 'test_cases.json') != read(report / 'protocol.json')['test_bank_sha256']:
            raise AssertionError('Reserved test bank changed')
        dump(report / 'summary.json', summary)
        for path in code:
            dest = report / 'code_archive' / path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        status('complete', status='completed', bc_updates=study['bc']['updates'],
               selected_bc_update=selected['update'], ready_for_ppo_pilot=ready)
    except Exception as exc:
        status('failed', status='failed', error=repr(exc)); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/four_obstacle_bc.yaml'); p.add_argument('--run', action='store_true')
    args = p.parse_args(); study = yaml.safe_load(Path(args.config).read_text())
    if study['ppo_steps'] != 0 or study['bc']['updates'] not in study['bc']['checkpoints']:
        raise ValueError('BC-only pilot with a final validation checkpoint required')
    print({'mode': 'run' if args.run else 'check_only', 'bc_updates': study['bc']['updates'], 'ppo_steps': 0}, flush=True)
    if args.run: run(study)


if __name__ == '__main__': main()
