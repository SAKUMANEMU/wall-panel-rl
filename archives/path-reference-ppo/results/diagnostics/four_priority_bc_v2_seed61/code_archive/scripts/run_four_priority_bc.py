"""Continue four-static BC; select on four validation only, then test once."""
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
from wall_panel_rl.rl.four_priority_bc import load_teacher, fit, score


def run(study):
    torch.set_num_threads(1); torch.manual_seed(study['seed'])
    models, report = Path(study['model_root']), Path(study['report_root'])
    if models.exists() or report.exists(): raise FileExistsError('Refusing priority-study overwrite')
    previous = Path(study['previous_report']); teacher_report = Path(study['teacher_report'])
    previous_summary, teacher_summary = read(previous / 'summary.json'), read(teacher_report / 'summary.json')
    if sha(study['initial_model']) != previous_summary['final_candidate_sha256']: raise ValueError('Wrong BC1000 source')
    if sha(study['teacher']) != teacher_summary['teacher']['teacher_sha256']: raise ValueError('Changed teacher')
    tp = read(teacher_report / 'protocol.json'); pp = read(previous / 'protocol.json')
    for path, expected in read(teacher_report / 'data_hashes.json').items():
        if sha(path) != expected: raise ValueError('Frozen scene bank changed')
    config, settings = tp['configuration'], tp['evaluation_settings']
    data_root = Path(tp['spec']['data_root']); splits = read(teacher_report / 'split.json')
    data = load_teacher(study['teacher'], splits['train'], [*splits['validation'], *splits['test']])
    cases = read(data_root / 'validation_cases.json')
    cached = read(previous / f'validation/update_{study["initial_bc_updates"]}.json')
    initial_metrics = cached['four']['metrics']['stage_target']
    expected = {r['seed']: r for r in cached['four']['rows']}
    model = PPO.load(study['initial_model'], device='cpu')
    before = copy.deepcopy(model.policy.state_dict()); counter = model.num_timesteps
    models.mkdir(parents=True); report.mkdir(parents=True)
    state = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'ppo_steps': 0}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report / 'status.json', state); print('FOUR_PRIORITY', json.dumps(state), flush=True)
    code = ['scripts/run_four_priority_bc.py', 'configs/four_priority_bc.yaml',
            'src/wall_panel_rl/rl/four_priority_bc.py', 'src/wall_panel_rl/rl/four_obstacle_bc.py',
            'src/wall_panel_rl/rl/multi_obstacle_adapter.py', 'src/wall_panel_rl/rl/episode_imitation.py',
            'src/wall_panel_rl/rl/conditional_recovery.py']
    inputs = {str(p): sha(p) for p in [Path(study['initial_model']), Path(study['teacher']),
        data_root / 'train_cases.json', data_root / 'validation_cases.json', data_root / 'test_cases.json']}
    dump(report / 'protocol.json', {'study': study, 'configuration': config, 'evaluation_settings': settings,
        'code_hashes': {p: sha(p) for p in code}, 'input_hashes': inputs,
        'training_scene_seeds': sorted(data['seeds'].tolist()), 'validation_seeds': splits['validation'],
        'reserved_test_seeds': splits['test'], 'selection': 'four success first; collisions not above source baseline; no joint limits; earlier wins ties',
        'two_obstacle_used_for_training': False, 'two_obstacle_used_for_selection': False,
        'four_obstacle_training_probability': 1., 'ppo_steps': 0, 'default_model_preserved': True})
    def evaluator(candidate, update):
        result = evaluate_controller(candidate, config, settings, cases, mode='ppo')
        metrics = result['metrics']['stage_target']
        if update == 0:
            assert metrics == initial_metrics
            assert all(r['reason'] == expected[r['seed']]['reason'] and r['steps'] == expected[r['seed']]['steps'] for r in result['rows'])
        value = {'four': metrics, 'score': score(metrics, initial_metrics['collision'])}
        dump(report / 'validation' / f'update_{update}.json', result)
        status('bc_validation', additional_bc_updates=update, total_bc_updates=study['initial_bc_updates']+update, four=metrics)
        return value
    try:
        status('four_only_bc', training_layouts=len(data['seeds']), training_transitions=len(data['obs']), updates_budget=study['bc']['updates'])
        metadata = fit(model, data, dict(study['bc'], seed=study['seed']), models / 'bc', evaluator)
        selected = metadata['selected']; chosen = PPO.load(models / 'bc/selected_model.zip', device='cpu')
        fixed = all(torch.equal(v, chosen.policy.state_dict()[k]) for k, v in before.items() if not k.startswith('multi_adapter.'))
        assert fixed and chosen.num_timesteps == counter
        source = PPO.load(study['initial_model'], device='cpu')
        probe = torch.as_tensor(data['obs'][:128].copy()); one = probe.clone(); zero = probe.clone()
        one[:, 30:70] = 0; zero[:, 20:70] = 0; probe = torch.cat([one, zero])
        with torch.no_grad(): protected = torch.equal(source.policy.tracking_mean(probe), chosen.policy.tracking_mean(probe))
        assert protected
        gained = tuple(selected['score']) > tuple(metadata['history'][0]['score'])
        # Two-box performance is diagnostic only; it cannot reject this model.
        two_settings = copy.deepcopy(settings); two_settings['active_stage'].update(count=2, name='two_static', mode='static')
        two_cases = [c for c in read(pp['study']['two_validation']) if c['suite'] == 'stage_target']
        two = evaluate_controller(chosen, config, two_settings, two_cases, mode='ppo')
        dump(report / 'two_diagnostic.json', two)
        test_before = test_after = None
        if gained and study['evaluate_reserved_test_after_improvement']:
            status('reserved_test', selected_additional_bc_updates=selected['additional_bc_updates'])
            tests = read(data_root / 'test_cases.json')
            test_before = evaluate_controller(source, config, settings, tests, mode='ppo')
            test_after = evaluate_controller(chosen, config, settings, tests, mode='ppo')
            test_before['independent_test'] = test_after['independent_test'] = True
            dump(report / 'test_source.json', test_before); dump(report / 'test_selected.json', test_after)
        shutil.copy2(models / 'bc/selected_model.zip', report / 'selected_model.zip')
        shutil.copy2(models / 'bc/final_model.zip', report / 'final_candidate.zip')
        dump(report / 'candidate_manifest.json', {'configuration': config, 'evaluation_settings': settings,
            'model_sha256': sha(report / 'selected_model.zip'), 'selected_total_bc_updates': study['initial_bc_updates']+selected['additional_bc_updates'],
            'default_model_preserved': True})
        for path, expected_hash in inputs.items():
            if sha(path) != expected_hash: raise AssertionError('Study input changed')
        summary = {'source_validation': initial_metrics, 'selected_bc': selected, 'additional_bc_updates_run': study['bc']['updates'],
            'selected_total_bc_updates': study['initial_bc_updates']+selected['additional_bc_updates'],
            'validation_improved': gained, 'two_diagnostic': two['metrics']['stage_target'],
            'two_used_for_selection': False, 'test_evaluated': test_after is not None,
            'test_used_for_selection': False, 'test_source': test_before['metrics']['stage_target'] if test_before else None,
            'test_selected': test_after['metrics']['stage_target'] if test_after else None,
            'frozen_state_unchanged': fixed, 'zero_one_actions_identical': protected,
            'source_unchanged': True, 'default_model_preserved': True, 'ppo_steps': 0,
            'selected_model_sha256': sha(report / 'selected_model.zip')}
        dump(report / 'summary.json', summary)
        for path in code:
            dest = report / 'code_archive' / path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        status('complete', status='completed', selected_total_bc_updates=summary['selected_total_bc_updates'],
               validation=selected['four'], test=summary['test_selected'])
    except Exception as exc:
        status('failed', status='failed', error=repr(exc)); raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/four_priority_bc.yaml'); p.add_argument('--run', action='store_true')
    args = p.parse_args(); study = yaml.safe_load(Path(args.config).read_text())
    if study['ppo_steps'] != 0 or study['bc']['updates'] not in study['bc']['checkpoints']: raise ValueError('BC-only study with final checkpoint required')
    print({'mode': 'run' if args.run else 'check_only', 'additional_bc_updates': study['bc']['updates'], 'four_task_probability': 1., 'ppo_steps': 0}, flush=True)
    if args.run: run(study)


if __name__ == '__main__': main()
