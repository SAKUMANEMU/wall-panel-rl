"""Diagnose, collect targeted training corrections, BC, then gated 10k PPO.

Only --run performs collection/training. The frozen new test is read by final
evaluation only. Previous studies and the default one-box model are preserved.
"""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from run_static_expansion import known_scene_seeds, freeze_fresh, project_cases
from run_two_static_warmup import stage, validation
from train_two_static_ppo import train as train_ppo
from wall_panel_rl.rl.avoidance_curriculum import stage_scene, gate_result
from wall_panel_rl.rl.avoidance_imitation import save_teacher, AlignmentTeacher
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import make_rrt, evaluate_controller
from wall_panel_rl.rl.multi_obstacle_warmup import warmup_actor, teacher_arrays
from wall_panel_rl.rl.multi_obstacle_adapter import initialize_adapter, frozen_base_preserved


def resolve(path):
    override = yaml.safe_load(Path(path).read_text())
    study = yaml.safe_load(Path(override['base_config']).read_text())
    study.update(override)
    return study


def correction_handoffs(length, leads):
    """Return safe pre-action indices, never the already-terminal state."""
    return sorted({max(0, length - 1 - int(lead)) for lead in leads})


def trace_policy(env, model, scene, strength):
    obs, info = env.reset(options={'scenario': scene, 'task': 'static', 'strength': strength})
    states = []
    while True:
        action = model.predict(obs, deterministic=True)[0]
        previous = obs.copy()
        obs, _, done, truncated, info = env.step(action)
        states.append({'observation': previous, 'action': np.asarray(action).copy(),
                       'executed_action': np.asarray(info['executed_action']).copy()})
        if done or truncated:
            return {'reason': info['termination_reason'], 'states': states,
                    'steps': env.env.steps, 'position_m': float(info['position_error'])}


def collect_targeted(study, config, one, two, scenes, model, output, status):
    """Audit second-box relevance and teach successful post-prefix RRT recoveries.

    Original references are never replaced. Only training layouts are accepted
    by the caller. Failed RRT attempts are journalled and never become labels.
    """
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    env = ConditionalRecoveryEnv(copy.deepcopy(config), [], two)
    one_env = ConditionalRecoveryEnv(copy.deepcopy(config), [], one)
    strength = [two['active_stage']['amplitude'], two['active_stage']['speed']]
    anchor = AlignmentTeacher(model)
    audits, jobs, attempts = [], [], []
    observations, actions, raw, offsets = [], [], [], [0]
    seeds, tasks, terminals, terminal_actions = [], [], [], []
    try:
        for index, original in enumerate(scenes):
            scene = stage_scene(original, two['active_stage'])
            single = copy.deepcopy(scene); single['obstacles'] = single['obstacles'][:1]
            single['obstacle_original_indices'] = single['obstacle_original_indices'][:1]
            one_env.reset(options={'scenario': single, 'task': 'static', 'strength': strength})
            expert = make_rrt(one_env, original['seed'] + 100000)
            env.reset(options={'scenario': scene, 'task': 'static', 'strength': strength})
            blocked = []
            if expert.planning_success:
                for edge, (a, b) in enumerate(zip(expert.path[:-1], expert.path[1:])):
                    if not env.env.collision.edge_valid(a, b):
                        blocked.append(edge)
            trace = trace_policy(env, model, scene, strength)
            audits.append({'seed': original['seed'], 'one_rrt_success': bool(expert.planning_success),
                'second_blocks_one_rrt': bool(blocked), 'blocked_edges': blocked,
                'policy_reason': trace['reason'], 'policy_steps': trace['steps'],
                'policy_final_position_m': trace['position_m']})
            if blocked:
                jobs.append({'seed': original['seed'], 'scene': scene, 'prefix': [], 'expected': None,
                             'kind': 'second_blocks_one_rrt', 'priority': 0})
            if trace['reason'] != 'success':
                for phase, handoff in enumerate(correction_handoffs(len(trace['states']), study['handoff_leads'])):
                    jobs.append({'seed': original['seed'], 'scene': scene,
                        'prefix': [r['action'] for r in trace['states'][:handoff]],
                        'expected': trace['states'][handoff]['observation'],
                        'kind': 'policy_failure_correction', 'priority': phase + 1})
            if (index + 1) % 6 == 0 or index + 1 == len(scenes):
                dump(output/'layout_audit.json', audits)
                status('audit_training_scenes', layouts=index+1,
                    second_blocks_one_rrt=sum(a['second_blocks_one_rrt'] for a in audits),
                    policy_failures=sum(a['policy_reason'] != 'success' for a in audits))
        jobs.sort(key=lambda j: (j['priority'], j['seed']))
        for job in jobs[:study['maximum_correction_attempts']]:
            obs, info = env.reset(options={'scenario': job['scene'], 'task': 'static', 'strength': strength})
            for action in job['prefix']:
                obs, _, done, truncated, info = env.step(action)
                if done or truncated:
                    raise AssertionError('Policy prefix terminated before its saved handoff')
            replay_error = (float(np.max(np.abs(obs - job['expected']))) if job['expected'] is not None else 0.)
            if replay_error > 1e-6:
                raise AssertionError(f'Prefix observation mismatch: {replay_error}')
            planner_seed = int(job['seed']) + 300000 + len(attempts)
            expert = make_rrt(env, planner_seed)
            xo, ya, ra = [], [], []
            if expert.planning_success:
                while True:
                    command = expert.action(obs); previous = obs.copy()
                    obs, _, done, truncated, info = env.step(command)
                    xo.append(previous); ya.append(info['executed_action']); ra.append(command.copy())
                    if done or truncated: break
                reason = info['termination_reason']
            else:
                reason = 'planning_failure'
            record = {'seed': job['seed'], 'kind': job['kind'], 'reason': reason,
                'scene': job['scene'], 'prefix_actions': np.asarray(job['prefix']).tolist(),
                'prefix_replay_error': replay_error, 'planner_seed': planner_seed,
                'saved_episode': len(seeds) if reason == 'success' else None,
                'planning_seconds': expert.planning_time, 'steps': env.env.steps}
            attempts.append(record)
            if reason == 'success':
                observations.extend(xo); actions.extend(ya); raw.extend(ra); offsets.append(len(observations))
                seeds.append(job['seed']); tasks.append('two_static'); terminals.append(obs.copy())
                terminal_actions.append(anchor.action(obs))
            if len(attempts) % 6 == 0:
                dump(output/'attempts.json', attempts)
                status('collect_targeted_rrt', attempts=len(attempts), successes=len(seeds))
        save_teacher(output/'teacher.npz', observations, actions, raw, offsets, seeds, tasks, terminals, terminal_actions)
        dump(output/'attempts.json', attempts)
        dump(output/'metadata.json', {'episodes': len(seeds), 'transitions': len(observations),
            'attempts': len(attempts), 'training_seeds_only': [s['seed'] for s in scenes],
            'second_blocks_one_rrt': sum(a['second_blocks_one_rrt'] for a in audits),
            'failures': {key: sum(a['reason'] == key for a in attempts)
                         for key in ('collision', 'timeout', 'joint_limit', 'planning_failure')},
            'prefixes_are_not_training_labels': True, 'reference_unchanged': True,
            'teacher_sha256': sha(output/'teacher.npz')})
    finally:
        env.close(); one_env.close()


def merge_teachers(base, corrections, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with np.load(Path(base)/'teacher.npz', allow_pickle=False) as a, np.load(Path(corrections)/'teacher.npz', allow_pickle=False) as b:
        save_teacher(output/'teacher.npz', np.concatenate([a['obs'], b['obs']]),
            np.concatenate([a['actions'], b['actions']]), np.concatenate([a['raw_actions'], b['raw_actions']]),
            np.r_[a['offsets'], b['offsets'][1:]+len(a['obs'])], np.r_[a['seeds'], b['seeds']],
            np.r_[a['tasks'], b['tasks']], np.concatenate([a['terminal_obs'], b['terminal_obs']]),
            np.concatenate([a['terminal_actions'], b['terminal_actions']]))
        metadata = {'parent_episodes': len(a['seeds']), 'new_episodes': len(b['seeds']),
            'episodes': len(a['seeds'])+len(b['seeds']), 'transitions': len(a['obs'])+len(b['obs']),
            'parent_sha256': sha(Path(base)/'teacher.npz'),
            'corrections_sha256': sha(Path(corrections)/'teacher.npz'), 'teacher_sha256': sha(output/'teacher.npz')}
    dump(output/'metadata.json', metadata)
    return metadata


def audit_corrective_replay(config, two, directory):
    """Replay an actual saved suffix for each new collection kind."""
    directory = Path(directory); records = read(directory/'attempts.json'); checks = []
    with np.load(directory/'teacher.npz', allow_pickle=False) as z:
        d = {name: z[name].copy() for name in z.files}
    env = ConditionalRecoveryEnv(copy.deepcopy(config), [], two)
    try:
        for kind in sorted({r['kind'] for r in records if r['reason'] == 'success'}):
            record = next(r for r in records if r['kind'] == kind and r['reason'] == 'success')
            obs, _ = env.reset(options={'scenario': record['scene'], 'task': 'static',
                'strength': [two['active_stage']['amplitude'], two['active_stage']['speed']]})
            for action in record['prefix_actions']:
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated: raise AssertionError('Replay prefix ended')
            i = record['saved_episode']; error = 0.
            for k in range(d['offsets'][i], d['offsets'][i+1]):
                error = max(error, float(np.max(np.abs(obs-d['obs'][k]))))
                obs, _, done, truncated, info = env.step(d['raw_actions'][k])
                error = max(error, float(np.max(np.abs(np.asarray(info['executed_action'])-d['actions'][k]))))
            error = max(error, float(np.max(np.abs(obs-d['terminal_obs'][i]))))
            if not (done or truncated) or info['termination_reason'] != 'success' or error > 1e-6:
                raise AssertionError('Corrective teacher physical replay failed')
            checks.append({'kind': kind, 'max_error': error, 'reason': info['termination_reason']})
    finally: env.close()
    return checks


def second_slot_sensitivity(model, observations):
    x = torch.as_tensor(observations[:2048], dtype=torch.float32)
    changed = x.clone(); changed[:, 30:33] += .15
    with torch.no_grad():
        action = model.policy.tracking_mean(x)
        difference = (model.policy.tracking_mean(changed)-action).abs()
        return {'mean_action_change_15cm': float(difference.mean()),
                'max_action_change_15cm': float(difference.max()),
                'adapter_mean_abs_correction': float(model.policy.adapter_correction(x).abs().mean())}


def ppo_learning_gate(selected, initial):
    """Require a learned validation gain, not just a qualifying initialization."""
    a, b = selected['two']['stage_target'], initial['two']['stage_target']
    better = ((a['collision'] < b['collision'] and a['success'] >= b['success']) or
              (a['success'] > b['success'] and a['collision'] <= b['collision']))
    return bool(selected['update'] > 0 and selected['ppo_ready'] and better and
                selected['second_slot_sensitivity']['max_action_change_15cm'] > 1e-5)


def run(study):
    data, models, report = map(Path, (study['data_root'], study['model_root'], study['report_root']))
    if data.exists() or models.exists(): raise FileExistsError('Refusing study overwrite')
    if report.exists() and set(p.name for p in report.iterdir()) != {'interface_diagnosis.json'}:
        raise FileExistsError('Only the pre-training interface diagnosis may already exist')
    parent = read(Path(study['source_report'])/'manifest.json'); source_hash = sha(study['initial_model'])
    if parent['model_sha256'] != source_hash: raise ValueError('Wrong original source')
    held = set(read(study['heldout_metadata'])['heldout_scene_seeds'])
    scenes = [s for s in read(study['raw_train']) if s['seed'] not in held]
    forbidden = known_scene_seeds(); data.mkdir(parents=True); models.mkdir(parents=True); report.mkdir(parents=True, exist_ok=True)
    (data/'resolved_study.yaml').write_text(yaml.safe_dump(study, sort_keys=False))
    state = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(),
             'runner_pid': os.getpid(), 'automatic_dynamic_stage': False, 'default_model_preserved': True}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report/'status.json', state); print('TWO_ADAPTER', json.dumps(state), flush=True)
    config = copy.deepcopy(parent['configuration']); one = stage(parent['evaluation_settings'], 1); two = stage(one, 2)
    code = ['scripts/run_two_static_adapter.py', 'configs/two_static_adapter.yaml',
        'src/wall_panel_rl/rl/multi_obstacle_adapter.py', 'src/wall_panel_rl/rl/multi_obstacle_warmup.py',
        'src/wall_panel_rl/rl/episode_imitation.py', 'scripts/train_two_static_ppo.py']
    dump(report/'protocol.json', {'study': study, 'source_sha256': source_hash,
        'code_hashes': {p: sha(p) for p in code}, 'train_seeds': [s['seed'] for s in scenes],
        'teacher_heldout_seeds': sorted(held), 'new_test_used_for_selection': False,
        'actor_update': 'new multi_adapter only; frozen original actor; zero mask for <2 obstacles',
        'augmentation': 'canonical obstacle order, no slot swapping',
        'ppo_gate': 'learned validation gain versus initialization + nonzero second-slot response + original BC readiness',
        'bc_updates': study['bc']['updates'], 'maximum_ppo_steps': study['ppo_steps']})
    try:
        status('freeze_new_test')
        fresh, failures = freeze_fresh(study, config, one, forbidden, status)
        dump(data/'fresh_full.json', fresh); dump(data/'generation_failures.json', failures)
        dump(data/'two_test.json', project_cases(fresh, two['active_stage']))
        for name in ('one_validation.json', 'two_validation.json'):
            shutil.copy2(Path(study['validation_root'])/name, data/name)
        one_cases, two_cases = read(data/'one_validation.json'), read(data/'two_validation.json')
        if {s['seed'] for s in scenes} & {c['scene']['seed'] for c in two_cases}:
            raise ValueError('Training/validation overlap')
        dump(report/'data_hashes.json', {str(p): sha(p) for p in
            (data/'fresh_full.json', data/'two_test.json', data/'one_validation.json', data/'two_validation.json')})
        initialized = initialize_adapter(study['initial_model'], study, study['seed'])
        initialized.save(models/'initialized_model.zip')
        reference = PPO.load(models/'initialized_model.zip', device='cpu')
        initial, _, _ = validation(initialized, config, two, one, two_cases, one_cases)
        dump(report/'initialization_validation.json', initial)
        status('targeted_collection', initial_two=initial['two']['stage_target'])
        collect_targeted(study, config, one, two, scenes, initialized, data/'corrections', status)
        dump(report/'teacher_replay.json', audit_corrective_replay(config, two, data/'corrections'))
        merged = merge_teachers(study['teacher_root'], data/'corrections', data/'teacher')
        teacher, parts = teacher_arrays(data/'teacher/teacher.npz', held)
        rows = np.concatenate([np.arange(teacher['offsets'][i], teacher['offsets'][i+1]) for i in parts['train']['two_static']])
        sample = teacher['obs'][rows]
        def evaluator(candidate, update):
            if not frozen_base_preserved(reference.policy, candidate.policy): raise AssertionError('Base actor changed')
            result, arows, brows = validation(candidate, config, two, one, two_cases, one_cases)
            result['second_slot_sensitivity'] = second_slot_sensitivity(candidate, sample)
            dump(report/'bc_validation'/f'update_{update}.json', {'summary': result, 'two_rows': arows, 'one_rows': brows})
            status('adapter_bc', bc_updates=update, two_validation=result['two']['stage_target'],
                one_validation=result['one']['stage_target'], sensitivity=result['second_slot_sensitivity'])
            return result
        spec = dict(study['bc'], seed=study['seed'], priority_episode_start=merged['parent_episodes'], priority_episode_weight=4.)
        warm, metadata = warmup_actor(initialized, data/'teacher/teacher.npz', held, spec, models/'bc', evaluator)
        selected = metadata['selected']; chosen_path = warm; ppo_started = False
        learning_gate = ppo_learning_gate(selected, initial)
        dump(report/'bc_learning_gate.json', {'passed': learning_gate, 'selected': selected})
        if learning_gate:
            bundle = copy.deepcopy(parent)
            for key in ('training_settings', 'evaluation_settings'): bundle[key] = stage(bundle[key], 2)
            bundle.update(model_sha256=sha(warm), warmup_metrics=selected, multi_obstacle_adapter=True)
            dump(data/'warm_source/manifest.json', bundle)
            ppo_study = {'source_report': str(data/'warm_source'), 'initial_model': str(warm),
                'train_scenes': study['raw_train'], 'heldout_scenes': study['heldout_metadata'],
                'validation_cases': str(data/'two_validation.json'), 'output_root': str(models/'ppo_run'),
                'report_root': str(report/'ppo_run'), 'ppo_steps': study['ppo_steps'],
                'ppo_eval_every': study['ppo_eval_every'], 'ppo_learning_rate': study['ppo_learning_rate'],
                'expert_probability': study['expert_probability'], 'seed': study['seed'],
                'auxiliary_bc': study['auxiliary_bc'], 'multi_obstacle_adapter': True}
            (data/'ppo_study.yaml').write_text(yaml.safe_dump(ppo_study, sort_keys=False))
            status('adapter_ppo', selected_bc_update=selected['update'])
            chosen_path, _ = train_ppo(ppo_study, one, one_cases, data/'teacher/teacher.npz', held, status)
            ppo_started = True
        else: status('bc_learning_gate_failed', selected_bc_update=selected['update'])
        chosen = PPO.load(chosen_path, device='cpu')
        if not frozen_base_preserved(reference.policy, chosen.policy): raise AssertionError('PPO base actor changed')
        final, _, _ = validation(chosen, config, two, one, two_cases, one_cases)
        # Lock the candidate BEFORE reading/scoring the new test bank.
        shutil.copy2(chosen_path, report/'candidate_model.zip')
        dump(report/'candidate_manifest.json', {'configuration': config, 'evaluation_settings': two,
            'model_sha256': sha(chosen_path), 'validation_eligible': final['eligible'],
            'multi_obstacle_adapter': True, 'default_model_preserved': True})
        status('final_new_test', ppo_started=ppo_started, candidate_sha256=sha(chosen_path))
        tests = read(data/'two_test.json')
        before = evaluate_controller(reference, config, two, tests, mode='ppo')
        after = evaluate_controller(chosen, config, two, tests, mode='ppo')
        one_test = read(study['one_regression'])
        old = evaluate_controller(reference, config, one, one_test, mode='ppo')
        regression = evaluate_controller(chosen, config, one, one_test, mode='ppo')
        if old['rows'] != regression['rows']: raise AssertionError('One-obstacle trajectories changed')
        for name, result in [('two_test_initialized', before), ('two_test_after', after), ('one_before', old), ('one_after', regression)]:
            dump(report/f'{name}.json', result)
        mixture = evaluate_controller(chosen, config, two, two_cases, mode='mixture')
        dump(report/'selected_mixture_validation.json', mixture)
        if sha(study['initial_model']) != source_hash: raise AssertionError('Original model file changed')
        dump(report/'summary.json', {'initial_validation': initial, 'selected_bc': selected,
            'bc_updates_run': study['bc']['updates'], 'ppo_started': ppo_started,
            'ppo_budget': study['ppo_steps'] if ppo_started else 0, 'bc_learning_gate': learning_gate,
            'final_validation': final, 'two_initialized_holdout': before['metrics'],
            'two_holdout_after': after['metrics'], 'one_before': old['metrics'], 'one_after': regression['metrics'],
            'heldout_gate': gate_result(after['metrics'], two['gate']),
            'second_slot_sensitivity': second_slot_sensitivity(chosen, sample),
            'candidate_path': str(chosen_path), 'candidate_sha256': sha(chosen_path),
            'original_actor_frozen': True, 'one_trajectories_identical': True,
            'new_test_used_for_selection': False, 'default_model_preserved': True})
        status('complete', status='completed', two_test=after['metrics']['stage_target'])
    except Exception as exc:
        status('failed', status='failed', error=repr(exc)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/two_static_adapter.yaml')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1); study = resolve(args.config)
    print({'mode': 'run' if args.run else 'check_only', 'bc_updates': study['bc']['updates'],
           'maximum_ppo_steps': study['ppo_steps']}, flush=True)
    if args.run: run(study)


if __name__ == '__main__': main()
