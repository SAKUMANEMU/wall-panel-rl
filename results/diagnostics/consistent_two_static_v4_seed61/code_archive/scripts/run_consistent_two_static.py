"""Broaden failing training layouts, audit RRT ambiguity, then bounded learning."""
import _bootstrap
import argparse
import copy
from datetime import datetime
import hashlib
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
from run_two_static_adapter import trace_policy, correction_handoffs, second_slot_sensitivity
from wall_panel_rl.rl.consistent_teacher import FixedRouteTeacher, canonical_suffix, path_diagnostics, select_canonical
from wall_panel_rl.rl.avoidance_curriculum import stage_scene, gate_result
from wall_panel_rl.rl.avoidance_imitation import save_teacher, AlignmentTeacher
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import make_rrt, evaluate_controller
from wall_panel_rl.rl.multi_obstacle_adapter import initialize_adapter, frozen_base_preserved
from wall_panel_rl.rl.multi_obstacle_warmup import teacher_arrays, warmup_actor


def learned_gain(candidate, initial):
    a, b = candidate['stage_target'], initial['stage_target']
    return ((a['success'] >= b['success'] and a['collision'] < b['collision']) or
            (a['success'] > b['success'] and a['collision'] <= b['collision']))


def no_regression(candidate, initial):
    a, b = candidate['stage_target'], initial['stage_target']
    return a['success'] >= b['success'] and a['collision'] <= b['collision'] and a['joint_limit'] == 0


def coverage_score(result, initial):
    retained = result['ppo_ready'] and no_regression(result['two'], initial['two']) and no_regression(result['legacy_two'], initial['legacy_two'])
    a, b = result['two']['stage_target'], result['legacy_two']['stage_target']
    return [2 if result['eligible'] and retained else int(retained),
            -a['collision'], a['success'], -b['collision'], b['success'], -a['timeout']]


def coverage_validation(model, config, two, one, cases, one_cases, legacy_cases):
    result, rows, one_rows = validation(model, config, two, one, cases, one_cases)
    old = evaluate_controller(model, config, two, legacy_cases, mode='ppo')
    result['legacy_two'] = old['metrics']
    result['eligible'] = result['eligible'] and gate_result(old['metrics'], two['gate'])['passed']
    return result, {'new_validation': rows, 'one_validation': one_rows, 'legacy_two': old['rows']}


def collect(study, config, two, scenes, initial, failed, output, status, resume=False):
    """One canonical route per layout; successful suffixes share its exact tail."""
    output = Path(output); output.mkdir(parents=True, exist_ok=resume)
    env = ConditionalRecoveryEnv(copy.deepcopy(config), [], two)
    strength = [two['active_stage']['amplitude'], two['active_stage']['speed']]
    anchor = AlignmentTeacher(initial)
    audits, attempts, jobs, canonical, route_audits = [], [], [], {}, []
    obs_rows, labels, raw, offsets, seeds, groups, terminal, terminal_labels = [], [], [], [0], [], [], [], []
    def save():
        save_teacher(output/'teacher.npz', obs_rows, labels, raw, offsets, seeds, groups, terminal, terminal_labels)
        dump(output/'attempts.json', attempts); dump(output/'layout_audit.json', audits)
        dump(output/'route_consistency.json', route_audits)
        dump(output/'pending_jobs.json', [{**j, 'prefix': np.asarray(j['prefix']).tolist(),
            'expected': np.asarray(j['expected']).tolist()} for j in jobs])
    def queue(scene, traces, digest):
        seen = set()
        for name, trace in traces.items():
            if trace['reason'] == 'success': continue
            for phase, handoff in enumerate(correction_handoffs(len(trace['states']), study['handoff_leads'])):
                expected = trace['states'][handoff]['observation']
                identity = (handoff, expected.round(6).tobytes())
                if identity in seen: continue
                seen.add(identity)
                jobs.append({'seed': scene['seed'], 'scene': scene, 'phase': phase,
                    'prefix': [r['action'] for r in trace['states'][:handoff]], 'expected': expected,
                    'source_policy': name, 'original_failure': trace['reason'], 'canonical_sha256': digest})
    def execute(scene, path, prefix, expected, kind, extra):
        obs, info = env.reset(options={'scenario': scene, 'task': 'static', 'strength': strength})
        for action in prefix:
            obs, _, done, truncated, info = env.step(action)
            if done or truncated: raise AssertionError('Saved correction prefix ended before handoff')
        error = float(np.max(np.abs(obs-expected))) if expected is not None else 0.
        if error > 1e-6: raise AssertionError('Correction prefix replay mismatch')
        suffix_index = None
        if prefix:
            path, suffix_index = canonical_suffix(env.env.joint_state()[0], path, env.env.collision.edge_valid)
        teacher = FixedRouteTeacher(env.env, path)
        x, y, z = [], [], []
        if teacher.planning_success:
            while True:
                command = teacher.action(obs); previous = obs.copy()
                obs, _, done, truncated, info = env.step(command)
                x.append(previous); y.append(info['executed_action']); z.append(command.copy())
                if done or truncated: break
            reason = info['termination_reason']
        else: reason = 'unsafe_canonical_edge' if teacher.invalid_edge_index is not None else 'no_safe_canonical_connector'
        record = {'seed': scene['seed'], 'scene': scene, 'kind': kind,
            'prefix_actions': np.asarray(prefix).tolist(), 'prefix_replay_error': error,
            'suffix_index': suffix_index, 'reason': reason, 'steps': env.env.steps,
            'saved_episode': len(seeds) if reason == 'success' else None, **extra}
        attempts.append(record)
        if reason == 'success':
            obs_rows.extend(x); labels.extend(y); raw.extend(z); offsets.append(len(obs_rows))
            seeds.append(scene['seed']); groups.append('two_static'); terminal.append(obs.copy()); terminal_labels.append(anchor.action(obs))
        return reason
    try:
        if resume:
            with np.load(output/'teacher.npz', allow_pickle=False) as z:
                obs_rows=z['obs'].tolist(); labels=z['actions'].tolist(); raw=z['raw_actions'].tolist()
                offsets=z['offsets'].tolist(); seeds=z['seeds'].tolist(); groups=z['tasks'].tolist()
                terminal=z['terminal_obs'].tolist(); terminal_labels=z['terminal_actions'].tolist()
            audits=read(output/'layout_audit.json'); attempts=read(output/'attempts.json'); route_audits=read(output/'route_consistency.json')
            successful={a['seed'] for a in attempts if a['kind']=='canonical' and a['reason']=='success'}
            canonical={r['seed']:np.asarray(r['selected_path']) for r in route_audits if r['seed'] in successful}
            if (output/'pending_jobs.json').exists(): jobs=read(output/'pending_jobs.json')
            else:
                by_seed={s['seed']:s for s in scenes}; models={'initialized':initial,'failed_v3':failed}
                for audit in audits:
                    if audit['seed'] not in canonical or not audit['failure_union']: continue
                    scene=stage_scene(by_seed[audit['seed']],two['active_stage'])
                    traces={name:trace_policy(env,model,scene,strength) for name,model in models.items() if audit[name]['reason']!='success'}
                    if any(t['reason']!=audit[name]['reason'] for name,t in traces.items()): raise AssertionError('Saved mining policy no longer reproduces')
                    digest=next(r['canonical_sha256'] for r in route_audits if r['seed']==audit['seed'])
                    queue(scene,traces,digest)
            if [s['seed'] for s in scenes[:len(audits)]] != [a['seed'] for a in audits]: raise AssertionError('Resume layout prefix changed')
            status('resume_canonical',saved_layouts=len(audits),saved_episodes=len(seeds)); save()
        completed={a['seed'] for a in audits}
        for i, original in enumerate(scenes):
            if original['seed'] in completed: continue
            scene = stage_scene(original, two['active_stage'])
            traces = {name: trace_policy(env, model, scene, strength) for name, model in [('initialized', initial), ('failed_v3', failed)]}
            audits.append({'seed': original['seed'], **{name: {'reason': t['reason'], 'steps': t['steps'],
                'position_m': t['position_m']} for name, t in traces.items()},
                'failure_union': any(t['reason'] != 'success' for t in traces.values())})
            obs, _ = env.reset(options={'scenario': scene, 'task': 'static', 'strength': strength})
            candidates, planner_attempts = [], []
            for j in range(study['expert_routes_per_layout']):
                planner_seed = original['seed'] + 400000 + j*1000
                expert = make_rrt(env, planner_seed)
                planner_attempts.append({'seed': planner_seed, 'success': bool(expert.planning_success), 'planning_seconds': expert.planning_time})
                if expert.planning_success:
                    checked=FixedRouteTeacher(env.env,expert.path)
                    planner_attempts[-1]['edge_recheck_valid']=checked.planning_success
                    planner_attempts[-1]['invalid_edge_index']=checked.invalid_edge_index
                    if checked.planning_success:
                        candidates.append({'seed': planner_seed, 'path': np.asarray(expert.path).copy(), 'first_action': checked.action(obs).copy()})
            if not candidates:
                route_audits.append({'seed': original['seed'], 'planner_attempts': planner_attempts, 'selected_seed': None})
                continue
            index, scores = select_canonical(candidates, scene['reference_joint_path'], initial.predict(obs, deterministic=True)[0])
            chosen = candidates[index]
            digest = hashlib.sha256(chosen['path'].astype('<f8').tobytes()).hexdigest()
            route_audits.append({'seed': original['seed'], 'planner_attempts': planner_attempts,
                **path_diagnostics([v['path'] for v in candidates], [v['first_action'] for v in candidates]),
                'candidate_scores': scores, 'selected_seed': chosen['seed'], 'canonical_sha256': digest,
                'selected_path': chosen['path'].tolist(), 'selected_first_action': chosen['first_action'].tolist()})
            reason = execute(scene, chosen['path'], [], None, 'canonical', {'canonical_sha256': digest})
            if reason == 'success':
                canonical[original['seed']] = chosen['path']
                queue(scene,traces,digest)
            if (i+1) % 8 == 0 or i+1 == len(scenes):
                save(); status('collect_canonical', audited_layouts=i+1, canonical_successes=len(seeds),
                    failing_layouts=sum(a['failure_union'] for a in audits),
                    route_conflict_layouts=sum(a.get('has_opposing_initial_directions', False) for a in route_audits))
        canonical_count = len(seeds)
        jobs.sort(key=lambda j: (j['phase'], j['seed'], j['source_policy']))
        for j, job in enumerate(jobs[:study['maximum_correction_episodes']]):
            execute(job['scene'], canonical[job['seed']], job['prefix'], job['expected'], 'consistent_correction',
                {k: job[k] for k in ('source_policy', 'original_failure', 'canonical_sha256')})
            if (j+1) % 8 == 0:
                save(); status('collect_consistent_corrections', attempted=j+1, successes=len(seeds)-canonical_count)
        save()
        dump(output/'metadata.json', {'completed': True, 'episodes': len(seeds), 'transitions': len(obs_rows),
            'canonical_episodes': canonical_count, 'correction_episodes': len(seeds)-canonical_count,
            'failing_layouts': sum(a['failure_union'] for a in audits),
            'correction_layouts': len({a['seed'] for a in attempts if a['kind'] == 'consistent_correction' and a['reason'] == 'success'}),
            'route_conflict_layouts': sum(a.get('has_opposing_initial_directions', False) for a in route_audits),
            'successful_planner_candidates': sum(sum(p['success'] for p in a['planner_attempts']) for a in route_audits),
            'unsafe_candidate_routes_rejected': sum(p.get('edge_recheck_valid') is False for a in route_audits for p in a['planner_attempts']),
            'attempted_layouts': len(audits), 'attempts': len(attempts),
            'prefixes_are_not_labels': True, 'canonical_tail_reused': True,
            'teacher_sha256': sha(output/'teacher.npz')})
    finally: env.close()


def assemble_teacher(base_path, canonical_root, output, heldout):
    """Remove old multi-route two-box training labels, retain one/zero-box anchors."""
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    with np.load(base_path, allow_pickle=False) as a, np.load(Path(canonical_root)/'teacher.npz', allow_pickle=False) as b:
        selected, seen_held = [], set()
        for i, (seed, task) in enumerate(zip(a['seeds'], a['tasks'])):
            if task != 'two_static': selected.append(i)
            elif int(seed) in heldout and int(seed) not in seen_held:
                selected.append(i); seen_held.add(int(seed))
        rows = np.concatenate([np.arange(a['offsets'][i], a['offsets'][i+1]) for i in selected])
        sizes = [int(a['offsets'][i+1]-a['offsets'][i]) for i in selected]
        offsets = np.r_[0, np.cumsum(sizes)]
        save_teacher(output/'teacher.npz', np.concatenate([a['obs'][rows], b['obs']]),
            np.concatenate([a['actions'][rows], b['actions']]), np.concatenate([a['raw_actions'][rows], b['raw_actions']]),
            np.r_[offsets, b['offsets'][1:]+len(rows)], np.r_[a['seeds'][selected], b['seeds']],
            np.r_[a['tasks'][selected], b['tasks']], np.concatenate([a['terminal_obs'][selected], b['terminal_obs']]),
            np.concatenate([a['terminal_actions'][selected], b['terminal_actions']]))
        meta = {'parent_selected_episodes': len(selected), 'removed_old_two_static_training_episodes': int(sum(a['tasks']=='two_static')-len(seen_held)),
            'episodes': len(selected)+len(b['seeds']), 'transitions': len(rows)+len(b['obs']),
            'correction_priority_start': len(selected)+read(Path(canonical_root)/'metadata.json')['canonical_episodes'],
            'source_sha256': sha(base_path), 'new_sha256': sha(Path(canonical_root)/'teacher.npz'),
            'teacher_sha256': sha(output/'teacher.npz')}
    dump(output/'metadata.json', meta); teacher_arrays(output/'teacher.npz', heldout)
    return meta


def audit_replay(config, two, root):
    root = Path(root); attempts = read(root/'attempts.json'); result = []
    with np.load(root/'teacher.npz', allow_pickle=False) as z: d = {k: z[k].copy() for k in z.files}
    env = ConditionalRecoveryEnv(copy.deepcopy(config), [], two)
    try:
        for kind in ('canonical', 'consistent_correction'):
            successes = [a for a in attempts if a['kind'] == kind and a['reason'] == 'success']
            for record in successes[:2]:
                obs, _ = env.reset(options={'scenario': record['scene'], 'task': 'static',
                    'strength': [two['active_stage']['amplitude'], two['active_stage']['speed']]})
                for action in record['prefix_actions']:
                    obs, _, done, truncated, _ = env.step(action)
                    if done or truncated: raise AssertionError('Prefix replay terminated')
                i = record['saved_episode']; error = 0.
                for k in range(d['offsets'][i], d['offsets'][i+1]):
                    error = max(error, float(np.max(np.abs(obs-d['obs'][k]))))
                    obs, _, done, truncated, info = env.step(d['raw_actions'][k])
                    error = max(error, float(np.max(np.abs(np.asarray(info['executed_action'])-d['actions'][k]))))
                if not (done or truncated) or info['termination_reason'] != 'success' or error > 1e-6:
                    raise AssertionError('Canonical teacher physical replay failed')
                result.append({'kind': kind, 'seed': record['seed'], 'max_error': error, 'reason': info['termination_reason']})
    finally: env.close()
    return result


def run(study, resume=False):
    data, models, report = map(Path, (study['data_root'], study['model_root'], study['report_root']))
    if not resume and any(p.exists() for p in (data, models, report)): raise FileExistsError('Refusing experiment overwrite')
    if resume:
        old=read(report/'protocol.json')
        if read(report/'status.json')['status']!='failed' or (models/'bc').exists(): raise ValueError('Resume is limited to interrupted pre-training collection')
        if old['study']!=study: raise ValueError('Resume configuration changed')
        if sha(study['initial_model'])!=old['source_sha256'] or sha(study['failed_model'])!=old['failed_model_sha256']:
            raise ValueError('Mining/initial source changed during interrupted collection')
        for path, expected in read(report/'data_hashes.json').items():
            if sha(path)!=expected: raise ValueError('Frozen resume data changed')
    parent = read(Path(study['source_report'])/'manifest.json'); original_hash = sha(study['initial_model'])
    if original_hash != parent['model_sha256']: raise ValueError('Wrong retained source')
    held = set(read(study['heldout_metadata'])['heldout_scene_seeds']); forbidden = known_scene_seeds()
    for p in (data, models, report): p.mkdir(parents=True, exist_ok=resume)
    config = copy.deepcopy(parent['configuration']); one = stage(parent['evaluation_settings'], 1); two = stage(one, 2)
    state = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(),
        'runner_pid': os.getpid(), 'default_model_preserved': True, 'automatic_dynamic_stage': False}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report/'status.json', state); print('CONSISTENT_TWO', json.dumps(state), flush=True)
    paths = ['configs/consistent_two_static.yaml', 'scripts/run_consistent_two_static.py', 'scripts/train_consistent_two_static.py',
        'src/wall_panel_rl/rl/consistent_teacher.py', 'src/wall_panel_rl/rl/multi_obstacle_adapter.py',
        'src/wall_panel_rl/rl/multi_obstacle_warmup.py', 'src/wall_panel_rl/rl/episode_imitation.py']
    dump(report/'protocol.json', {'study': study, 'source_sha256': original_hash, 'failed_model_sha256': sha(study['failed_model']),
        'code_hashes': {p: sha(p) for p in paths}, 'new_test_used_for_selection': False,
        'canonical_selection_cost': 'joint length +0.5 reference departure +0.25 initial base-action MSE',
        'label_rule': 'one selected route per layout; checked connector plus literal canonical tail for corrections',
        'selection': 'new64 validation and legacy20 non-regression; old one/zero obstacle preservation; independent100 test last',
        'collection_resumed': resume, 'original_collection_archive': 'interrupted_collection_1' if resume else None})
    try:
        if not resume:
            banks = {}
            for name, count, seed in [('new_train', study['new_train_count'], study['new_train_seed_start']),
                                      ('validation', study['validation_count'], study['validation_seed_start']),
                                      ('test', study['test_count'], study['test_seed_start'])]:
                spec = {'fresh_seed_start': seed, 'fresh_targets': count, 'maximum_generation_attempts': count*study['generation_attempt_factor']}
                status('freeze_'+name)
                full, rejects = freeze_fresh(spec, config, one, forbidden, status)
                forbidden.update(s['seed'] for s in full); banks[name] = full
                dump(data/(name+'_full.json'), full); dump(data/(name+'_rejections.json'), rejects)
            scenes = [s for s in read(study['raw_train']) if s['seed'] not in held] + banks['new_train']
            dump(data/'train.json', scenes)
            cases = project_cases(banks['validation'], two['active_stage']); dump(data/'validation_cases.json', cases)
            dump(data/'test_cases.json', project_cases(banks['test'], two['active_stage']))
            dump(report/'split.json', {'train_seeds': [s['seed'] for s in scenes], 'heldout_teacher_seeds': sorted(held),
                'validation_seeds': [s['seed'] for s in banks['validation']], 'test_seeds': [s['seed'] for s in banks['test']]})
        else:
            scenes=read(data/'train.json'); cases=read(data/'validation_cases.json')
        one_cases = read(study['one_validation']); legacy = read(study['legacy_validation'])
        dump(report/'data_hashes.json', {str(p): sha(p) for p in
            (data/'train.json', data/'validation_cases.json', data/'test_cases.json', Path(study['legacy_validation']), Path(study['one_validation']))})
        if not resume:
            initial = initialize_adapter(study['initial_model'], study, study['seed']); initial.save(models/'initialized_model.zip')
        else:
            initial=PPO.load(models/'initialized_model.zip',device='cpu');initial.policy.configure_learning(train_critic=False)
        reference = PPO.load(models/'initialized_model.zip', device='cpu')
        if not resume:
            baseline, baseline_rows = coverage_validation(initial, config, two, one, cases, one_cases, legacy)
            dump(report/'baseline.json', {'summary': baseline, 'rows': baseline_rows})
        else: baseline=read(report/'baseline.json')['summary']
        status('collect_training', initial_new_validation=baseline['two']['stage_target'], initial_legacy_validation=baseline['legacy_two']['stage_target'])
        collect(study, config, two, scenes, initial, PPO.load(study['failed_model'], device='cpu'), data/'canonical', status,resume=resume)
        dump(report/'teacher_replay.json', audit_replay(config, two, data/'canonical'))
        assembled = assemble_teacher(study['base_teacher'], data/'canonical', data/'teacher', held)
        teacher, parts = teacher_arrays(data/'teacher/teacher.npz', held)
        rows = np.concatenate([np.arange(teacher['offsets'][i], teacher['offsets'][i+1]) for i in parts['train']['two_static']])
        sample = teacher['obs'][rows]
        def evaluator(candidate, update):
            if not frozen_base_preserved(reference.policy, candidate.policy): raise AssertionError('Original actor changed')
            result, details = coverage_validation(candidate, config, two, one, cases, one_cases, legacy)
            result['score'] = coverage_score(result, baseline)
            result['second_slot_sensitivity'] = second_slot_sensitivity(candidate, sample)
            dump(report/'bc_validation'/f'update_{update}.json', {'summary': result, 'rows': details})
            status('consistent_bc', bc_updates=update, new_validation=result['two']['stage_target'], legacy_validation=result['legacy_two']['stage_target'])
            return result
        spec = dict(study['bc'], seed=study['seed'], priority_episode_start=assembled['correction_priority_start'], priority_episode_weight=4.)
        warm, metadata = warmup_actor(initial, data/'teacher/teacher.npz', held, spec, models/'bc', evaluator)
        selected = metadata['selected']; chosen_path = warm; ppo_started = False
        learning_gate = (selected['update'] > 0 and selected['ppo_ready'] and learned_gain(selected['two'], baseline['two'])
            and no_regression(selected['legacy_two'], baseline['legacy_two'])
            and selected['second_slot_sensitivity']['max_action_change_15cm'] > 1e-5)
        dump(report/'bc_learning_gate.json', {'passed': learning_gate, 'selected': selected})
        if learning_gate:
            from train_consistent_two_static import train
            bundle = copy.deepcopy(parent)
            for key in ('training_settings', 'evaluation_settings'): bundle[key] = stage(bundle[key], 2)
            bundle['model_sha256'] = sha(warm); dump(data/'warm_source/manifest.json', bundle)
            ppo_spec = dict(study, source_report=str(data/'warm_source'), initial_model=str(warm),
                train_scenes=str(data/'train.json'), heldout_scenes=study['heldout_metadata'],
                validation_cases=str(data/'validation_cases.json'), output_root=str(models/'ppo_run'), report_root=str(report/'ppo_run'))
            (data/'ppo_study.yaml').write_text(yaml.safe_dump(ppo_spec, sort_keys=False))
            status('consistent_ppo', selected_bc_update=selected['update'])
            chosen_path = train(ppo_spec, one, one_cases, legacy, baseline, data/'teacher/teacher.npz', held, status)
            ppo_started = True
        else: status('bc_learning_gate_failed', selected_bc_update=selected['update'])
        chosen = PPO.load(chosen_path, device='cpu')
        if not frozen_base_preserved(reference.policy, chosen.policy): raise AssertionError('Original actor changed')
        final, _ = coverage_validation(chosen, config, two, one, cases, one_cases, legacy)
        shutil.copy2(chosen_path, report/'candidate_model.zip')
        dump(report/'candidate_manifest.json', {'configuration': config, 'evaluation_settings': two,
            'model_sha256': sha(chosen_path), 'validation_eligible': final['eligible'], 'default_model_preserved': True})
        status('final_new_test', candidate_sha256=sha(chosen_path), ppo_started=ppo_started)
        tests = read(data/'test_cases.json')
        before = evaluate_controller(reference, config, two, tests, mode='ppo')
        after = evaluate_controller(chosen, config, two, tests, mode='ppo')
        before['independent_test'] = after['independent_test'] = True
        one_before = evaluate_controller(reference, config, one, read(study['one_regression']), mode='ppo')
        one_after = evaluate_controller(chosen, config, one, read(study['one_regression']), mode='ppo')
        if one_before['rows'] != one_after['rows']: raise AssertionError('Protected one-obstacle trajectories changed')
        for name, result in [('test_initialized', before), ('test_candidate', after), ('one_before', one_before), ('one_after', one_after)]:
            dump(report/f'{name}.json', result)
        if sha(study['initial_model']) != original_hash: raise AssertionError('Source file changed')
        dump(report/'summary.json', {'baseline': baseline, 'selected_bc': selected, 'bc_updates_run': study['bc']['updates'],
            'ppo_started': ppo_started, 'ppo_steps': study['ppo_steps'] if ppo_started else 0, 'final_validation': final,
            'test_initialized': before['metrics'], 'test_candidate': after['metrics'],
            'one_before': one_before['metrics'], 'one_after': one_after['metrics'],
            'test_gate': gate_result(after['metrics'], two['gate']), 'candidate_path': str(chosen_path),
            'candidate_sha256': sha(chosen_path), 'second_slot_sensitivity': second_slot_sensitivity(chosen, sample),
            'source_unchanged': True, 'default_model_preserved': True, 'one_trajectories_identical': True,
            'new_test_used_for_selection': False})
        status('complete', status='completed', test=after['metrics']['stage_target'])
    except Exception as exc:
        status('failed', status='failed', error=repr(exc)); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/consistent_two_static.yaml'); parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume-collection',action='store_true',help='Resume only an interrupted pre-training collection')
    args = parser.parse_args(); torch.set_num_threads(1); study = yaml.safe_load(Path(args.config).read_text())
    print({'mode': 'run' if args.run else 'check_only', 'new_training_layouts': study['new_train_count'],
        'new_validation_layouts': study['validation_count'], 'new_test_layouts': study['test_count'],
        'bc_updates': study['bc']['updates'], 'maximum_ppo_steps': study['ppo_steps']}, flush=True)
    if args.resume_collection and not args.run: parser.error('--resume-collection requires --run')
    if args.run: run(study,resume=args.resume_collection)


if __name__ == '__main__': main()
