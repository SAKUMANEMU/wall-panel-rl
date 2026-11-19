"""Training-only corrective suffixes along each layout's saved expert route."""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from run_two_static_adapter import trace_policy, correction_handoffs
from wall_panel_rl.rl.avoidance_imitation import save_teacher, AlignmentTeacher
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.consistent_teacher import canonical_suffix, FixedRouteTeacher
from wall_panel_rl.planning.static_validation import execution_edge_valid, dense_route_valid


DATA = Path('data/four_policy_corrections_v3_seed61')
REPORT = Path('results/diagnostics/four_policy_corrections_v3_seed61')
SOURCE = Path('results/diagnostics/four_safe_teacher_v1_seed61')


def route_from_teacher(d, i, env, endpoint, commands=False):
    lo, hi = d['offsets'][i:i+2]
    span = env.upper - env.lower
    states = env.lower + (d['obs'][lo:hi, :7].astype(float)+1)*.5*span
    if commands: tail = states + d['raw_actions'][lo:hi]*env.max_increment
    else: tail = np.vstack([states[1:], env.lower+(d['terminal_obs'][i, :7].astype(float)+1)*.5*span])
    path = np.vstack([states[:1], tail, endpoint])
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1)>1e-7]
    return path[keep]


def run():
    torch.set_num_threads(1)
    if DATA.exists() or REPORT.exists(): raise FileExistsError('Refusing correction overwrite')
    tp = read(SOURCE / 'protocol.json'); splits = read(SOURCE / 'split.json')
    cases = read(Path(tp['spec']['data_root']) / 'train_cases.json')
    with np.load(Path(tp['spec']['data_root']) / 'teacher/teacher.npz', allow_pickle=False) as z:
        base = {k: z[k].copy() for k in z.files}
    assert set(base['seeds'].tolist()) == set(splits['train'])
    assert not set(splits['train']) & set(splits['validation'] + splits['test'])
    config, settings = tp['configuration'], tp['evaluation_settings']
    paths = {'bc1000': 'results/diagnostics/four_obstacle_bc_v1_seed61/final_candidate.zip',
             'bc5000': 'results/diagnostics/four_priority_bc_v2_seed61/final_candidate.zip'}
    models = {name: PPO.load(path, device='cpu') for name, path in paths.items()}
    hashes = {p: sha(p) for p in paths.values()}
    DATA.mkdir(parents=True); REPORT.mkdir(parents=True)
    state = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'training_steps': 0}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(REPORT / 'status.json', state); print('FOUR_CORRECTIONS', json.dumps(state), flush=True)
    dump(REPORT / 'protocol.json', {'configuration': config, 'evaluation_settings': settings,
         'spec': tp['spec'], 'source_models': hashes, 'only_train_seeds': splits['train'],
         'label_rule': 'verified connector then literal saved expert-route tail; successful actual execution only',
         'head_leads': [2, 6], 'planning_clearance_m': config['collision']['planning_clearance'],
         'code_sha256': sha(__file__), 'training_steps': 0})
    dump(REPORT / 'split.json', splits); dump(REPORT / 'data_hashes.json', read(SOURCE / 'data_hashes.json'))
    env = ConditionalRecoveryEnv(copy.deepcopy(config), [], settings); anchor = AlignmentTeacher(models['bc1000'])
    observations, labels, raw, offsets, seeds, terminals, anchors = [], [], [], [0], [], [], []
    attempts, audits, routes = [], [], []
    by_seed = {int(seed): i for i, seed in enumerate(base['seeds'])}
    def save():
        save_teacher(DATA / 'corrections.npz', observations, labels, raw, offsets, seeds,
                     ['four_static']*len(seeds), terminals, anchors)
        dump(DATA / 'attempts.json', attempts); dump(DATA / 'audit.json', audits); dump(DATA / 'canonical_routes.json', routes)
    try:
        for n, case in enumerate(cases):
            scene = case['scene']; seed = int(scene['seed']); i = by_seed[seed]
            env.reset(options={'scenario': scene, 'task': 'static', 'strength': case['strength']})
            path = None; method = None
            for commands in (False, True):
                candidate = route_from_teacher(base, i, env.env, env.cursor.path[-1], commands)
                if dense_route_valid(env.env.collision, candidate, .005):
                    path, method = candidate, 'recovered_commands' if commands else 'physical_teacher_states'; break
            routes.append({'seed': seed, 'verified': path is not None, 'method': method,
                           'path': path.tolist() if path is not None else None})
            seen = set()
            for name, model in models.items():
                trace = trace_policy(env, model, scene, case['strength'])
                audits.append({'seed': seed, 'source': name, 'reason': trace['reason'], 'steps': trace['steps'], 'canonical_verified': path is not None})
                if trace['reason'] == 'success' or path is None: continue
                for handoff in correction_handoffs(len(trace['states']), [2, 6]):
                    expected = trace['states'][handoff]['observation']
                    identity = (handoff, expected.round(6).tobytes())
                    if identity in seen: continue
                    seen.add(identity)
                    prefix = [s['action'] for s in trace['states'][:handoff]]
                    obs, info = env.reset(options={'scenario': scene, 'task': 'static', 'strength': case['strength']})
                    reference = env.cursor.path.copy()
                    for action in prefix:
                        obs, _, done, truncated, info = env.step(action)
                        if done or truncated: raise AssertionError('Prefix ended before safe handoff')
                    error = float(np.max(np.abs(obs-expected)))
                    if error > 1e-6: raise AssertionError('Prefix observation mismatch')
                    checker = env.env.collision
                    suffix, index = canonical_suffix(env.env.joint_state()[0], path,
                        lambda a, b: execution_edge_valid(checker, a, b, env.env.max_increment, .005))
                    teacher = FixedRouteTeacher(env.env, suffix)
                    verified = teacher.planning_success and dense_route_valid(checker, teacher.path, .005)
                    x, y, z = [], [], []
                    if verified:
                        while True:
                            command = teacher.action(obs); previous = obs.copy()
                            obs, _, done, truncated, info = env.step(command)
                            x.append(previous); y.append(info['executed_action']); z.append(command.copy())
                            if done or truncated: break
                        reason = info['termination_reason']
                        post = dense_route_valid(checker, teacher.path, .005)
                    else: reason, post = 'no_verified_connector', False
                    assert np.array_equal(reference, env.cursor.path)
                    accepted = bool(verified and post and reason == 'success')
                    attempts.append({'seed': seed, 'source': name, 'handoff': handoff, 'prefix_actions': np.asarray(prefix).tolist(),
                        'prefix_error': error, 'suffix_index': index, 'pre_verified': bool(verified), 'post_verified': bool(post),
                        'reason': reason, 'accepted': accepted, 'saved_episode': len(seeds) if accepted else None})
                    if accepted:
                        observations.extend(x); labels.extend(y); raw.extend(z); offsets.append(len(observations))
                        seeds.append(seed); terminals.append(obs.copy()); anchors.append(anchor.action(obs))
            save(); status('collect', completed_layouts=n+1, correction_episodes=len(seeds), attempts=len(attempts))
    finally: env.close()
    assert set(seeds).issubset(set(splits['train']))
    # Original expert episodes stay intact; append correction episodes explicitly.
    with np.load(DATA / 'corrections.npz', allow_pickle=False) as z: extra = {k: z[k].copy() for k in z.files}
    merged = {k: np.concatenate([base[k], extra[k]]) for k in base if k != 'offsets'}
    merged['offsets'] = np.r_[base['offsets'], extra['offsets'][1:] + len(base['obs'])]
    save_teacher(DATA / 'teacher.npz', merged['obs'], merged['actions'], merged['raw_actions'], merged['offsets'],
        merged['seeds'], merged['tasks'], merged['terminal_obs'], merged['terminal_actions'])
    with np.load(DATA / 'teacher.npz', allow_pickle=False) as z:
        assert all(np.array_equal(z[k][:len(v)], v) for k, v in base.items())
    # Replay representative newly collected corrections with their policy prefix.
    checks = []; env = ConditionalRecoveryEnv(copy.deepcopy(config), [], settings)
    by_case = {c['scene']['seed']: c for c in cases}
    try:
        for record in [a for a in attempts if a['accepted']][:3]:
            case = by_case[record['seed']]; obs, _ = env.reset(options={'scenario': case['scene'], 'task': 'static', 'strength': case['strength']})
            for command in record['prefix_actions']: obs, _, _, _, _ = env.step(command)
            i = record['saved_episode']; error = 0.
            for k in range(extra['offsets'][i], extra['offsets'][i+1]):
                error = max(error, float(np.max(np.abs(obs-extra['obs'][k]))))
                obs, _, done, truncated, info = env.step(extra['raw_actions'][k])
                error = max(error, float(np.max(np.abs(np.asarray(info['executed_action'])-extra['actions'][k]))))
            assert (done or truncated) and info['termination_reason'] == 'success' and error <= 1e-6
            checks.append({'seed': record['seed'], 'maximum_error': error, 'reason': info['termination_reason']})
    finally: env.close()
    for path, expected_hash in hashes.items(): assert sha(path) == expected_hash
    meta = {'episodes': len(merged['seeds']), 'transitions': len(merged['obs']), 'correction_episodes': len(seeds),
        'corrected_layouts': len(set(seeds)), 'teacher_sha256': sha(DATA / 'teacher.npz'), 'ready_for_bc': bool(seeds),
        'original64_arrays_preserved': True, 'training_steps': 0}
    dump(DATA / 'metadata.json', meta); dump(REPORT / 'summary.json', {'teacher': meta, 'replay': checks,
        'split_disjoint': True, 'validation_test_not_used': True, 'training_steps': 0})
    shutil.copy2(__file__, REPORT / 'collector.py'); status('complete', status='completed', **meta)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--run', action='store_true'); args = p.parse_args()
    print({'mode': 'run' if args.run else 'check_only', 'training_steps': 0}, flush=True)
    if args.run: run()


if __name__ == '__main__': main()
