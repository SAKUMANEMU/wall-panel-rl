"""Replay13 collisions, then freeze larger disjoint four-static scene banks."""
import _bootstrap
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from run_static_expansion import known_scene_seeds
from wall_panel_rl.rl.rrt_validation import run_episode
from wall_panel_rl.rl.static_demo_scene import fresh_static_case


def run(spec):
    torch.set_num_threads(1)
    data, report = Path(spec['data_root']), Path(spec['preparation_report'])
    if data.exists() or report.exists(): raise FileExistsError('Refusing bank overwrite')
    source = Path(spec['source_report']); model_hash = sha(spec['source_model'])
    manifest = read(source/'candidate_manifest.json'); protocol = read(source/'protocol.json')
    if model_hash != manifest['model_sha256']: raise ValueError('Wrong PPO10000 source')
    config, settings = protocol['configuration'], protocol['evaluation_settings']
    model = PPO.load(spec['source_model'], device='cpu')
    old = Path(spec['old_data_root']); old_train = read(old/'train_cases.json')
    old_val, old_test = read(old/'validation_cases.json'), read(old/'test_cases.json')
    seen = known_scene_seeds(); data.mkdir(parents=True); report.mkdir(parents=True)
    state = {'status':'running', 'started_at':datetime.now().astimezone().isoformat(), 'training_steps':0}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report/'status.json', state); print('EXPANDED_FOUR', json.dumps(state), flush=True)
    dump(report/'protocol.json', {'spec':spec, 'configuration':config, 'evaluation_settings':settings,
        'source_model_sha256':model_hash, 'existing_scene_seeds_excluded_from_new_generation':sorted(seen),
        'code_hashes':{'scripts/prepare_expanded_four.py':sha(__file__),
                       'src/wall_panel_rl/rl/static_demo_scene.py':sha('src/wall_panel_rl/rl/static_demo_scene.py'),
                       'src/wall_panel_rl/rl/on_path_obstacles.py':sha('src/wall_panel_rl/rl/on_path_obstacles.py')},
        'geometry_rule':'same four nominal-motion blockers; endpoints valid; no with-obstacle outcome filtering',
        'old64_test_role':'previously used test, now regression/diagnostic only', 'training_steps':0})
    try:
        cached = {r['seed']:r for r in read(source/'test_selected.json')['rows']}
        rows = []
        for case in old_test:
            expected = cached[case['scene']['seed']]
            if expected['reason'] != 'collision': continue
            row, trace = run_episode(model, config, settings, case, 'ppo')
            assert row['reason'] == expected['reason'] and row['steps'] == expected['steps']
            contact = row['dominant_contact']; fraction = contact.get('nominal_path_fraction') if contact else None
            row['contact_stage'] = 'unknown' if fraction is None else ('early' if fraction < .35 else 'late' if fraction > .65 else 'middle')
            rows.append(row); dump(report/'collision_traces'/f'{row["seed"]}.json', trace)
        assert len(rows) == 13
        dump(report/'collisions.json', rows)
        diagnosis = {'episodes':len(rows), 'dominant_body':dict(Counter(r['dominant_contact']['moving_body'] if r['dominant_contact'] else 'unknown' for r in rows)),
            'dominant_slot':dict(Counter(str(r['dominant_contact']['obstacle_slot']) if r['dominant_contact'] else 'unknown' for r in rows)),
            'contact_stage':dict(Counter(r['contact_stage'] for r in rows)), 'step_range':[min(r['steps'] for r in rows),max(r['steps'] for r in rows)],
            'all_reproduced':True, 'diagnostic_only_no_test_actions_used_as_labels':True}
        dump(report/'collision_summary.json', diagnosis)
        flat = [{'seed':r['seed'], 'step':r['steps'], 'stage':r['contact_stage'],
                 **{k:(r['dominant_contact'] or {}).get(k) for k in ['kind','moving_body','obstacle_slot','moving_link_index','distance_m','nominal_path_fraction']}} for r in rows]
        with (report/'collisions.csv').open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
        status('collision_diagnosis_complete', diagnosis=diagnosis)
        banks = {}; rejected = []
        for name, count, first in [('fresh_train',spec['fresh_train_count'],spec['fresh_train_seed']),
                                   ('validation',spec['validation_count'],spec['validation_seed']),
                                   ('test',spec['test_count'],spec['test_seed'])]:
            bank = []
            for seed in range(first, first+count*spec['generation_factor']):
                if seed in seen: raise ValueError('Repeated scene seed')
                try: case,_ = fresh_static_case(config, settings, seed, model=model)
                except RuntimeError as exc:
                    rejected.append({'bank':name,'seed':seed,'reason':str(exc)}); continue
                bank.append(case); seen.add(seed)
                if len(bank)%16==0: status('freeze_'+name, accepted=len(bank), target=count, rejected=len(rejected))
                if len(bank)==count: break
            if len(bank)!=count: raise RuntimeError('Scene generation budget exhausted')
            banks[name] = bank; dump(data/f'{name}_cases.json', bank)
        banks['train'] = old_train+banks['fresh_train']
        banks['old_validation'] = old_val; banks['old_test_regression'] = old_test
        for name in ('train','old_validation','old_test_regression'): dump(data/f'{name}_cases.json', banks[name])
        splits = {k:sorted(c['scene']['seed'] for c in bank) for k,bank in banks.items()}
        assert len(splits['train'])==192 and len(set(splits['train']))==192
        for a in ('train','validation','test','old_validation','old_test_regression'):
            for b in ('train','validation','test','old_validation','old_test_regression'):
                if a!=b: assert not set(splits[a])&set(splits[b])
        dump(data/'split.json', splits); dump(data/'generation_rejections.json', rejected)
        dump(report/'data_hashes.json', {str(p):sha(p) for p in data.glob('*.json')})
        assert sha(spec['source_model'])==model_hash
        dump(report/'summary.json', {'bank_counts':{k:len(v) for k,v in banks.items()}, 'collision_diagnosis':diagnosis,
             'generation_rejections':len(rejected), 'splits_disjoint':True, 'source_model_unchanged':True, 'training_steps':0})
        shutil.copy2(__file__, report/'executed_preparation.py')
        status('complete', status='completed', bank_counts={k:len(v) for k,v in banks.items()})
    except Exception as exc:
        status('failed',status='failed',error=repr(exc)); raise


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',default='configs/four_expanded_ppo.yaml'); p.add_argument('--run',action='store_true')
    args=p.parse_args(); spec=yaml.safe_load(Path(args.config).read_text())
    print({'mode':'run' if args.run else 'check_only','train':192,'validation':64,'test':100},flush=True)
    if args.run: run(spec)


if __name__=='__main__': main()
