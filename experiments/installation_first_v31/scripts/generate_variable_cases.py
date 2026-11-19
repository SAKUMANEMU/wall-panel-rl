"""Generate balanced, disjoint train/validation/test scene banks by geometry."""
import _bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import torch
import yaml
from run_adaptation import save, Status, digest
from wall_panel_rl.submission import load_agent, load_runtime
from wall_panel_rl.rl.variable_scene import generate_variable_case
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.envs.known_route_motion import upgrade_case
import numpy as np

CACHE = {}


def initialize(config_path):
    torch.set_num_threads(1)
    CACHE.update(model=load_agent(ROOT/'models/source_ppo77.zip'), runtime=load_runtime(), spec=yaml.safe_load(Path(config_path).read_text()))


def generate(job):
    split, count, index, first_seed = job
    attempts = []
    settings = CACHE['spec']['generation']
    for offset in range(settings['max_seed_attempts_per_case']):
        seed = first_seed + offset
        try:
            case = generate_variable_case(CACHE['runtime']['configuration'], CACHE['runtime']['settings'],
                                          settings, seed, count, CACHE['model'])
            case = upgrade_case(case,np.random.default_rng(seed+28000000),settings['speed_range_m_s'])
            case['split'] = split
            attempts.append({'seed': seed, 'geometry_valid': True})
            return dict(split=split, count=count, index=index, case=case, attempts=attempts)
        except (ValueError, RuntimeError) as exc:
            attempts.append({'seed': seed, 'geometry_valid': False, 'reason': str(exc)})
    return dict(split=split, count=count, index=index, case=None, attempts=attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/variable_weaving.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    args.output.mkdir(parents=True)
    spec = yaml.safe_load(args.config.read_text())
    status = Status(args.output)
    jobs = []
    for split_index, split in enumerate(('train', 'validation', 'test')):
        for count in spec['counts']:
            for index in range(spec[f'{split}_per_count']):
                # Separate blocks preserve split/count quotas despite rejections.
                seed = spec['generation']['seed_base'] + split_index*100000 + count*10000 + index*100
                jobs.append((split, count, index, seed))
    save(args.output/'protocol.json', {'geometry_only_selection': True, 'outcomes_with_obstacles_not_filtered': True,
        'configuration': spec, 'source_model_sha256': digest(ROOT/'models/source_ppo77.zip'),
        'scene_count': len(jobs), 'fixed_disjoint_seed_blocks': True})
    save(args.output/'runtime.json', load_runtime())
    results = []
    with ProcessPoolExecutor(spec['workers'], mp_context=mp.get_context('spawn'),
                             initializer=initialize, initargs=(str(args.config),)) as pool:
        futures = {pool.submit(generate, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save(args.output/'attempts.json', [{k: v for k, v in r.items() if k != 'case'} for r in results])
            if result['case'] is None:
                status.update('generation_failed', status='failed', failed_job=futures[future])
                raise RuntimeError('Scene generation exhausted a quota; see attempts.json')
            save(args.output/'cases'/f"{result['case']['scene']['seed']}.json", result['case'])
            status.update('generate_scenes', completed=len(results), planned=len(jobs),
                          latest_count=result['count'], latest_seed=result['case']['scene']['seed'])
    ids = []
    for split in ('train', 'validation', 'test'):
        cases = [r['case'] for r in sorted(results, key=lambda r: (r['count'], r['index'])) if r['split'] == split]
        ids.extend(c['scene']['seed'] for c in cases)
        save(args.output/f'{split}_cases.json', cases)
    assert len(ids) == len(set(ids)), 'Split leakage'
    status.update('complete', status='completed', completed=len(results), planned=len(jobs))


if __name__ == '__main__':
    main()
