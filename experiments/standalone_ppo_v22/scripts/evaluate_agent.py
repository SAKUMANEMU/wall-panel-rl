"""Evaluate unchanged agents on frozen cases; all failures stay in the denominator."""
import _bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.submission import load_agent, load_json, run_episode, save_json

WORK = {}


def initialize(model_path, output, save_recordings):
    torch.set_num_threads(1)
    WORK.update(model=load_agent(model_path), output=Path(output), save_recordings=save_recordings)


def evaluate_one(job):
    mode, case = job
    result, recording = run_episode(WORK['model'], case, mode)
    if WORK['save_recordings']:
        save_json(WORK['output']/'recordings'/f'{mode}_{result["seed"]}.json', recording)
    return result


def summarize(rows):
    counts = {reason:sum(r['reason']==reason for r in rows)
              for reason in ['success','collision','timeout','joint_limit','planning_failure']}
    assert sum(counts.values())==len(rows)
    return {'episodes':len(rows), **counts, 'success_rate':counts['success']/len(rows),
            'success_with_weaving':sum(r['success_with_weaving'] for r in rows),
            'success_and_weaving_rate':sum(r['success_with_weaving'] for r in rows)/len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['ppo','enhanced','rrt','mixture'], default='enhanced')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--bank', type=Path, default=ROOT/'data/validation_cases.json')
    parser.add_argument('--model', type=Path, default=ROOT/'models/current_ppo.zip')
    parser.add_argument('--selection-seed', type=int, default=181)
    parser.add_argument('--save-recordings', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cases = load_json(args.bank)
    if not 1 <= args.episodes <= len(cases) or args.workers<1:
        parser.error('Choose a positive episode count within the bank and at least one worker')
    cases = cases[:args.episodes]
    output = args.output or ROOT/'outputs'/f'evaluation_{time.time_ns()}'
    if output.exists():
        parser.error('Refusing to overwrite an existing evaluation directory')
    output.mkdir(parents=True)
    digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
    rng = np.random.default_rng(args.selection_seed)
    # A mixture owns a whole episode. Failed branches never switch controllers.
    jobs = [('rrt' if rng.random()<.5 else 'enhanced', c) if args.mode=='mixture'
            else (args.mode, c) for c in cases]
    save_json(output/'protocol.json', {'mode':args.mode, 'episodes':args.episodes,
        'model_sha256':digest, 'selection_seed':args.selection_seed,
        'assignments':[{'seed':c['scene']['seed'],'mode':m} for m,c in jobs],
        'training_performed':False, 'source_results_overwritten':False})
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'),
        initializer=initialize, initargs=(str(args.model.resolve()), str(output.resolve()), args.save_recordings)) as pool:
        futures = [pool.submit(evaluate_one, job) for job in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            save_json(output/'results.json', {'metrics':summarize(rows), 'rows':rows})
            print(f'{len(rows)}/{len(cases)} complete: {summarize(rows)}', flush=True)
    assert hashlib.sha256(args.model.read_bytes()).hexdigest()==digest
    print(f'Evaluation saved: {output.resolve()}', flush=True)


if __name__=='__main__':
    main()
