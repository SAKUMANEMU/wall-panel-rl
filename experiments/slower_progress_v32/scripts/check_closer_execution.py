"""Matched development probes before model adaptation; no success filtering.

Uses the first training scene of each requested count, never validation/test
outcomes to tune the controller. Both modes use identical cases and weights.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import torch
from run_adaptation import save,Status
from run_variable_training import summary
from wall_panel_rl.submission import load_runtime,load_json,load_agent,run_episode
from wall_panel_rl.utils.config import ROOT


def probe(job):
    label,case,output=job
    torch.set_num_threads(1)
    runtime=load_runtime()
    if label=='installation_controller':runtime=load_runtime(ROOT/'configs/installation_runtime.json')
    if label=='v29_controller':
        runtime['controller']=load_json(ROOT/'configs/v29_baseline_controller.json')
    result,recording=run_episode(load_agent(),case,'enhanced',runtime)
    result.update(obstacle_count=len(case['scene']['obstacles']),execution_variant=label)
    save(Path(output)/'recordings'/label/f"{case['scene']['seed']}.json",recording)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--counts',nargs='+',type=int,default=[2,3,4,5])
    parser.add_argument('--variants',nargs='+',choices=['v29_controller','installation_controller'],
                        default=['v29_controller','installation_controller'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    bank=load_json(args.scenes/'train_cases.json')
    cases=[next(c for c in bank if len(c['scene']['obstacles'])==count) for count in args.counts]
    save(args.output/'cases.json',cases)
    status=Status(args.output);rows=[]
    jobs=[(label,case,str(args.output)) for label in args.variants for case in cases]
    status.update('matched_physical_probes',completed=0,planned=len(jobs),training_started=False)
    with ProcessPoolExecutor(2,mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(probe,job) for job in jobs]):
            rows.append(future.result())
            metrics={label:summary([r for r in rows if r['execution_variant']==label])
                     for label in args.variants}
            save(args.output/'results.json',dict(rows=rows,metrics=metrics,complete=len(rows)==len(jobs)))
            status.update('matched_physical_probes',completed=len(rows),planned=len(jobs),latest=rows[-1])
    status.update('complete',status='completed')


if __name__=='__main__':main()
