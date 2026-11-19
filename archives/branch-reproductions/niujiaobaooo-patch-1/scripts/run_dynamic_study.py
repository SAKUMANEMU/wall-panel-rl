"""Run the user-approved pilot, two exact-budget runs, then six-cell evaluation.

Preparation must already have completed. Status files/logs survive UI disconnects.
This runner never expands budgets or changes rewards based on outcomes.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('models/dynamic70_study')


def status(phase,**kwargs):
    data=dict(phase=phase,updated_unix=time.time(),**kwargs)
    p=ROOT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
    print(data,flush=True)


def run(command,log):
    with open(log,'w') as output:
        subprocess.run([sys.executable,'scripts/train_dynamic_study.py',*command],stdout=output,stderr=subprocess.STDOUT,check=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-preparation',action='store_true');args=parser.parse_args()
    if args.wait_for_preparation:
        status('preparing')
        while not (ROOT/'scene_hashes.json').exists():time.sleep(10)
    if not (ROOT/'scene_hashes.json').exists():raise RuntimeError('Run prepare first')
    status('pilot',budget=50000)
    run(['pilot'],'logs/dynamic70-pilot.log')
    pilot=json.loads((ROOT/'pilot'/'training.json').read_text())
    if pilot['steps']!=50000:raise RuntimeError('Pilot step budget mismatch')
    status('formal_training',pilot=pilot,budget_per_condition=1000000)
    workers=[]
    try:
        for condition in ['clean','noise']:
            stream=open(f'logs/dynamic70-{condition}.log','w')
            process=subprocess.Popen([sys.executable,'scripts/train_dynamic_study.py','train','--condition',condition],stdout=stream,stderr=subprocess.STDOUT)
            workers.append((condition,process,stream))
        while any(p.poll() is None for _,p,_ in workers):
            for name,p,_ in workers:
                if p.poll() not in (None,0):raise RuntimeError(f'{name} training failed; see its log')
            time.sleep(10)
        for name,p,_ in workers:
            if p.returncode:raise RuntimeError(f'{name} failed')
            record=json.loads((ROOT/f'ppo_{name}'/'training.json').read_text())
            if record['steps']!=1000000:raise RuntimeError('Formal step budget mismatch')
    finally:
        for _,p,stream in workers:
            if p.poll() is None:p.terminate()
            stream.close()
    status('evaluation',episodes=600)
    run(['evaluate'],'logs/dynamic70-evaluation.log')
    summary=json.loads(Path('results/raw/dynamic70_main/summary.json').read_text())
    status('complete',summary=summary)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));raise
