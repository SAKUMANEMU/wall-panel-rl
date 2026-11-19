"""Generate larger-obstacle banks, then launch the authorized BC/PPO study."""
from pathlib import Path
import json, subprocess, sys, os, datetime
ROOT=Path('/home/johnsonchen/wall-panel-rl')
EXP=ROOT/'experiments/known_route_speed_v28'
OUT=ROOT/'results/diagnostics/known_route_speed_v28_seed20261116'
def status(phase, state='running', **extra):
    record={'status':state,'phase':phase,'pid':os.getpid(),'updated_at':datetime.datetime.now().astimezone().isoformat(),**extra}
    p=OUT/'pipeline_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(p)
try:
    for phase,script,arguments in [
        ('generate_scenes','generate_variable_cases.py',['--output',str(OUT/'scenes')]),
        ('train_and_evaluate','run_variable_training.py',['--scenes',str(OUT/'scenes'),'--output',str(OUT/'training')])]:
        status(phase)
        with (OUT/(phase+'.log')).open('ab',buffering=0) as log:
            command=[sys.executable,'-u',str(EXP/'scripts'/script),*arguments]
            child=subprocess.Popen(command,cwd=EXP,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status(phase,child_pid=child.pid)
            result=child.wait()
            if result:raise RuntimeError(phase+' exited with code '+str(result))
    status('complete','completed')
except BaseException as exc:
    status('failed','failed',error=repr(exc));raise
