import json, os, subprocess, time
from pathlib import Path
from datetime import datetime
run=Path(__file__).resolve().parent
work=Path("/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo")
python="/home/johnsonchen/wall-panel-rl/.conda-env/bin/python"
phases=[("prepare","--prepare"),("teacher","--collect-teacher"),("bc","--pretrain"),("ppo","--train")]
state={"stage":"one_static","started_at":datetime.now().astimezone().isoformat(),"runner_pid":os.getpid(),"ppo_budget":10000,"automatic_stage_advance":False,"completed_phases":[]}
def save():
    tmp=run/"status.json.tmp"
    tmp.write_text(json.dumps(state,indent=2))
    tmp.replace(run/"status.json")
try:
    env=dict(os.environ);env.pop("PYTHONPATH",None);env["PYTHONUNBUFFERED"]="1"
    for phase,flag in phases:
        state.update(status="running",phase=phase,phase_started_at=datetime.now().astimezone().isoformat())
        with (run/(phase+".log")).open("w") as log:
            child=subprocess.Popen([python,"-u","scripts/staged_avoidance.py","--stage","one_static",flag],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid;save()
            rc=child.wait()
        if rc:
            state.update(status="failed",return_code=rc,finished_at=datetime.now().astimezone().isoformat());save();raise SystemExit(rc)
        state["completed_phases"].append(phase);save()
    state.update(status="completed",finished_at=datetime.now().astimezone().isoformat());state.pop("child_pid",None);save()
except Exception as e:
    state.update(status="failed",error=repr(e),finished_at=datetime.now().astimezone().isoformat());save();raise
