import json, os, subprocess
from pathlib import Path
from datetime import datetime
run=Path(__file__).resolve().parent
work=Path("/home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo")
python="/home/johnsonchen/wall-panel-rl/.conda-env/bin/python"
state={"stage":"one_static","version":"v3","started_at":datetime.now().astimezone().isoformat(),"runner_pid":os.getpid(),"ppo_budget":10000,"automatic_stage_advance":False,"completed_phases":[]}
def save():
    tmp=run/"status.json.tmp"
    tmp.write_text(json.dumps(state,indent=2))
    tmp.replace(run/"status.json")
def finish(status,**kwargs):
    state.update(status=status,finished_at=datetime.now().astimezone().isoformat(),**kwargs)
    state.pop("child_pid",None);save()
try:
    env=dict(os.environ);env.pop("PYTHONPATH",None);env["PYTHONUNBUFFERED"]="1"
    for phase,flag in [("prepare","--prepare"),("teacher","--collect-teacher"),("bc","--pretrain"),("ppo","--train")]:
        state.update(status="running",phase=phase,phase_started_at=datetime.now().astimezone().isoformat())
        with (run/(phase+".log")).open("w") as log:
            child=subprocess.Popen([python,"-u","scripts/staged_avoidance.py","--stage","one_static",flag],cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
            state["child_pid"]=child.pid;save();rc=child.wait()
        if rc:
            finish("failed",return_code=rc);raise SystemExit(rc)
        state["completed_phases"].append(phase);save()
        if phase=="bc":
            gate=work/"models/staged_avoidance_v3_seed61/one_static/bc/closed_loop_gate.json"
            result=json.loads(gate.read_text());state["bc_gate_passed"]=result["passed"]
            state["bc_gate_path"]=str(gate);save()
            if not result["passed"]:
                finish("stopped_bc_gate",reason="BC closed-loop validation failed; PPO was not launched")
                raise SystemExit(0)
    assessment=work/"models/staged_avoidance_v3_seed61/one_static/ppo/assessment.json"
    result=json.loads(assessment.read_text())
    finish("completed",stage_gate_passed=result["passed"],assessment_path=str(assessment))
except Exception as error:
    finish("failed",error=repr(error));raise
