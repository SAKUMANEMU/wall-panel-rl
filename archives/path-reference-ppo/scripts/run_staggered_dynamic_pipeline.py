"""Run the authorized dynamic-only warmup/PPO pilot after preparation completes."""
import _bootstrap
from datetime import datetime
import json,subprocess,sys,time
from pathlib import Path
import yaml
from wall_panel_rl.utils.config import ROOT
from run_targeted_bc import read,dump


def main():
    spec=yaml.safe_load((ROOT/'configs/four_staggered_dynamic_training.yaml').read_text())
    report=ROOT/'results/diagnostics/four_staggered_dynamic_pipeline_v11_seed91'
    if report.exists():raise FileExistsError('Refusing pipeline overwrite')
    report.mkdir(parents=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'dynamic_probability':1.,'ppo_budget':spec['steps']}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('STAGGERED_PIPELINE',json.dumps(state),flush=True)
    logs=Path('/home/johnsonchen/wall-panel-rl/logs')
    try:
        status('waiting_for_preparation')
        while True:
            path=ROOT/spec['preparation_report']/'status.json'
            if path.exists():
                try:p=read(path)
                except json.JSONDecodeError:time.sleep(1);continue
                if p['phase']=='failed':raise RuntimeError('Preparation failed: '+str(p))
                if p['phase']=='completed':break
            time.sleep(5)
        visible_prep=visible.parent/Path(spec['preparation_report']).name
        if not visible_prep.exists():visible_prep.symlink_to(ROOT/spec['preparation_report'],target_is_directory=True)
        stages=[('dynamic_diagnosis_and_bc',['scripts/run_staggered_dynamic_warmup.py','--run'],spec['warmup_report']),
                ('dynamic_ppo',['scripts/train_four_dynamic_ppo.py','--config','configs/four_staggered_dynamic_training.yaml','--train'],spec['report_root']),
                ('audit_and_curves',['scripts/summarize_staggered_dynamic.py'],spec['report_root'])]
        for phase,command,child_report in stages:
            log=logs/('four_staggered_v11_seed91_'+phase+'.log')
            with log.open('x') as stream:
                process=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                status(phase,child_pid=process.pid,child_report=child_report,log=str(log))
                while process.poll() is None:time.sleep(5)
                if process.returncode:raise RuntimeError(f'{phase} failed with exit code {process.returncode}; see {log}')
        summary=read(ROOT/spec['report_root']/'summary.json')
        status('complete',status='completed',child_pid=None,selected=summary['selected'],independent_test=summary['independent_test'])
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise

if __name__=='__main__':main()
