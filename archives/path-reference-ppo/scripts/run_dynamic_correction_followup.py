"""Proceed to a full-strength 2k PPO pilot only after corrective BC improves."""
import _bootstrap
from datetime import datetime
from pathlib import Path
import json,subprocess,sys,time
import yaml
from wall_panel_rl.utils.config import ROOT
from run_targeted_bc import read,dump


def main():
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_corrections.yaml').read_text());report=ROOT/'results/diagnostics/four_dynamic_correction_study_v12_seed101'
    if report.exists():raise FileExistsError('Refusing followup overwrite')
    report.mkdir(parents=True);visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'dynamic_only':True,'motion':[.18,.12]}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra);dump(report/'status.json',state);print('CORRECTION_FOLLOWUP',json.dumps(state),flush=True)
    def child(command,log):
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,'-u',*command],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            status(state['phase'],child_pid=p.pid,log=str(log))
            while p.poll() is None:time.sleep(5)
            if p.returncode:raise RuntimeError(f'Child failed ({p.returncode}): {log}')
    try:
        status('waiting_for_corrective_bc')
        while True:
            p=ROOT/spec['bc_report']/'status.json'
            if p.exists():
                try:current=read(p)
                except json.JSONDecodeError:time.sleep(1);continue
                if current['status']=='failed':raise RuntimeError('Corrective BC failed: '+str(current))
                if current['status']=='completed':break
            time.sleep(5)
        bc=read(ROOT/spec['bc_report']/'summary.json')
        if bc['validation_improved']:
            status('full_strength_ppo',bc_selected=bc['selected'])
            child(['scripts/train_four_dynamic_ppo.py','--config','configs/four_dynamic_correction_ppo.yaml','--train'],Path('/home/johnsonchen/wall-panel-rl/logs/four_dynamic_correction_ppo_v12_seed101.log'))
        else:status('bc_not_improved_ppo_skipped',bc_selected=bc['selected'])
        status('auditing_and_exporting',ppo_started=bc['validation_improved'])
        child(['scripts/summarize_dynamic_correction_study.py'],Path('/home/johnsonchen/wall-panel-rl/logs/four_dynamic_correction_study_v12_seed101_audit.log'))
        status('complete',status='completed',child_pid=None,summary=read(report/'summary.json'))
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise

if __name__=='__main__':main()
