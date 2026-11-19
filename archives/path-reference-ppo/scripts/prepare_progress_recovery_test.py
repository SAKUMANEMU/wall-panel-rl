"""Freeze a new test bank; no controller outcomes used for scene construction."""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json,multiprocessing as mp,copy
from datetime import datetime
import torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.motion_aware_placement import fresh_motion_aware_case
from run_targeted_bc import read,dump,sha
WORK={}
def initialize(spec):
    torch.set_num_threads(1);p=read(ROOT/spec['preparation_report']/'protocol.json')
    WORK.update(config=p['configuration'],settings=p['evaluation_settings'],profile=yaml.safe_load((ROOT/spec['motion_profile']).read_text()),model=PPO.load(ROOT/spec['source_model'],device='cpu'))
def generate(seed):
    try:return {'case':fresh_motion_aware_case(WORK['config'],WORK['settings'],WORK['profile'],seed,WORK['model'])}
    except (RuntimeError,ValueError) as e:return {'rejection':{'seed':seed,'reason':str(e)}}
def main():
    spec=yaml.safe_load((ROOT/'configs/four_progress_recovery.yaml').read_text());root=ROOT/spec['fresh_test_root'];report=ROOT/spec['preparation_root']
    if root.exists() or report.exists():raise FileExistsError('Refusing fresh-test overwrite')
    root.mkdir(parents=True);report.mkdir(parents=True);visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    config_hash=sha(ROOT/'configs/four_progress_recovery.yaml');profile_hash=sha(ROOT/spec['motion_profile']);model_hash=sha(ROOT/spec['source_model'])
    cases=[];rejects=[];seed=spec['fresh_test_seed']
    def status(phase,**extra):
        s={'phase':phase,'updated_at':datetime.now().astimezone().isoformat(),'test_policy_evaluations':0,**extra};dump(report/'status.json',s);print('FRESH_TEST',json.dumps(s),flush=True)
    try:
        with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
            while len(cases)<spec['fresh_test_count']:
                status('generating',accepted=len(cases),target=spec['fresh_test_count'],next_seed=seed)
                ids=list(range(seed,seed+min(24,max(4,spec['fresh_test_count']-len(cases)))));seed+=len(ids)
                if seed>spec['fresh_test_seed']+3000:raise RuntimeError('Too many geometric construction failures')
                for result in pool.map(generate,ids):
                    if 'case' in result and len(cases)<spec['fresh_test_count']:cases.append(result['case'])
                    elif 'rejection' in result:rejects.append(result['rejection'])
        prior=read(ROOT/spec['data_root']/'split.json');used=set(prior['train']+prior['validation']+prior['test']+prior['previously_excluded']);ids=[c['scene']['seed'] for c in cases]
        assert len(set(ids))==len(ids) and not set(ids)&used
        assert all(c['task']=='dynamic' and c['strength']==[.18,.12] for c in cases)
        dump(root/'test_cases.json',cases);dump(root/'split.json',{'test':ids,'excluded_prior_ids':sorted(used)})
        dump(report/'rejections.json',rejects)
        assert sha(ROOT/'configs/four_progress_recovery.yaml')==config_hash and sha(ROOT/spec['motion_profile'])==profile_hash and sha(ROOT/spec['source_model'])==model_hash
        dump(report/'protocol.json',{'spec':spec,'source_model_sha256':model_hash,'configuration_sha256':config_hash,'motion_profile_sha256':profile_hash,'test_cases_sha256':sha(root/'test_cases.json'),'selection_rule':'Geometry and complete obstacle-free nominal reference only; no actual controller success filtering','test_policy_evaluations':0,'cases':len(cases),'rejections':len(rejects)})
        status('completed',accepted=len(cases),rejections=len(rejects))
    except Exception as e:status('failed',error=repr(e));raise
if __name__=='__main__':main()
