"""Train-only dynamic corrections from source-policy pre-collision prefixes."""
import _bootstrap
import argparse,copy,json,multiprocessing as mp,shutil
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_corrections import trace_policy,corrective_episode
from wall_panel_rl.rl.four_ppo_continuation import hard_probabilities
from run_targeted_bc import read,dump,sha

WORK={}

def initialize(spec):
    torch.set_num_threads(1);p=read(ROOT/spec['preparation_report']/'protocol.json')
    WORK.update(spec=spec,config=p['configuration'],settings=p['evaluation_settings'],model=PPO.load(ROOT/spec['source_model'],device='cpu'))

def trace(case):return trace_policy(WORK['config'],WORK['settings'],case,WORK['model'])

def correction(job):
    case,saved,handoff,seed=job
    return corrective_episode(WORK['config'],WORK['settings'],case,saved,handoff,seed)

def run(spec,config_file):
    report=ROOT/spec['report_root'];data=ROOT/spec['output_root']
    if report.exists() or data.exists():raise FileExistsError('Refusing corrective-data overwrite')
    p=read(ROOT/spec['preparation_report']/'protocol.json')
    for path,h in p['input_hashes'].items():assert sha(ROOT/path)==h
    source=sha(ROOT/spec['source_model']);assert source==read(ROOT/spec['source_report']/'candidate_manifest.json')['model_sha256']
    split=read(ROOT/spec['data_root']/'split.json');train=set(split['train']);held=set(split['validation']+split['test']+split['previously_excluded']);assert not train&held
    cases=read(ROOT/spec['data_root']/'train_cases.json');by_seed={c['scene']['seed']:c for c in cases}
    report.mkdir(parents=True);data.mkdir(parents=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    state={'status':'running','optimizer_updates':0,'static_episodes':0}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra);dump(report/'status.json',state);print('CORRECTIONS',json.dumps(state),flush=True)
    files=[str(config_file.relative_to(ROOT)),'scripts/collect_dynamic_corrections.py','src/wall_panel_rl/rl/dynamic_corrections.py',
           'src/wall_panel_rl/rl/dynamic_four_training.py','src/wall_panel_rl/rl/slow_dynamic_four.py','src/wall_panel_rl/rl/conditional_recovery.py',
           'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/baselines/classical.py','src/wall_panel_rl/planning/timed_rrt.py','src/wall_panel_rl/envs/wall_panel_env.py']
    hashes={f:sha(ROOT/f) for f in files};dump(report/'protocol.json',{'spec':spec,'configuration':p['configuration'],'evaluation_settings':p['evaluation_settings'],
        'source_sha256':source,'input_hashes':p['input_hashes'],'code_hashes':hashes,'train_seeds':sorted(train),'excluded_seeds':sorted(held),
        'rule':'Exact source PPO prefix from original reset, timed RRT suffix at original time; pre/post checks and full prefix-plus-suffix independent replay; suffix-only labels',
        'heldout_policy_evaluations':0,'optimizer_updates':0})
    for f in files:
        dest=report/'code_archive'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    try:
        with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
            traces=[];status('tracing_source_train',processed=0,total=len(cases))
            for start in range(0,len(cases),24):
                traces.extend(pool.map(trace,cases[start:start+24]));status('tracing_source_train',processed=len(traces),total=len(cases),outcomes=dict(Counter(t['reason'] for t in traces)))
            anchors=np.stack([s['observation'] for t in traces if t['reason']=='success' for s in t['states'][::spec['anchor_stride']]])
            failed=[t for t in traces if t['reason']!='success'];rng=np.random.default_rng(spec['seed']);rng.shuffle(failed)
            # First sweep changes layouts before trying the second takeover lead.
            jobs=[];seen=set()
            for lead in spec['handoff_leads']:
                for t in failed:
                    handoff=max(0,len(t['states'])-lead);identity=(t['seed'],handoff)
                    if identity in seen:continue
                    seen.add(identity);jobs.append((by_seed[t['seed']],t,handoff,int(rng.integers(1,2**31-1))))
            accepted=[];attempts=[];status('collecting_corrective_suffixes',accepted=0,target=spec['correction_episodes'],attempts=0)
            for start in range(0,min(len(jobs),spec['maximum_attempts']),8):
                batch=jobs[start:min(start+8,spec['maximum_attempts'])]
                for result in pool.map(correction,batch):
                    attempts.append(result['row'])
                    if result['row']['accepted_for_imitation']:accepted.append(result)
                dump(report/'attempts.json',attempts);status('collecting_corrective_suffixes',accepted=len(accepted),target=spec['correction_episodes'],attempts=len(attempts))
                if len(accepted)>=spec['correction_episodes']:break
        if len(accepted)<16:raise RuntimeError('Insufficient verified corrective suffixes')
        assert all(r['row']['seed'] in train and r['row']['seed'] not in held and r['case']['task']=='dynamic' for r in accepted)
        labels=[pair for episode in accepted for pair in episode['pairs'][:spec['label_head_steps']]]
        torch.save({'episodes':accepted,'success_anchor_observations':anchors,'training_pairs':labels,'dynamic_only':True,'train_only':True},data/'corrections.pt')
        failures=[t['seed'] for t in failed];prob=hard_probabilities(cases,failures,.25)
        dump(data/'sampling.json',{'hard_training_seeds':sorted(failures),'probabilities':prob.tolist(),'hard_fraction':.25})
        summary={'verified_corrections':len(accepted),'corrected_layouts':len({e['row']['seed'] for e in accepted}),
            'head_training_pairs':len(labels),'full_verified_suffix_pairs':sum(len(e['pairs']) for e in accepted),'success_anchor_states':len(anchors),
            'nonzero_time_corrections':sum(e['row']['handoff']>0 for e in accepted),'source_outcomes':dict(Counter(t['reason'] for t in traces)),
            'attempts':len(attempts),'attempt_outcomes':dict(Counter(r['reason'] for r in attempts)),
            'train_only':True,'dynamic_only':True,'reference_and_clock_preserved':True,'optimizer_updates':0,'source_unchanged':sha(ROOT/spec['source_model'])==source}
        for f,h in hashes.items():assert sha(ROOT/f)==h,'Code changed during correction collection'
        dump(report/'summary.json',summary);dump(report/'data_hashes.json',{str(f.relative_to(ROOT)):sha(f) for f in data.glob('*')})
        status('complete',status='completed',summary=summary)
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_dynamic_corrections.yaml');p.add_argument('--run',action='store_true');a=p.parse_args();spec=yaml.safe_load(a.config.read_text())
    print({'mode':'run' if a.run else 'check_only','dynamic_only':True,'target_corrections':spec['correction_episodes']},flush=True)
    if a.run:run(spec,a.config.resolve())

if __name__=='__main__':main()
