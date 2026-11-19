"""Freeze train/validation/reserved-test dynamic banks without learning updates."""
import _bootstrap
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import copy
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.slow_dynamic_four import load_profile,dynamic_configuration,add_motion,validate_placement,fresh_dynamic_case
from wall_panel_rl.rl.four_ppo_continuation import hard_probabilities
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors,fresh_motion_aware_case
from run_targeted_bc import read,dump,sha

WORK={}


def initialize(spec):
    torch.set_num_threads(1);source=read(ROOT/spec['source_report']/'candidate_manifest.json')
    profile=load_profile(ROOT/spec['motion_profile'])
    config,settings=dynamic_configuration(source['configuration'],source['evaluation_settings'],profile)
    WORK.update(config=config,settings=settings,profile=profile,model=PPO.load(ROOT/spec['source_model'],device='cpu'))


def construct(item):
    kind,value=item
    try:
        if WORK['profile'].get('motion_aware_placement'):
            case=(fit_motion_anchors(WORK['config'],WORK['settings'],WORK['profile'],value) if kind=='reuse' else
                  fresh_motion_aware_case(WORK['config'],WORK['settings'],WORK['profile'],value,WORK['model']))
        elif kind=='reuse':case=add_motion(value,WORK['profile']);validate_placement(WORK['config'],case)
        else:case,_=fresh_dynamic_case(WORK['config'],WORK['settings'],WORK['profile'],value,model=WORK['model'])
        return {'case':case}
    except (RuntimeError,ValueError) as exc:
        return {'rejection':{'seed':value['scene']['seed'] if kind=='reuse' else value,
            'stage':'nominal_reference' if 'Cannot build a complete nominal route' in str(exc) else 'geometry','reason':str(exc)}}


def prepare(spec,config_file):
    data=ROOT/spec['data_root'];report=ROOT/spec['preparation_report']
    if data.exists() or report.exists():raise FileExistsError('Refusing preparation overwrite')
    data.mkdir(parents=True);report.mkdir(parents=True)
    source=read(ROOT/spec['source_report']/'candidate_manifest.json');model_hash=sha(ROOT/spec['source_model'])
    assert model_hash==source['model_sha256']
    profile=load_profile(ROOT/spec['motion_profile']);config,settings=dynamic_configuration(source['configuration'],source['evaluation_settings'],profile)
    old_split=read(ROOT/spec['source_data']/'split.json')
    excluded=set(old_split['validation']+old_split['test']+old_split['old_validation']+old_split['old_test_regression'])
    preflight=ROOT/'results/diagnostics/four_slow_dynamic_preflight_v1_seed61/cases.json'
    if preflight.exists():excluded.update(c['scene']['seed'] for c in read(preflight))
    for bank in spec.get('exclude_case_banks',[]):excluded.update(c['scene']['seed'] for c in read(ROOT/bank))
    excluded.update(read(ROOT/spec['source_report']/'protocol.json').get('excluded_seeds',[]))
    rejections=[];unassigned=[]
    def status(phase,**extra):
        state={'phase':phase,'updated_at':datetime.now().astimezone().isoformat(),'training_steps':0,**extra}
        dump(report/'status.json',state);print('DYNAMIC_PREPARATION',json.dumps(state),flush=True)
    try:
        with ProcessPoolExecutor(max_workers=spec['preparation_workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
            status('checking_original_train',requested=spec['train_count'])
            train=[];old=read(ROOT/spec.get('source_train_cases',spec['source_data']+'/train_cases.json'))
            for start in range(0,len(old),32):
                for result in pool.map(construct,[('reuse',c) for c in old[start:start+32]]):
                    if 'case' in result:train.append(result['case'])
                    else:rejections.append(dict(split='train',**result['rejection']))
                status('checking_original_train',accepted=len(train),processed=min(start+32,len(old)))
            reused=len(train)
            def fill(name,cases,target,start_seed):
                seed=start_seed;attempts=0
                while len(cases)<target:
                    size=min(24,max(4,target-len(cases)));seeds=list(range(seed,seed+size));seed+=size;attempts+=size
                    if attempts>target*20:raise RuntimeError('Too many construction rejections for '+name)
                    if set(seeds)&excluded:raise ValueError('Reserved seed collision')
                    status('generating_'+name,accepted=len(cases),requested=target,next_seed=seeds[0])
                    for result in pool.map(construct,[('new',s) for s in seeds]):
                        if 'case' in result:
                            if len(cases)<target:cases.append(result['case'])
                            else:unassigned.append(result['case']['scene']['seed'])
                        else:rejections.append(dict(split=name,**result['rejection']))
                return cases
            train=fill('train',train,spec['train_count'],spec['new_train_seed'])
            validation=fill('validation',[],spec['validation_count'],spec['validation_seed'])
            test=fill('reserved_test',[],spec['test_count'],spec['test_seed'])
        split={name:[c['scene']['seed'] for c in cases] for name,cases in (('train',train),('validation',validation),('test',test))}
        sets=[set(v) for v in split.values()]
        assert all(len(s)==len(split[k]) for s,k in zip(sets,split))
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
        assert not sets[0]&excluded
        source_protocol=read(ROOT/spec['source_report']/'protocol.json')
        hard_source=(read(ROOT/spec['source_sampling_file']) if spec.get('source_sampling_file') else source_protocol)
        hard=sorted(set(hard_source['hard_training_seeds'])&sets[0])
        probabilities=hard_probabilities(train,hard,spec['hard_training_fraction'])
        for name,cases in (('train',train),('validation',validation),('test',test)):dump(data/(name+'_cases.json'),cases)
        dump(data/'split.json',dict(split,previously_excluded=sorted(excluded)))
        dump(data/'sampling.json',{'hard_training_seeds':hard,'probabilities':probabilities.tolist()})
        dump(report/'construction_rejections.json',rejections)
        summary={'train':len(train),'validation':len(validation),'reserved_test':len(test),'reused_source_train':reused,
            'new_train':len(train)-reused,'rejections_by_split_and_stage':dict(Counter(r['split']+'/'+r['stage'] for r in rejections)),
            'unassigned_constructed_seeds':unassigned,'training_steps':0,'test_policy_evaluations':0,
            'source_unchanged':sha(ROOT/spec['source_model'])==model_hash,'splits_disjoint':True}
        inputs={str(p.relative_to(ROOT)):sha(p) for p in data.glob('*.json')}
        provenance=[spec.get('source_train_cases',spec['source_data']+'/train_cases.json')]
        provenance+=spec.get('exclude_case_banks',[])
        if spec.get('source_sampling_file'):provenance.append(spec['source_sampling_file'])
        inputs.update({f:sha(ROOT/f) for f in provenance})
        files=[str(Path(config_file).relative_to(ROOT)),spec['motion_profile'],'scripts/prepare_four_dynamic_training.py',
            'src/wall_panel_rl/rl/slow_dynamic_four.py','src/wall_panel_rl/rl/static_demo_scene.py','src/wall_panel_rl/rl/on_path_obstacles.py','src/wall_panel_rl/rl/motion_aware_placement.py']
        dump(report/'protocol.json',{'spec':spec,'configuration':config,'evaluation_settings':settings,'source_sha256':model_hash,
            'input_hashes':inputs,'code_hashes':{f:sha(ROOT/f) for f in files},'source_train_sha256':sha(ROOT/spec.get('source_train_cases',spec['source_data']+'/train_cases.json')),
            'rule':'Scene/reference construction only; no dynamic-policy success filtering; reserved test never used for training or selection'})
        for f in files:
            dest=report/'code_archive'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
        dump(report/'summary.json',summary);status('completed',summary=summary)
        (report/'README.md').write_text('# Dynamic training bank preparation\n\n'+json.dumps(summary,indent=2)+'\n\nTest actions have not been evaluated. Preparation may reject wall/endpoint/spacing geometry or an unavailable complete obstacle-free nominal reference; all such exclusions are recorded. No actual dynamic-policy result is used to choose scenes.\n')
    except Exception as exc:status('failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_dynamic_ppo.yaml');p.add_argument('--prepare',action='store_true')
    args=p.parse_args();spec=yaml.safe_load(args.config.read_text())
    print({'mode':'prepare' if args.prepare else 'check_only','train':spec['train_count'],'validation':spec['validation_count'],'reserved_test':spec['test_count'],'training_steps':0},flush=True)
    if args.prepare:prepare(spec,args.config.resolve())


if __name__=='__main__':main()
