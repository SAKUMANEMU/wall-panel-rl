"""Frozen-model enhanced-controller comparison on a wider, fixed scene family.

No PPO/BC optimizer runs. Guide weights and the workspace are frozen before
scene construction; final-test outcomes never select controller parameters.
"""
import _bootstrap
import argparse,copy,json,multiprocessing as mp,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch,yaml
from run_adaptation import Status,save,digest,jobs,metrics
from wall_panel_rl.submission import load_json,load_agent,load_runtime,run_episode
from wall_panel_rl.rl.expanded_scope import expanded_case
from wall_panel_rl.utils.config import ROOT

CACHE={}

def initialize():
    torch.set_num_threads(1)
    CACHE.update(runtime=load_runtime(),profile=yaml.safe_load((ROOT/'configs/scene.yaml').read_text()),
                 scope=yaml.safe_load((ROOT/'configs/expanded_scope.yaml').read_text()),models={})


def policy(name):
    if name not in CACHE['models']:
        files={'v21':'current_ppo.zip','v22':'standalone_v22.zip',
               'high':'placement_guide_high.zip','low':'placement_guide_low.zip'}
        CACHE['models'][name]=load_agent(ROOT/'models'/files[name])
    return CACHE['models'][name]


def runtime_for(variant):
    rt=copy.deepcopy(CACHE['runtime']);c=rt['controller']
    c['recovery_strategy']='original'
    if variant in ('early','diverse'):
        c['recovery'].update(minimum_step=60,window=16,cooldown=16,maximum_rescues=5,commit_steps=6)
    if variant=='diverse':
        c['recovery_strategy']='diverse_v23';c['selected']['horizon']=18
        c['recovery'].update(horizon=48)
    if variant not in ('original','early','diverse'):raise ValueError('Unknown controller variant')
    return rt


def generate(job):
    tag,seed=job;rt=CACHE['runtime']
    try:
        case=expanded_case(rt['configuration'],rt['settings'],CACHE['profile'],CACHE['scope'],seed,
                           policy(tag),rt['controller'],tag)
        return {'guide':tag,'seed':seed,'geometry_valid':True,'case':case}
    except (RuntimeError,ValueError) as exc:
        return {'guide':tag,'seed':seed,'geometry_valid':False,'reason':str(exc)}


def evaluate_one(job):
    model,variant,case,label,record_root=job
    row,recording=run_episode(policy(model),case,'enhanced',runtime_for(variant))
    row.update(policy=model,recovery_variant=variant,evaluation_group=label,
        waiting_fraction=row['zero_command_steps']/max(1,row['steps']),
        stationary_fraction=row['stationary_steps']/max(1,row['steps']))
    if record_root:
        save(Path(record_root)/f'{model}_{variant}_{row["seed"]}.json',recording)
    return row


def evaluate(pool,model,variant,bank,out,status,label,record=False):
    futures={pool.submit(evaluate_one,(model,variant,c,label,str(out/'recordings'/label) if record else None)):c['scene']['seed'] for c in bank}
    rows=[]
    while futures:
        ready,_=wait(futures,timeout=15,return_when=FIRST_COMPLETED)
        for future in ready:
            futures.pop(future);rows.append(future.result())
        status.update('evaluate_'+label,policy=model,recovery_variant=variant,
            completed_episodes=len(rows),planned_episodes=len(bank))
        if rows:
            save(out/'evaluations'/f'{label}.json',{'policy':model,'recovery_variant':variant,
                'complete':not futures,'episodes_planned':len(bank),'metrics':metrics(rows),'rows':sorted(rows,key=lambda x:x['seed'])})
    return {'metrics':metrics(rows),'rows':rows}


def rank(result,baseline):
    m=result['metrics'];b=baseline['metrics']
    safe=m['joint_limit']<=b['joint_limit'] and m['collision']<=b['collision']
    return (int(safe),m['success'],-m['joint_limit'],-m['collision'],
            -m['mean_waiting_fraction'],-float(np.mean([r['steps'] for r in result['rows']])))


def run(args):
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Choose a fresh run output directory')
    out.mkdir(parents=True);status=Status(out);initialize()
    paths=[ROOT/'models'/name for name in ['current_ppo.zip','standalone_v22.zip','placement_guide_high.zip','placement_guide_low.zip']]
    paths += [ROOT/'configs'/name for name in ['runtime.json','scene.yaml','preliminary_scene.yaml','expanded_scope.yaml']]
    prefill_manifest=ROOT/'checks_scope/generation_prefill_manifest.json'
    if prefill_manifest.exists():
        manifest=load_json(prefill_manifest)
        for name,sha in manifest.items():assert digest(ROOT/name)==sha,'Preflight input changed'
        paths += [prefill_manifest,ROOT/'checks_scope/final_generation_attempts.json',ROOT/'checks_scope/final_accepted_cases.json']
    hashes={str(p):digest(p) for p in paths}
    save(out/'protocol.json',{'scope':CACHE['scope'],'input_hashes':hashes,'weights_frozen':True,
        'training_performed':False,'development_per_guide':args.development_per_guide,
        'test_per_guide':args.test_per_guide,'workers':args.workers,
        'candidate_policies':['v21','v22'],'candidate_controllers':['original','early','diverse'],
        'selection':'Within baseline collision/limit counts, maximize development success, then fewer failures/waits/steps. Exact ties keep earlier candidate.',
        'final_test_used_for_selection':False,'script_sha256':digest(Path(__file__)),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src').rglob('*.py')}})
    try:
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            banks={'high':[],'low':[]};quota=args.development_per_guide+args.test_per_guide
            next_seed={'high':args.start_seed,'low':args.start_seed+1};attempts=[]
            if prefill_manifest.exists():
                attempts=load_json(ROOT/'checks_scope/final_generation_attempts.json')
                accepted=load_json(ROOT/'checks_scope/final_accepted_cases.json')
                for tag,case in accepted.items():
                    assert case['scene']['expanded_scope_v23']['guide']==tag
                    banks[tag].append(case)
                for tag in ('high','low'):
                    prior=[r['seed'] for r in attempts if r['guide']==tag]
                    if prior:next_seed[tag]=max(prior)+2
            while min(len(banks[k]) for k in banks)<quota and len(attempts)<args.max_attempts:
                requested=[]
                for tag in ('high','low'):
                    if len(banks[tag])>=quota:continue
                    count=min(4,quota-len(banks[tag]))
                    for _ in range(count):requested.append((tag,next_seed[tag]));next_seed[tag]+=2
                status.update('generate_expanded_scenes',high_valid=min(len(banks['high']),quota),
                    low_valid=min(len(banks['low']),quota),per_guide_target=quota,attempts=len(attempts))
                batch=jobs(pool,generate,requested,status,'generate_expanded_scenes')
                for row in batch:
                    case=row.pop('case',None);attempts.append(row)
                    if case is not None:banks[row['guide']].append(case)
                save(out/'generation_attempts.json',attempts);save(out/'all_valid_cases.json',banks)
            if any(len(banks[k])<quota for k in banks):raise RuntimeError('Expanded-scene construction budget exhausted; partial data retained')
            development=[];test=[]
            for tag in ('high','low'):
                development+=banks[tag][:args.development_per_guide]
                test+=banks[tag][args.development_per_guide:quota]
            rng=np.random.default_rng(args.seed);rng.shuffle(development);rng.shuffle(test)
            save(out/'expanded_development_cases.json',development);save(out/'expanded_test_cases.json',test)
            high_known=load_json(ROOT/'data/selection_cases.json')[:8]
            low_known=load_json(ROOT/'data/low_diagnostic_cases.json')[:8]
            dev=high_known+low_known+development
            assert not {c['scene']['seed'] for c in dev}&{c['scene']['seed'] for c in test}
            save(out/'controller_development_cases.json',dev)
            save(out/'split.json',{'development':[c['scene']['seed'] for c in dev],
                'expanded_test':[c['scene']['seed'] for c in test],
                'known_cases_are_development_only':True,'frozen_before_controller_comparison':True})
            baseline=evaluate(pool,'v21','original',dev,out,status,'development_v21_original')
            best=baseline;chosen=('v21','original');history=[]
            for model in ('v21','v22'):
                for variant in ('original','early','diverse'):
                    result=baseline if (model,variant)==('v21','original') else evaluate(pool,model,variant,dev,out,status,f'development_{model}_{variant}')
                    history.append({'policy':model,'recovery_variant':variant,'metrics':result['metrics']})
                    if rank(result,baseline)>rank(best,baseline):best=result;chosen=(model,variant)
                    save(out/'comparison.json',history)
                    save(out/'selection.json',{'policy':chosen[0],'recovery_variant':chosen[1],
                        'development_metrics':best['metrics'],'selection_frozen':False})
            frozen={'policy':chosen[0],'recovery_variant':chosen[1],'development_metrics':best['metrics'],'selection_frozen':True}
            save(out/'frozen_selection.json',frozen)
            rt=runtime_for(chosen[1]);save(out/'selected_runtime.json',rt)
            chosen_file=ROOT/'models'/('current_ppo.zip' if chosen[0]=='v21' else 'standalone_v22.zip')
            (out/'models').mkdir();__import__('shutil').copy2(chosen_file,out/'models/selected_model.zip')
            # All outcomes in each frozen test/regression bank remain counted.
            baseline_test=evaluate(pool,'v21','original',test,out,status,'expanded_test_baseline')
            selected_test=evaluate(pool,*chosen,test,out,status,'expanded_test_selected',True)
            low=evaluate(pool,*chosen,load_json(ROOT/'data/low_diagnostic_cases.json'),out,status,'low32_diagnostic',True)
            old=evaluate(pool,*chosen,load_json(ROOT/'data/validation_cases.json'),out,status,'old64_regression')
            for p,sha in hashes.items():assert digest(p)==sha,'Frozen input modified'
            save(out/'summary.json',{'selected_policy':chosen[0],'selected_controller':chosen[1],
                'baseline_development':baseline['metrics'],'selected_development':best['metrics'],
                'expanded_test_baseline':baseline_test['metrics'],'expanded_test_selected':selected_test['metrics'],
                'low32_diagnostic':low['metrics'],'old64_regression':old['metrics'],
                'training_performed':False,'model_weights_unchanged':True,'scenario_rules_frozen':True})
            status.update('complete',status='completed')
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--development-per-guide',type=int,default=8)
    p.add_argument('--test-per-guide',type=int,default=16);p.add_argument('--start-seed',type=int,default=1231000)
    p.add_argument('--seed',type=int,default=20261113);p.add_argument('--max-attempts',type=int,default=1600)
    a=p.parse_args()
    if min(a.workers,a.development_per_guide,a.test_per_guide)<1 or a.start_seed%2:p.error('Invalid study budget')
    run(a)
if __name__=='__main__':main()
