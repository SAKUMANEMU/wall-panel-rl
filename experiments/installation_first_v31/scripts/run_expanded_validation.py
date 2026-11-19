"""Frozen v31 verification: fresh paired scenes and retrospective 2x2 diagnosis.

No optimizer, model selection or parameter tuning is performed. Completed jobs
are atomic and reusable after interruption; each resumed job uses frozen inputs.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import traceback
import numpy as np
import torch
import yaml
from run_adaptation import Status,save,digest
from expanded_validation_metrics import metrics,paired,trace_evidence,diagnostic_markdown
from wall_panel_rl.submission import load_agent,load_json,run_episode
from wall_panel_rl.rl.variable_scene import generate_variable_case
from wall_panel_rl.envs.known_route_motion import upgrade_case
from wall_panel_rl.rl.slow_dynamic_four import validate_placement
from wall_panel_rl.utils.config import ROOT

WORK={}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def copy_frozen(source,destination):
    if destination.exists():
        if digest(source)!=digest(destination):raise ValueError('Frozen input differs: '+str(destination))
    else:
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)


def prepare(args):
    output=args.output.resolve();inputs=output/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    source=args.source_run.resolve();original=load_json(source/'training/protocol.json')
    for name,sha in original['source_files'].items():
        if digest(ROOT/name)!=sha:raise ValueError('v31 source changed since training: '+name)
    copy_frozen(source/'training/models/selected_model.zip',inputs/'v31_model.zip')
    copy_frozen(ROOT/'models/current_ppo.zip',inputs/'v29_model.zip')
    if digest(inputs/'v29_model.zip')!=original['source_model_sha256']:
        raise ValueError('Restored baseline model is not the original v29 source')
    copy_frozen(ROOT/'models/source_ppo77.zip',inputs/'guide77.zip')
    copy_frozen(source/'training/selected_runtime.json',inputs/'installation_runtime.json')
    runtime=load_json(inputs/'installation_runtime.json')
    baseline=copy.deepcopy(runtime)
    baseline['controller']=load_json(ROOT/'configs/v29_baseline_controller.json')
    old=inputs/'old_runtime.json'
    if old.exists() and canonical_hash(load_json(old))!=canonical_hash(baseline):
        raise ValueError('Baseline runtime changed')
    save(old,baseline)
    spec=yaml.safe_load((source/'training_config.yaml').read_text())
    profile=spec['generation']
    copy_frozen(source/'training_config.yaml',inputs/'source_training_config.yaml')
    excluded=set()
    for run_name in ['known_route_speed_v28_seed20261116','construction_site_v29_training_seed20261117',
                     'close_weaving_v30_seed20261118','installation_first_v31_seed20261118']:
        for split in ['train','validation','test']:
            p=source.parent/run_name/'scenes'/f'{split}_cases.json'
            if p.exists():excluded.update(int(c['scene']['seed']) for c in load_json(p))
    cases=[];evidence=[]
    for seed in [4720000,4720100]:
        destination=inputs/f'original_{seed}.json'
        copy_frozen(source/f'training/recordings/independent_test_enhanced/{seed}.json',destination)
        recording=load_json(destination);cases.append(recording['case']);evidence.append(trace_evidence(recording))
        validate_placement(runtime['configuration'],recording['case'])
    save(output/'dual_timeout_analysis.json',{'original_evidence':evidence,'counterfactuals_complete':False})
    (output/'DUAL_TIMEOUT_ANALYSIS.md').write_text(diagnostic_markdown(evidence,[]))
    protocol={'training_performed':False,'parameters_changed':False,'model_selection_performed':False,
        'scenes_per_count':args.per_count,'counts':[2,3,4,5],'seed_base':args.seed_base,
        'generator_workers':args.generator_workers,'evaluation_workers':args.workers,
        'generation_profile':profile,'excluded_seed_ids':sorted(excluded),
        'geometry_only_acceptance':True,'evaluated_policy_outcomes_not_used_for_scene_acceptance':True,
        'paired_same_physical_cases':True,'original_diagnostic_seeds':[4720000,4720100],
        'original_diagnostic_cases_are_not_fresh_test_samples':True,
        'source_files':original['source_files'],
        'inputs_sha256':{p.name:digest(p) for p in inputs.iterdir() if p.is_file()},
        'analysis_scripts_sha256':{name:digest(ROOT/'scripts'/name) for name in
             ['run_expanded_validation.py','expanded_validation_metrics.py']}}
    prior=output/'protocol.json'
    if prior.exists() and load_json(prior)!=protocol:raise ValueError('Existing study protocol differs')
    save(prior,protocol)
    return protocol,runtime,cases,evidence


def initialize_generation(inputs,runtime,profile,excluded):
    torch.set_num_threads(1)
    WORK.update(guide=load_agent(Path(inputs)/'guide77.zip'),runtime=runtime,
                profile=profile,excluded=set(excluded))


def generate_one(job):
    count,index,first_seed=job;attempts=[];profile=WORK['profile']
    for offset in range(profile['max_seed_attempts_per_case']):
        seed=first_seed+offset
        if seed in WORK['excluded']:
            attempts.append({'seed':seed,'accepted':False,'reason':'previous study seed'});continue
        try:
            case=generate_variable_case(WORK['runtime']['configuration'],WORK['runtime']['settings'],
                                       profile,seed,count,WORK['guide'])
            case=upgrade_case(case,np.random.default_rng(seed+28000000),profile['speed_range_m_s'])
            case['split']='expanded_frozen_verification'
            attempts.append({'seed':seed,'accepted':True})
            return {'count':count,'index':index,'case':case,'attempts':attempts}
        except (ValueError,RuntimeError) as exc:
            attempts.append({'seed':seed,'accepted':False,'reason':str(exc)})
    return {'count':count,'index':index,'case':None,'attempts':attempts}


def scene_bank(output,protocol,runtime,status):
    jobs=[(count,index,protocol['seed_base']+count*10000+index*100)
          for count in protocol['counts'] for index in range(protocol['scenes_per_count'])]
    rows={};pending=[]
    for job in jobs:
        p=output/'scenes/generation'/f'{job[0]}_{job[1]}.json'
        if p.exists():
            row=load_json(p)
            if row['case'] is not None:rows[job[:2]]=row;continue
        pending.append(job)
    status.update('generate_expanded_scenes',generation_completed=len(rows),generation_planned=len(jobs))
    if pending:
        with ProcessPoolExecutor(protocol['generator_workers'],mp_context=mp.get_context('spawn'),
             initializer=initialize_generation,initargs=(str(output/'inputs'),runtime,
                      protocol['generation_profile'],protocol['excluded_seed_ids'])) as pool:
            futures=[pool.submit(generate_one,job) for job in pending]
            for future in as_completed(futures):
                row=future.result();save(output/'scenes/generation'/f"{row['count']}_{row['index']}.json",row)
                if row['case'] is None:raise RuntimeError('Geometry quota exhausted; attempts retained')
                rows[(row['count'],row['index'])]=row
                status.update('generate_expanded_scenes',generation_completed=len(rows),generation_planned=len(jobs),
                              latest_seed=row['case']['scene']['seed'])
    cases=[rows[job[:2]]['case'] for job in jobs]
    seeds=[c['scene']['seed'] for c in cases]
    if len(seeds)!=len(set(seeds)) or set(seeds)&set(protocol['excluded_seed_ids']):
        raise ValueError('Fresh scene bank overlaps or duplicates seeds')
    save(output/'scenes/cases.json',cases)
    speeds=[m['speed_m_s'] for c in cases for m in c['scene']['obstacle_motion']]
    dims=[box['dimensions'] for c in cases for box in c['scene']['obstacles']]
    initial_distances=[float(np.linalg.norm(np.asarray(c['scene']['nominal_motion']['panel_positions'][0])-
                          np.asarray(c['scene']['target_position']))) for c in cases]
    save(output/'scenes/coverage.json',{'scene_count':len(cases),'counts':{
        str(n):sum(len(c['scene']['obstacles'])==n for c in cases) for n in protocol['counts']},
        'speed_quantiles_m_s':np.quantile(speeds,[0,.25,.5,.75,1]).tolist(),
        'dimension_min_m':np.min(dims,axis=0).tolist(),'dimension_max_m':np.max(dims,axis=0).tolist(),
        'initial_target_distance_min_m':min(initial_distances),'initial_target_distance_max_m':max(initial_distances)})
    return cases


def initialize_evaluation(inputs):
    torch.set_num_threads(1);inputs=Path(inputs)
    WORK.update(models={name:load_agent(inputs/f'{name}_model.zip') for name in ['v31','v29']},
        runtimes={'installation':load_json(inputs/'installation_runtime.json'),'old':load_json(inputs/'old_runtime.json')},
        model_hashes={name:digest(inputs/f'{name}_model.zip') for name in ['v31','v29']})


def evaluate_one(job):
    arm,model_key,controller_key,case,output=job
    runtime=WORK['runtimes'][controller_key]
    result,recording=run_episode(WORK['models'][model_key],case,'enhanced',runtime)
    result.update(arm=arm,obstacle_count=len(case['scene']['obstacles']),model_key=model_key,
        controller_key=controller_key,case_sha256=canonical_hash(case),
        max_obstacle_speed_m_s=max(m['speed_m_s'] for m in case['scene']['obstacle_motion']),
        model_sha256=WORK['model_hashes'][model_key],runtime_sha256=canonical_hash(runtime))
    result['timeout_diagnosis']=recording['timeout_diagnosis']
    recording['result']=copy.deepcopy(result)
    output=Path(output)
    save(output/'recordings'/arm/f"{result['seed']}.json",recording)
    save(output/'rows'/arm/f"{result['seed']}.json",result)
    return result


def evaluate_jobs(pool,output,cases,arms,inputs,status,label):
    rows=[];jobs=[];runtime_hashes={'installation':canonical_hash(load_json(inputs/'installation_runtime.json')),
                                  'old':canonical_hash(load_json(inputs/'old_runtime.json'))}
    model_hashes={name:digest(inputs/f'{name}_model.zip') for name in ['v31','v29']}
    for case in cases:
        for arm,(model_key,controller_key) in arms.items():
            path=output/'rows'/arm/f"{case['scene']['seed']}.json"
            if path.exists():
                row=load_json(path)
                if (row['case_sha256']!=canonical_hash(case) or row['model_sha256']!=model_hashes[model_key] or
                    row['runtime_sha256']!=runtime_hashes[controller_key]):raise ValueError('Cached evaluation differs')
                rows.append(row)
            else:jobs.append((arm,model_key,controller_key,case,str(output)))
    planned=len(cases)*len(arms)
    def update():
        result={'complete':len(rows)==planned,'planned':planned,'completed':len(rows),
                'metrics':{arm:metrics([r for r in rows if r['arm']==arm]) for arm in arms},
                'rows':sorted(rows,key=lambda r:(r['seed'],r['arm']))}
        if label=='expanded':result['paired']=paired(rows)
        save(output/'results.json',result)
        status.update('evaluate_'+label,evaluation_completed=len(rows),evaluation_planned=planned,
                      metrics={arm:result['metrics'][arm]['all'] for arm in arms})
        return result
    update()
    for future in as_completed([pool.submit(evaluate_one,job) for job in jobs]):
        rows.append(future.result());update()
    return update()


def final_report(output,result):
    lines=['# Expanded frozen v31 verification','',
        'No model updates or model selection were performed. Both combinations ran the same',
        'fresh scenes. Geometry, motion distributions, installation tolerances and budget are unchanged.','',
        '| Combination | Episodes | Success | Collision | Timeout | Success rate | Wilson 95% interval |',
        '|---|---:|---:|---:|---:|---:|---|']
    for arm in ['v31_selected','v29_baseline']:
        m=result['metrics'][arm]['all'];ci=m['success_wilson95']
        lines.append(f"| {arm} | {m['episodes']} | {m['success']} | {m['collision']} | {m['timeout']} | {m['success_rate']:.1%} | {ci[0]:.1%}--{ci[1]:.1%} |")
    lines+=['','## Results by obstacle count','',
        '| Count | v31 success | v31 collision / timeout | v29 success | v29 collision / timeout |',
        '|---|---:|---:|---:|---:|']
    for count in ['2','3','4','5']:
        a=result['metrics']['v31_selected']['by_count'][count];b=result['metrics']['v29_baseline']['by_count'][count]
        lines.append(f"| {count} | {a['success']}/{a['episodes']} | {a['collision']} / {a['timeout']} | {b['success']}/{b['episodes']} | {b['collision']} / {b['timeout']} |")
    p=result['paired']
    lines+=['','## Paired outcomes','',f"Both succeeded: {p['both_success']}; v31 only: {p['v31_only_success']}; v29 only: {p['v29_only_success']}; both failed: {p['both_failed']}.",
        f"Paired success-rate difference: {p['paired_success_rate_difference']:+.1%}.",'',
        'The retrospective two-case diagnostic is separate and is not counted in these totals.',
        'These are results for PPO plus predictive control and recovery, not pure PPO.','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    locations=np.arange(4)
    for offset,arm,color in [(-.18,'v31_selected','#267c9f'),(.18,'v29_baseline','#b77c3c')]:
        groups=[result['metrics'][arm]['by_count'][str(n)] for n in [2,3,4,5]]
        rates=np.array([m['success_rate'] for m in groups])
        errors=np.array([[max(0.,m['success_rate']-m['success_wilson95'][0]) for m in groups],
                         [max(0.,m['success_wilson95'][1]-m['success_rate']) for m in groups]])
        axes[0].bar(locations+offset,rates,width=.34,yerr=errors,capsize=3,label=arm,color=color)
    axes[0].set_xticks(locations,['2','3','4','5'])
    axes[0].set(xlabel='Dynamic obstacle count',ylabel='Installation success rate',ylim=(0,1.05),title='Wilson 95% intervals')
    axes[0].legend(fontsize=8)
    labels=['v31_selected','v29_baseline'];bottom=np.zeros(2)
    for reason,color in [('success','#398d67'),('collision','#b95555'),('timeout','#cc9a36'),
                         ('joint_limit','#7f648f'),('planning_failure','#777777')]:
        heights=np.array([result['metrics'][arm]['all'][reason] for arm in labels])
        axes[1].bar(labels,heights,bottom=bottom,label=reason,color=color);bottom+=heights
    axes[1].set(ylabel='Episode count',title='All outcomes retained');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/'expanded_outcomes.png',dpi=160);plt.close(fig)


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/'study.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=Status(output)
        try:
            protocol,runtime,diagnostics,evidence=prepare(args)
            if args.prepare_only:
                status.update('prepared',status='prepared',training_performed=False);return
            cases=scene_bank(output,protocol,runtime,status)
            with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),
                 initializer=initialize_evaluation,initargs=(str(output/'inputs'),)) as pool:
                arms={f'{model}_model__{controller}':(model,controller)
                      for model in ['v31','v29'] for controller in ['installation','old']}
                diagnosed=evaluate_jobs(pool,output/'diagnostics',diagnostics,arms,output/'inputs',status,'diagnostics')
                save(output/'dual_timeout_analysis.json',{'original_evidence':evidence,
                    'counterfactuals_complete':True,'counterfactual_rows':diagnosed['rows']})
                (output/'DUAL_TIMEOUT_ANALYSIS.md').write_text(diagnostic_markdown(evidence,diagnosed['rows']))
                arms={'v31_selected':('v31','installation'),'v29_baseline':('v29','old')}
                result=evaluate_jobs(pool,output/'expanded',cases,arms,output/'inputs',status,'expanded')
            for name,sha in protocol['source_files'].items():
                if digest(ROOT/name)!=sha:raise ValueError('Source changed during verification')
            for name,sha in protocol['inputs_sha256'].items():
                if digest(output/'inputs'/name)!=sha:raise ValueError('Frozen input changed')
            for name,sha in protocol['analysis_scripts_sha256'].items():
                if digest(ROOT/'scripts'/name)!=sha:raise ValueError('Analysis script changed during verification')
            final_report(output,result)
            save(output/'summary.json',{'complete':True,'fresh_scenes':len(cases),'paired_episodes':len(result['rows']),
                 'training_performed':False,'model_weights_unchanged':True,'metrics':result['metrics'],'paired':result['paired']})
            status.update('complete',status='completed',training_performed=False)
        except BaseException as exc:
            status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-count',type=int,default=20)
    parser.add_argument('--seed-base',type=int,default=5300000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--generator-workers',type=int,default=2)
    parser.add_argument('--prepare-only',action='store_true',help='Freeze inputs and write retrospective evidence without simulations')
    args=parser.parse_args()
    if not 1<=args.per_count<=100 or min(args.workers,args.generator_workers)<1:
        parser.error('Use positive workers and at most 100 scenes per count')
    run(args)


if __name__=='__main__':main()
