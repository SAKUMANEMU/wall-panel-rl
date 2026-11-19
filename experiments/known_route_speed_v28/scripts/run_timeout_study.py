"""Targeted timeout reduction with immutable PPO weights and case banks.

Known low32 and high16 are development data. Prespecified candidates are selected
without expanded-test outcomes; v23's frozen expanded bank is checked afterwards.
"""
import _bootstrap
import argparse,copy,json,multiprocessing as mp,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch
from run_adaptation import Status,save,digest,metrics
from wall_panel_rl.submission import load_agent,load_json,load_runtime,run_episode
from wall_panel_rl.utils.config import ROOT

PROJECT=ROOT.parents[1]
BASE=PROJECT/'results/diagnostics/enhanced_scope_v23_baseline_low32'
V21=PROJECT/'results/diagnostics/dynamic_adaptation_v21_seed20261112'
SCOPE=PROJECT/'results/diagnostics/enhanced_scope_v23_seed20261113'
CACHE={}


def initialize():
    torch.set_num_threads(1);CACHE.update(model=load_agent(),runtime=load_runtime())


def runtime_for(variant):
    rt=copy.deepcopy(CACHE['runtime'])
    if variant!='original':
        rt['controller']['selected']['horizon']=12
        rt['controller']['recovery_strategy']='timeout_v24'
        rt['controller']['recovery'].update(minimum_step=40,window=16,cooldown=12,
            maximum_rescues=6,horizon=60,commit_steps=8,
            allow_temporary_detour=variant=='detour')
    if variant not in ('original','strict','detour'):raise ValueError('Unknown variant')
    return rt


def evaluate_one(job):
    variant,group,case,record_root=job
    row,recording=run_episode(CACHE['model'],case,'enhanced',runtime_for(variant))
    row.update(source_bank=group,timeout_strategy=variant,
        waiting_fraction=row['zero_command_steps']/max(row['steps'],1),
        stationary_fraction=row['stationary_steps']/max(row['steps'],1))
    if record_root:save(Path(record_root)/f'{row["seed"]}.json',recording)
    return row


def aggregate(rows):
    return {group:metrics([r for r in rows if group=='all' or r['source_bank']==group])
            for group in ('all','high','low') if group=='all' or any(r['source_bank']==group for r in rows)}


def evaluate(pool,variant,bank,out,status,label,record=False):
    futures={pool.submit(evaluate_one,(variant,group,case,str(out/'recordings'/label) if record else None)):case['scene']['seed'] for group,case in bank}
    rows=[]
    while futures:
        ready,_=wait(futures,timeout=15,return_when=FIRST_COMPLETED)
        for future in ready:futures.pop(future);rows.append(future.result())
        status.update('evaluate_'+label,timeout_strategy=variant,completed_episodes=len(rows),planned_episodes=len(bank))
        if rows:save(out/'evaluations'/f'{label}.json',{'variant':variant,'complete':not futures,
            'metrics':aggregate(rows),'rows':sorted(rows,key=lambda r:r['seed'])})
    return {'metrics':aggregate(rows),'rows':rows}


def safe_score(candidate,baseline):
    c=candidate['metrics'];b=baseline['metrics']
    eligible=(c['all']['joint_limit']<=b['all']['joint_limit'] and
              c['all']['collision']<=b['all']['collision'] and c['high']['success']>=b['high']['success'])
    return (int(eligible),c['all']['success'],-c['all']['collision'],-c['all']['timeout'],
            -c['all']['mean_waiting_fraction'])


def baseline_from_cache():
    low=load_json(BASE/'results.json')
    high=load_json(V21/'evaluations/ppo_24576.json')
    assert len(low['rows'])==32 and len(high['enhanced']['rows'])==16
    assert load_json(BASE/'protocol.json')['model_sha256']==digest(ROOT/'models/current_ppo.zip')
    assert high['model_sha256']==digest(ROOT/'models/current_ppo.zip')
    rows=[]
    for group,data in [('low',low['rows']),('high',high['enhanced']['rows'])]:
        for r in data:
            rows.append(dict(r,source_bank=group,timeout_strategy='original',
                waiting_fraction=r['zero_command_steps']/max(r['steps'],1),
                stationary_fraction=r['stationary_steps']/max(r['steps'],1)))
    return {'metrics':aggregate(rows),'rows':rows,'historical_measured_baseline':True}


def run(args):
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True);status=Status(out);initialize()
    baseline=baseline_from_cache();save(out/'evaluations/baseline.json',baseline)
    high=load_json(ROOT/'data/selection_cases.json');low=load_json(ROOT/'data/low_diagnostic_cases.json')
    bank=[('high',c) for c in high]+[('low',c) for c in low]
    hashes={str(p):digest(p) for p in [ROOT/'models/current_ppo.zip',ROOT/'configs/runtime.json',
        ROOT/'data/selection_cases.json',ROOT/'data/low_diagnostic_cases.json',BASE/'results.json',V21/'evaluations/ppo_24576.json']}
    save(out/'protocol.json',{'objective':'Reduce enhanced-controller timeouts while preserving collision/limit counts and high16 successes',
        'baseline':baseline['metrics'],'development_cases':48,'policies_frozen':True,'training_performed':False,
        'candidates':['strict','detour'],'new_recovery_start':40,'recovery_horizon':60,'max_rescues':6,
        'future_waypoint_offsets':[0,3,6,12],'persistent_joint_offset_rad':.28,'persistent_target_steps':16,
        'temporary_task_cost_increase_bound':.12,'minimum_clearance_gain_m':.015,
        'episode_limit_unchanged':240,'collision_rule_unchanged':True,'global_rrt_fallback':False,
        'input_hashes':hashes,'script_sha256':digest(Path(__file__)),
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src').rglob('*.py')},
        'expanded_test_path':str(SCOPE/'expanded_test_cases.json'),'expanded_test_not_used_for_selection':True})
    chosen='original';best=baseline;history=[]
    try:
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            for variant in ('strict','detour'):
                candidate=evaluate(pool,variant,bank,out,status,variant,True)
                history.append({'variant':variant,'metrics':candidate['metrics']})
                if safe_score(candidate,baseline)>safe_score(best,baseline):chosen=variant;best=candidate
                save(out/'comparison.json',history)
                save(out/'selection.json',{'variant':chosen,'metrics':best['metrics'],'frozen':False})
            save(out/'frozen_selection.json',{'variant':chosen,'metrics':best['metrics'],
                'selected_before_expanded_test':True})
            save(out/'selected_runtime.json',runtime_for(chosen))
            (out/'models').mkdir();__import__('shutil').copy2(ROOT/'models/current_ppo.zip',out/'models/selected_model.zip')
            old=evaluate(pool,chosen,[('regression64',c) for c in load_json(ROOT/'data/validation_cases.json')],out,status,'old64_regression')
            save(out/'development_summary.json',{'selected':chosen,'baseline':baseline['metrics'],
                'selected_development':best['metrics'],'old64_regression':old['metrics'],'expanded_test_pending':True})
            while not (SCOPE/'expanded_test_cases.json').exists():
                parent=load_json(SCOPE/'status.json') if (SCOPE/'status.json').exists() else {}
                if parent.get('status')=='failed':
                    raise RuntimeError('v23 scene generation failed; development results retained, expanded test not run')
                status.update('waiting_for_frozen_expanded_test',scope_status=parent.get('phase'))
                time.sleep(15)
            test=load_json(SCOPE/'expanded_test_cases.json')
            assert not {c['scene']['seed'] for _,c in bank}&{c['scene']['seed'] for c in test}
            save(out/'expanded_test_input.json',{'path':str(SCOPE/'expanded_test_cases.json'),
                'sha256':digest(SCOPE/'expanded_test_cases.json'),'count':len(test)})
            test_bank=[('expanded_test',c) for c in test]
            original=evaluate(pool,'original',test_bank,out,status,'expanded_test_original')
            selected=evaluate(pool,chosen,test_bank,out,status,'expanded_test_selected',True)
            for p,sha in hashes.items():assert digest(p)==sha,'Frozen input modified'
            save(out/'summary.json',{'selected':chosen,'baseline':baseline['metrics'],
                'selected_development':best['metrics'],'old64_regression':old['metrics'],
                'expanded_test_baseline':original['metrics'],'expanded_test_selected':selected['metrics'],
                'training_performed':False,'model_weights_unchanged':True})
            status.update('complete',status='completed')
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.workers<1:p.error('Positive worker count required')
    run(a)
if __name__=='__main__':main()
