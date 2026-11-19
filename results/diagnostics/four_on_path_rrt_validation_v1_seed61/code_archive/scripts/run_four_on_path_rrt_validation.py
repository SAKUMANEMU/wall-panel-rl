"""Actual RRT-Connect vs initialization execution on the fixed99 on-path scenes.

One primary RRT seed per scene. Two diagnostic retries on primary failures are
reported separately. No scene selection, planner tuning, fallback or training.
"""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from wall_panel_rl.rl.rrt_validation import run_episode,counts,feasibility


def collision_summary(rows):
    collided=[r for r in rows if r['reason']=='collision']
    dominant=Counter();slots=Counter();body=Counter()
    for r in collided:
        c=r['dominant_contact']
        dominant['unknown' if c is None else c['kind']+'/'+c['moving_body']]+=1
        if c and c['obstacle_slot'] is not None:slots[str(c['obstacle_slot'])]+=1
        for kind in {c['moving_body'] for c in r['contacts']}:body[kind]+=1
    return {'collision_episodes':len(collided),'dominant_contact_categories':dict(dominant),
        'dominant_obstacle_slots':dict(slots),'episodes_with_contact_by_body':dict(body),
        'body_counts_can_overlap':True,
        'first_collision_steps':[r['first_collision_step'] for r in collided]}


def run(study):
    root=Path(study['report_root']);source=Path(study['source_report'])
    if root.exists():raise FileExistsError('Refusing baseline overwrite')
    manifest=read(source/'manifest.json');config=copy.deepcopy(manifest['configuration']);settings=copy.deepcopy(manifest['evaluation_settings'])
    cases=read(study['cases']);expected=read(source/'on_path.json')
    if len(cases)!=99 or len({c['scene']['seed'] for c in cases})!=99:raise ValueError('Expected the fixed99 distinct scenes')
    if any(len(c['scene']['obstacles'])!=4 or c['scene']['obstacle_motion'] for c in cases):raise ValueError('Four static boxes required')
    model_hash=sha(source/'selected_model.zip');case_hash=sha(study['cases'])
    if model_hash!=manifest['model_sha256']:raise ValueError('Wrong initialized model')
    root.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'training_steps':0,'scenes':99}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(root/'status.json',state);print('RRT_FOUR',json.dumps(state),flush=True)
    code=['scripts/run_four_on_path_rrt_validation.py','configs/four_on_path_rrt_validation.yaml',
        'src/wall_panel_rl/rl/rrt_validation.py','src/wall_panel_rl/baselines/classical.py',
        'src/wall_panel_rl/planning/rrt_connect.py','src/wall_panel_rl/envs/collision.py']
    dump(root/'protocol.json',{'study':study,'configuration':config,'evaluation_settings':settings,
        'source_model_sha256':model_hash,'scene_bank_sha256':case_hash,'cases':99,
        'code_hashes':{p:sha(p) for p in code},'primary_seed_rule':'scene seed +100000; no retries replace primary results',
        'retry_rule':'each primary failure gets two additional seeds; report separately',
        'execution_rule':'unchanged ClassicalController, unchanged joint actuator and 240-step horizon',
        'post_plan_recheck':'diagnostic only, AFTER actual execution; never filters the baseline',
        'independent_test':False,'training_steps':0})
    try:
        model=PPO.load(source/'selected_model.zip',device='cpu');ppo=[];primary=[];retries=[]
        cached={r['seed']:r for r in expected['rows'] if r['suite']=='stage_target'}
        status('replay_initialization')
        for i,case in enumerate(cases):
            row,trace=run_episode(model,config,settings,case,'ppo')
            old=cached[row['seed']]
            if row['reason']!=old['reason'] or row['steps']!=old['steps']:
                raise AssertionError('Initialization no longer reproduces cached execution')
            ppo.append(row);dump(root/'traces/ppo'/f'{row["seed"]}.json',trace)
            with (root/'ppo_episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            if (i+1)%10==0 or i+1==len(cases):status('replay_initialization',completed=i+1,metrics=counts(ppo))
        if counts(ppo)!=expected['metrics']['stage_target']:raise AssertionError('Initialization result counts changed')
        status('rrt_primary',completed=0)
        for i,case in enumerate(cases):
            seed=int(case['scene']['seed'])+study['primary_planner_seed_offset']
            row,trace=run_episode(None,config,settings,case,'rrt',seed);row['attempt']=0;primary.append(row)
            dump(root/'traces/rrt_primary'/f'{row["seed"]}.json',trace)
            with (root/'rrt_primary_episodes.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            status('rrt_primary',completed=i+1,metrics=counts(primary),plans_found=sum(r['planning_success'] for r in primary))
        case_by_seed={c['scene']['seed']:c for c in cases}
        failed=[r for r in primary if r['reason']!='success']
        status('rrt_diagnostic_retries',failed_primary_scenes=len(failed),completed_retry_trials=0)
        for row in failed:
            for attempt,offset in enumerate(study['retry_planner_seed_offsets'],1):
                retried,trace=run_episode(None,config,settings,case_by_seed[row['seed']],'rrt',row['seed']+offset)
                retried['attempt']=attempt;retries.append(retried)
                dump(root/'traces/rrt_retries'/f'{row["seed"]}_{attempt}.json',trace)
                with (root/'rrt_retry_episodes.jsonl').open('a') as f:f.write(json.dumps(retried)+'\n')
                status('rrt_diagnostic_retries',completed_retry_trials=len(retries),retry_metrics=counts(retries))
        if sha(source/'selected_model.zip')!=model_hash or sha(study['cases'])!=case_hash:raise AssertionError('Source changed')
        proof=feasibility(primary,retries)
        paired=Counter(f'{a["reason"]} -> {b["reason"]}' for a,b in zip(ppo,primary))
        dump(root/'summary.json',{'initialization':counts(ppo),'rrt_primary':counts(primary),
            'rrt_primary_plans_found':sum(r['planning_success'] for r in primary),
            'rrt_retries':counts(retries),'feasibility':proof,'paired_initialization_to_rrt':dict(paired),
            'collision_details':{'initialization':collision_summary(ppo),'rrt_primary':collision_summary(primary),'rrt_retries':collision_summary(retries)},
            'source_model_unchanged':True,'scene_bank_unchanged':True,'training_steps':0})
        for path in code:
            dest=root/'code_archive'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        status('complete',status='completed',rrt_primary=counts(primary),feasible_by_execution=proof['scenes_with_any_successful_execution'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_on_path_rrt_validation.yaml')
    p.add_argument('--run',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text())
    if study['training_steps']!=0 or len(study['retry_planner_seed_offsets'])!=2:raise ValueError('Bounded read-only validation')
    print({'mode':'run' if args.run else 'check_only','cases':99,'maximum_rrt_trials':297,'training_steps':0},flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
