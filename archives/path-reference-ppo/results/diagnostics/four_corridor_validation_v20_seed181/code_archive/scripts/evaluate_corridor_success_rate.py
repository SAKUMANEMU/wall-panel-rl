"""Measure unchanged controllers on64 new current larger-stagger scenes.

The preliminary physical episode defines placement only. Freeze64 geometrically
valid final scenes before evaluating any final-scene controller. Never select
scenes by final success or by the measured weave events.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.slow_dynamic_four import load_profile, dynamic_configuration, SlowDynamicFourEnv, planning_snapshot, assert_planning_paused
from wall_panel_rl.rl.vertical_corridor import fresh_vertical_corridor_case
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
from wall_panel_rl.visualization.episode_replay import capture_frame, recording_payload
from evaluate_vertical_layout import read, dump, sha, prior_seeds, metrics, timeout_category, collision_objects
from probe_vertical_corridor import passage_measurements

WORK = {}


def initialize(protocol, report):
    torch.set_num_threads(1)
    model = PPO.load(Path(report)/'frozen_model.zip', device='cpu')
    assert model.observation_space.shape == (77,) and model.action_space.shape == (7,)
    WORK.update(protocol=protocol, report=Path(report), model=model,
                weights=copy.deepcopy(model.policy.state_dict()))


def unchanged_model():
    assert all(torch.equal(v,WORK['model'].policy.state_dict()[k]) for k,v in WORK['weights'].items())


def heartbeat(phase, **extra):
    dump(WORK['report']/'workers'/f'{os.getpid()}.json',
         {'phase':phase, 'updated_at':datetime.now().astimezone().isoformat(), **extra})


def generate(seed):
    s=WORK['protocol']
    heartbeat('constructing_preliminary_route_and_final_geometry',seed=seed)
    begin=time.perf_counter()
    try:
        case=fresh_vertical_corridor_case(s['configuration'],s['settings'],s['profile'],seed,
                                          WORK['model'],s['controller'])
        return {'seed':seed,'case':case,'generation_wall_s':time.perf_counter()-begin}
    except (RuntimeError,ValueError) as exc:
        return {'seed':seed,'rejection':str(exc),'generation_wall_s':time.perf_counter()-begin}
    finally:
        unchanged_model()
        heartbeat('idle')


def episode(job):
    mode,case=job
    s=WORK['protocol'];seed=int(case['scene']['seed'])
    env=SlowDynamicFourEnv(copy.deepcopy(s['configuration']),[],copy.deepcopy(s['settings']))
    started=time.perf_counter();control_seconds=0.;reward=0.;zero=0;stationary=0;longest=0;streak=0
    trace=[]
    heartbeat('evaluation',mode=mode,seed=seed,step=0)
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        reference=env.cursor.path.copy();frames=[capture_frame(env.env)]
        assert env.env.config['simulation']['max_steps']==240 and case['strength']==[.18,.12]
        assert len(env.env.obstacle_ids)==4 and len(case['scene']['obstacle_motion'])==4
        heights=case['scene']['final_height_stagger']
        assert heights['anchor_height_span_m']>=.35 and heights['minimum_adjacent_delta_m']>=.14
        assert case['scene']['demo_height_offsets_m']==[0.,-.05,0.,0.]
        c=s['controller']
        controller=ProgressRecoveryController(WORK['model'],env,c['selected']['horizon'],c['gains'],c['recovery']) if mode=='ppo_progress' else None
        while True:
            before=planning_snapshot(env);q=env.env.joint_state()[0].copy();start=time.perf_counter()
            action=controller.action(obs) if controller else WORK['model'].predict(obs,deterministic=True)[0]
            elapsed=time.perf_counter()-start;control_seconds+=elapsed
            assert_planning_paused(env,before)
            obs,r,done,truncated,info=env.step(action);reward+=r
            frames.append(capture_frame(env.env))
            is_zero=bool(np.max(np.abs(action))<1e-8)
            is_still=bool(np.max(np.abs(env.env.joint_state()[0]-q))<1e-4)
            zero+=is_zero;stationary+=is_still;streak=streak+1 if is_still else 0;longest=max(longest,streak)
            trace.append({'step':int(env.env.steps),'time_s':env.env.sim_time,
                          'position_m':float(info['position_error']),
                          'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                          'zero_command':is_zero,'stationary':is_still,
                          'reference_index':int(env.cursor.index),'control_wall_s':elapsed})
            if env.env.steps%20==0:heartbeat('evaluation',mode=mode,seed=seed,step=int(env.env.steps))
            if done or truncated:break
        unchanged_model()
        assert np.array_equal(reference,env.cursor.path)
        row={'mode':mode,'seed':seed,'reason':info['termination_reason'],'steps':int(env.env.steps),
             'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
             'reward':float(reward),'zero_command_steps':int(zero),'stationary_steps':int(stationary),
             'longest_stationary_run':longest,'reference_index':int(env.cursor.index),'reference_last':len(reference)-1,
             'initial_planning_s':0.,'control_wall_s':control_seconds,'total_wall_s':time.perf_counter()-started,
             'physics_elapsed_s':env.env.sim_time,'weights_changed':False,'reference_preserved':True,
             'uses_global_rrt':False,'local_rescues':controller.rescues if controller else 0,
             'local_recovery_actions':controller.recovery_actions if controller else 0,
             'predictive_interventions':controller.interventions if controller else 0,
             'discarded_prediction_steps':controller.prediction_steps if controller else 0,
             'recovery_events':controller.recovery_events if controller else [],
             'gain_counts':controller.gain_counts if controller else {},
             'collision_objects':collision_objects(env.env) if info['termination_reason']=='collision' else [],
             'trace':trace}
        target=s['configuration']['target']
        row['timeout_category']=timeout_category(row,target['position_tolerance'],target['orientation_tolerance_deg'])
        payload=recording_payload(s['configuration'],case,row,frames)
        measurements=passage_measurements(payload)
        row['passage_count']=measurements['passage_count']
        row['ordered_passages']=measurements['ordered_alternating_passages']
        row['successful_with_verified_weaving']=row['reason']=='success' and measurements['passage_count']>=3
        dump(WORK['report']/'recordings'/f'{mode}_{seed}.json',payload)
        return row
    finally:
        env.close()
        heartbeat('idle')


def summary_metrics(rows):
    result=metrics(rows)
    result['successful_with_verified_weaving']=sum(r['successful_with_verified_weaving'] for r in rows)
    result['success_and_weaving_rate']=result['successful_with_verified_weaving']/len(rows) if rows else 0.
    return result


def run(spec,spec_path):
    report=ROOT/spec['report_root'];data=ROOT/spec['data_root']
    if report.exists() or data.exists():raise FileExistsError('Refusing frozen evaluation overwrite')
    profile=load_profile(ROOT/spec['motion_profile'])
    assert profile['placement_route_source']=='preliminary_physical_episode' and profile['enforce_final_stagger']
    manifest=read(ROOT/spec['source_report']/'candidate_manifest.json')
    config,settings=dynamic_configuration(manifest['configuration'],manifest['evaluation_settings'],profile)
    controller=read(ROOT/spec['controller_config']);used=prior_seeds()
    # Explicitly exclude all previous selected-development/demo seeds, too.
    for name in ['outcomes.json','selected_case.json','fresh_generated_case.json']:
        path=ROOT/'results/diagnostics/four_vertical_corridor_v19'/name
        if path.exists():
            values=read(path)
            for value in values:
                used.add(int(value['scene']['seed'] if 'scene' in value else value['seed']))
    report.mkdir(parents=True);data.mkdir(parents=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    files=list((ROOT/'src').rglob('*.py'))+[spec_path,ROOT/'scripts/evaluate_corridor_success_rate.py',
        ROOT/'scripts/evaluate_vertical_layout.py',ROOT/'scripts/probe_vertical_corridor.py',
        ROOT/spec['motion_profile'],ROOT/profile['preliminary_motion_profile'],ROOT/spec['controller_config'],
        ROOT/spec['source_report']/'candidate_manifest.json']
    hashes={str(path.relative_to(ROOT)):sha(path) for path in files}
    model_sha=sha(ROOT/spec['source_model']);shutil.copy2(ROOT/spec['source_model'],report/'frozen_model.zip')
    for name in hashes:
        dest=report/'code_archive'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    protocol={'spec':spec,'configuration':config,'settings':settings,'profile':profile,'controller':controller,
        'source_model_sha256':model_sha,'code_hashes':hashes,'optimizer_updates':0,'final_test_episodes':0,
        'selection':'First64 new-seed final geometries passing the exact current demo generator; preliminary controller outcome is not a success filter. No final-controller or weave-based selection.',
        'rate_denominator':'64 valid final scenes. Construction rejection counts/yield reported separately; no success-filtered replacement of final episodes.',
        'modes':'Current animation controller PPO+prediction+recovery, and pure PPO with the same projection/reference environment wrapper. No RRT/mixed evaluation in this study.',
        'weaving':'Secondary diagnostic: successful installation plus at least3 ordered near-box passages with alternating whole-panel above/below clearance>=1cm, XY distance<=24cm and alternating actual height changes>=7cm.',
        'budget':'Unchanged240 real control steps; known moving obstacles and paused simulation during prediction. Virtual trials are not training transitions.',
        'geometry':'14–22cm base offsets plus O2 downward5cm; final anchor span>=35cm and minimum adjacent height difference>=14cm after adjustment;18cm motion amplitude and0.12m/s speed bound.'}
    dump(report/'protocol.json',protocol)
    state={'status':'running','optimizer_updates':0,'final_test_episodes':0}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('CORRIDOR_RATE',json.dumps(state),flush=True)
    cases=[];attempts=[];results={};seed=spec['first_scene_seed']
    try:
        with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=mp.get_context('spawn'),
                                 initializer=initialize,initargs=(protocol,str(report))) as pool:
            while len(cases)<spec['validation_count']:
                batch=[]
                while len(batch)<min(16,max(4,spec['validation_count']-len(cases))):
                    if seed not in used:batch.append(seed)
                    seed+=1
                if len(attempts)+len(batch)>spec['maximum_generation_attempts']:raise RuntimeError('Generation attempt budget exhausted')
                status('generating',accepted=len(cases),target=64,attempted=len(attempts),batch_completed=0,batch_total=len(batch))
                futures={pool.submit(generate,n):n for n in batch};received={}
                for future in as_completed(futures):
                    value=future.result();received[value['seed']]=value
                    if len(received)%4==0 or len(received)==len(batch):
                        status('generating',batch_completed=len(received),batch_valid_received=sum('case' in x for x in received.values()))
                for n in batch:
                    value=received[n]
                    if 'case' in value:
                        selected=len(cases)<spec['validation_count']
                        if selected:cases.append(value['case'])
                        attempts.append({'seed':n,'accepted':selected,'geometry_valid':True,'generation_wall_s':value['generation_wall_s'],
                                         'preliminary_reason':value['case']['scene']['preliminary_episode']['reason']})
                    else:attempts.append({'seed':n,'accepted':False,'geometry_valid':False,
                                           'reason':value['rejection'],'generation_wall_s':value['generation_wall_s']})
                dump(report/'construction_attempts.json',attempts)
                status('generating',accepted=len(cases),attempted=len(attempts))
            ids=[c['scene']['seed'] for c in cases]
            assert len(ids)==len(set(ids))==64 and not set(ids)&used
            dump(data/'validation_cases.json',cases)
            dump(data/'split.json',{'validation':ids,'excluded_prior_ids':sorted(used),'train':[],'test':[]})
            bank_sha=sha(data/'validation_cases.json')
            dump(report/'frozen_inputs.json',{'validation_bank_sha256':bank_sha,'model_sha256':model_sha,
                'final_layout_evaluations_before_freeze':0,'accepted':64,'attempts':len(attempts),
                'geometry_rejected':sum(not x['geometry_valid'] for x in attempts),
                'unused_valid_candidates':sum(x['geometry_valid'] and not x['accepted'] for x in attempts)})
            for mode in spec['execution_modes']:
                rows=[];status('evaluating',mode=mode,completed=0,total=64,outcomes=None)
                futures=[pool.submit(episode,(mode,c)) for c in cases]
                for future in as_completed(futures):
                    rows.append(future.result())
                    ordered=sorted(rows,key=lambda r:ids.index(r['seed']))
                    dump(report/f'{mode}.json',{'metrics':summary_metrics(ordered),'rows':ordered})
                    if len(rows)%4==0 or len(rows)==64:
                        status('evaluating',completed=len(rows),outcomes=summary_metrics(rows))
                results[mode]=sorted(rows,key=lambda r:ids.index(r['seed']))
        assert sha(data/'validation_cases.json')==bank_sha
        assert sha(ROOT/spec['source_model'])==sha(report/'frozen_model.zip')==model_sha
        for name,digest in hashes.items():assert sha(ROOT/name)==digest,f'Frozen runtime changed: {name}'
        summary={'metrics':{mode:summary_metrics(rows) for mode,rows in results.items()},
                 'model_weights_changed':False,'optimizer_updates':0,'final_test_episodes':0,
                 'valid_scenes':64,'final_scene_rollouts':64*len(results),'runtime_hashes_verified':True,
                 'construction_attempts':len(attempts),'construction_rejected':sum(not x['geometry_valid'] for x in attempts)}
        dump(report/'summary.json',summary)
        status('complete',status='completed',process_exited=True,summary=summary)
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/four_corridor_success_rate.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args()
    spec=yaml.safe_load(args.config.read_text())
    assert spec['validation_count']==64 and set(spec['execution_modes'])=={'ppo','ppo_progress'}
    print({'mode':'run' if args.run else 'check_only','training':False,'final_test':False},flush=True)
    if args.run:run(spec,args.config.resolve())


if __name__=='__main__':main()
