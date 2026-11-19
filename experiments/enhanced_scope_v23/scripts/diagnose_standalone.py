"""Read-only rollout/geometry diagnosis. No fitting, model selection or training.

Counterfactual controls are reported separately from the frozen test result.
The test bank is now an inspected diagnostic set and cannot be reused as a fresh
final test after future design changes informed by this analysis.
"""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
import copy,json,multiprocessing as mp,time,traceback
from pathlib import Path
import numpy as np
import pybullet as p
import torch
from run_adaptation import Status,save,digest,jobs,initialize_worker
from wall_panel_rl.submission import load_json,load_agent,load_runtime,CaseBankEnv,run_episode
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv
from wall_panel_rl.rl.waypoint_tracking import reference_observation
from wall_panel_rl.utils.config import ROOT

RUN=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/standalone_ppo_v22_seed20261112')
OUT=RUN.parent/'standalone_ppo_v22_diagnosis_20261113'
CACHE={}


def get_model(which):
    if which not in CACHE:
        CACHE[which]=load_agent(ROOT/'models/current_ppo.zip' if which=='source' else RUN/'models/selected_model.zip')
    return CACHE[which]


def collision_pairs(e,use_contacts):
    result=[]
    def inspect(a,b,label_a,label_b,allowed=lambda contact:True):
        contacts=p.getContactPoints(a,b,physicsClientId=e.client) if use_contacts else p.getClosestPoints(a,b,0,physicsClientId=e.client)
        for c in contacts:
            if float(c[8])<=0 and allowed(c):
                result.append({'pair':label_a+'-'+label_b,'link_a':int(c[3]),'link_b':int(c[4]),
                    'distance_m':float(c[8]),'point_a':list(c[5]),'point_b':list(c[6])})
    for body,label in [(x,f'O{i+1}') for i,x in enumerate(e.obstacle_ids)]+[(e.wall,'wall')]:
        inspect(e.robot,body,'arm',label);inspect(e.panel,body,'panel',label)
    inspect(e.robot,e.robot,'arm','arm',lambda c:abs(c[3]-c[4])>1)
    inspect(e.robot,e.panel,'arm','panel',lambda c:c[3]!=e.ee)
    return result


def run_rollout(job):
    group,case,which,variant=job;model=get_model(which);rt=load_runtime()
    if variant=='rrt':
        row,_=run_episode(model,case,'rrt',rt)
        row.update(group=group,variant=variant,model=which)
        return row
    env=SlowDynamicFourEnv(rt['configuration'],[],rt['settings'])
    start=time.perf_counter();event=[];trace=[];projected=0;saturated=[]
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        e=env.env;initial_clearance=float(info.get('obstacle_clearance_capped_m',.08))
        initial_collision=bool(info['collision'])
        if variant=='no_obstacles':
            for body in e.obstacle_ids:p.removeBody(body,physicsClientId=e.client)
            e.obstacle_ids=[];e.obstacles=[];e.obstacle_motion=[]
            p.performCollisionDetection(physicsClientId=e.client)
            obs=reference_observation(e.observation(),env.ref,e.lower,e.upper)
        elif variant=='static':
            e.config['obstacles']['freeze_motion']=True
            obs=reference_observation(e.observation(),env.ref,e.lower,e.upper)
        original=e.collision.is_collision
        def hooked(use_contacts=False,margin=None):
            hit=original(use_contacts=use_contacts,margin=margin)
            if hit and not event:
                event.append({'step':e.steps+1,'time_s':float(e.sim_time),
                    'uses_contact_cache':bool(use_contacts),'pairs':collision_pairs(e,use_contacts)})
            return hit
        e.collision.is_collision=hooked
        while True:
            action=np.zeros(7,np.float32) if variant=='zero' else model.predict(obs,deterministic=True)[0]
            saturated.append(float(np.mean(np.abs(action)>=.99)))
            previous_reference=env.cursor.index
            obs,reward,done,truncated,info=env.step(action)
            projected+=int(np.max(np.abs(np.asarray(info['executed_action'])-action))>1e-6)
            trace.append({'step':e.steps,'position_m':float(info['position_error']),
                'orientation_deg':float(np.rad2deg(info['orientation_error'])),
                'clearance_m':None if info['obstacle_clearance_capped_m'] is None else float(info['obstacle_clearance_capped_m']),
                'reference_index':env.cursor.index,'previous_reference_index':previous_reference,
                'max_abs_action':float(np.max(np.abs(action)))})
            if done or truncated:break
        return {'group':group,'seed':case['scene']['seed'],'model':which,'variant':variant,
            'reason':info['termination_reason'],'steps':e.steps,'physics_elapsed_s':float(e.sim_time),
            'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
            'initial_collision':initial_collision,'initial_clearance_capped_m':initial_clearance,
            'first_collision':event[0] if event else None,'projected_steps':projected,
            'mean_saturated_action_fraction':float(np.mean(saturated)),
            'wall_seconds':time.perf_counter()-start,'last_steps':trace[-5:],
            'reference_final_index':env.cursor.index,'reference_length':len(env.cursor.path)}
    finally:env.close()


def check_interface(case):
    rt=load_runtime();a=SlowDynamicFourEnv(rt['configuration'],[],rt['settings'])
    b=CaseBankEnv([case])
    try:
        obs,info=a.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        other,_=b.reset(seed=20261113,options={'case_index':0})
        assert np.array_equal(obs,other),'Training/evaluation reset observations differ'
        e=a.env;q,dq=e.joint_state();pose=e.panel_pose()
        manual=np.r_[2*(q-e.lower)/(e.upper-e.lower)-1,dq/2,
                     np.asarray(e.observed_target[0])-pose[0],
                     __import__('wall_panel_rl.envs.geometry',fromlist=['rotation_error']).rotation_error(e.observed_target[1],pose[1])/np.pi,
                     np.concatenate([np.r_[center-np.asarray(pose[0]),dimensions,e.obstacle_velocity(i),1.] for i,(center,dimensions) in enumerate(e.obstacles)]),
                     np.zeros(10),2*(a.ref-e.lower)/(e.upper-e.lower)-1].astype(np.float32)
        assert np.allclose(obs,manual,atol=1e-6),'Observation semantic layout differs'
        assert abs(np.dot(case['scene']['target_orientation'],e.target_pose[1]))>1-1e-6
        models={}
        for which in ('source','selected'):
            model=get_model(which);x=torch.as_tensor(obs[None])
            with torch.no_grad():
                action,_,logp=model.policy(x,deterministic=True)
                _,checked_logp,_=model.policy.evaluate_actions(x,action)
                mean=model.policy.get_distribution(x).distribution.loc
            predicted=model.predict(obs,deterministic=True)[0]
            assert np.allclose(predicted,action.numpy()[0],atol=1e-6)
            assert np.allclose(predicted,mean.numpy()[0],atol=1e-6)
            assert torch.allclose(logp,checked_logp,atol=1e-6)
            assert np.isfinite(predicted).all() and np.max(np.abs(predicted))<=1.000001
            models[which]={'forward_predict_mean_match':True,'likelihood_match':True}
        action=get_model('selected').predict(obs,deterministic=True)[0]
        first=a.step(action);second=b.step(action)
        assert np.allclose(first[0],second[0],atol=1e-6)
        assert abs(first[1]-second[1])<1e-6 and first[2:4]==second[2:4]
        return {'seed':case['scene']['seed'],'obs_shape':list(obs.shape),'finite':bool(np.isfinite(obs).all()),
                'train_eval_reset_and_step_match':True,'manual_observation_layout_match':True,'models':models}
    finally:a.close();b.close()


def scene_features(cases,label):
    rows=[]
    for case in cases:
        s=case['scene'];motion=s['obstacle_motion'];anchors=np.array([m['anchor'] for m in motion]);q=np.array(s['reference_joint_path'])
        points=np.array(s['nominal_motion']['panel_positions']);witness=s['unadjusted_motion_anchor_placement']['obstacles']
        row={'dataset':label,'seed':s['seed'],'mean_anchor_z_m':float(anchors[:,2].mean()),
            'min_anchor_z_m':float(anchors[:,2].min()),'max_anchor_z_m':float(anchors[:,2].max()),
            'mean_anchor_y_m':float(anchors[:,1].mean()),'height_span_m':float(np.ptp(anchors[:,2])),
            'preliminary_reason':s['preliminary_episode']['reason'],'preliminary_steps':s['preliminary_episode']['steps'],
            'reference_joint_arc_rad':float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()),
            'placement_route_arc_m':float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            'placement_route_min_z_m':float(points[:,2].min()),'placement_route_max_z_m':float(points[:,2].max()),
            'arm_witness_boxes':sum(bool(v['intersects_arm']) for v in witness),
            'panel_witness_boxes':sum(bool(v['intersects_panel']) for v in witness),
            'motion_amplitude_bound':float(max(np.linalg.norm(m['extent']) for m in motion)),
            'motion_speed_bound':float(max(np.linalg.norm(np.array(m['extent'])*m['omega']) for m in motion)),
            'dimensions':[b['dimensions'] for b in s['obstacles']],
            'initial_q':s['initial_q'],'anchor_centers':anchors.tolist()}
        rows.append(row)
    return rows


def main():
    torch.set_num_threads(1)
    if OUT.exists():raise FileExistsError('Use existing diagnostic outputs rather than overwrite')
    OUT.mkdir();status=Status(OUT)
    paths={'train':ROOT/'data/train_cases.json','validation':ROOT/'data/selection_cases.json',
           'old64':ROOT/'data/validation_cases.json','fresh32':RUN/'frozen_test_cases.json'}
    banks={k:load_json(v) for k,v in paths.items()}
    inputs={str(v):digest(v) for v in paths.values()}
    inputs.update({str(ROOT/'models/current_ppo.zip'):digest(ROOT/'models/current_ppo.zip'),
                  str(RUN/'models/selected_model.zip'):digest(RUN/'models/selected_model.zip')})
    save(OUT/'protocol.json',{'input_hashes':inputs,'training_performed':False,'test_data_used_for_training':False,
        'fresh32_status_after_diagnosis':'Inspected diagnostic bank; future model changes require another untouched test bank',
        'code_sha256':digest(Path(__file__))})
    try:
        geometry=[row for label,bank in banks.items() for row in scene_features(bank,label)]
        save(OUT/'scene_geometry.json',geometry)
        interfaces=[check_interface(bank[0]) for bank in banks.values()]
        save(OUT/'interface_checks.json',interfaces)
        with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            results=[]
            for label,bank in banks.items():
                model_names=('source','selected') if label=='fresh32' else ('selected',)
                for which in model_names:
                    group=jobs(pool,run_rollout,[(label,c,which,'dynamic') for c in bank],status,f'{label}_{which}_replay')
                    results.extend(group);save(OUT/'rollouts.json',results)
            # Counterfactual controls do not alter the recorded original outcomes.
            for variant in ('zero','no_obstacles','static','rrt'):
                group=jobs(pool,run_rollout,[('fresh32',c,'selected',variant) for c in banks['fresh32']],status,'control_'+variant)
                results.extend(group);save(OUT/'rollouts.json',results)
        for filename,sha in inputs.items():assert digest(filename)==sha,'Diagnostic modified frozen inputs'
        save(OUT/'verification.json',{'frozen_inputs_unchanged':True,'training_performed':False,
            'interface_cases_checked':len(interfaces),'rollouts':len(results)})
        status.update('complete',status='completed')
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
