"""Read-only evaluation diagnostics: plan discovery is separate from execution."""
import copy
import time
import numpy as np
import pybullet as p
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import make_rrt,episode_record


REASONS=('success','collision','timeout','joint_limit','planning_failure')


def contacts_at_failure(env,scene):
    """Mirror the environment's collision exclusions and preserve obstacle slots."""
    margin=float(env.config['collision']['margin']); found={}
    pairs=[]
    for i,body in enumerate(env.obstacle_ids):
        pairs.extend([(env.robot,body,'obstacle','arm',i+1),(env.panel,body,'obstacle','panel',i+1)])
    pairs.extend([(env.robot,env.wall,'wall','arm',None),(env.panel,env.wall,'wall','panel',None),
                  (env.robot,env.robot,'self','arm',None),(env.robot,env.panel,'arm_panel','arm',None)])
    for a,b,kind,moving,slot in pairs:
        candidates=list(p.getContactPoints(a,b,physicsClientId=env.client))
        candidates+=list(p.getClosestPoints(a,b,max(0.,margin),physicsClientId=env.client))
        for c in candidates:
            if c[8]>margin or (kind=='self' and abs(c[3]-c[4])<=1) or (kind=='arm_panel' and c[3]==env.ee):continue
            key=(kind,moving,slot,int(c[3]),int(c[4]))
            record={'kind':kind,'moving_body':moving,'obstacle_slot':slot,
                'moving_link_index':int(c[3]),'other_link_index':int(c[4]),'distance_m':float(c[8])}
            if slot is not None:
                placement=scene.get('on_path_placement',{}).get('obstacles',[])
                record['nominal_path_fraction']=placement[slot-1]['actual_fraction'] if placement else None
            if key not in found or c[8]<found[key]['distance_m']:found[key]=record
    return sorted(found.values(),key=lambda c:(c['distance_m'],c['kind'],c['moving_body']))


def run_episode(model,config,settings,case,controller,planner_seed=None):
    if controller not in ('ppo','rrt'):raise ValueError('Use a pure controller for paired evaluation')
    env=ConditionalRecoveryEnv(copy.deepcopy(config),[],settings)
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        expert=make_rrt(env,planner_seed) if controller=='rrt' else None
        planned=np.asarray(expert.path).tolist() if expert is not None and expert.planning_success else []
        total=0.;joints=[env.env.joint_state()[0].tolist()];positions=[list(env.env.panel_pose()[0])]
        start=time.perf_counter()
        if expert is not None and not expert.planning_success:
            info=dict(info,termination_reason='planning_failure')
        else:
            while True:
                action=expert.action(obs) if expert is not None else model.predict(obs,deterministic=True)[0]
                obs,reward,done,truncated,info=env.step(action);total+=reward
                joints.append(env.env.joint_state()[0].tolist());positions.append(list(env.env.panel_pose()[0]))
                if done or truncated:break
        elapsed=time.perf_counter()-start
        row=episode_record(env,info,controller,total,expert.planning_time if expert else 0.)
        row.update(suite=case['suite'],planner_seed=int(planner_seed) if planner_seed is not None else None,
            execution_wall_seconds=elapsed)
        row['planning_success']=bool(expert.planning_success) if expert else None
        row['goal_endpoint_valid']=bool(expert.ik_success) if expert else None
        row['static_path_verified']=getattr(expert,'static_path_verified',None) if expert else None
        row['static_validation_resolution']=getattr(expert,'static_validation_resolution',None) if expert else None
        row['rejected_path_candidates']=getattr(expert.planner,'rejected_paths',0) if expert and hasattr(expert,'planner') else None
        row['contacts']=contacts_at_failure(env.env,case['scene']) if row['reason']=='collision' else []
        row['dominant_contact']=row['contacts'][0] if row['contacts'] else None
        row['first_collision_step']=env.env.steps if row['reason']=='collision' else None
        row['first_collision_sim_seconds']=float(env.env.sim_time) if row['reason']=='collision' else None
        row['position_within_tolerance']=row['position_m']<config['target']['position_tolerance']
        row['orientation_within_tolerance']=row['orientation_deg']<config['target']['orientation_tolerance_deg']
        # Diagnostic recheck happens AFTER physical execution and cannot filter
        # or replace the primary baseline action sequence or its result.
        invalid=[];begin=time.perf_counter()
        for index,(a,b) in enumerate(zip(planned[:-1],planned[1:])):
            if not env.env.collision.edge_valid(np.asarray(a),np.asarray(b)):invalid.append(index)
        row['planned_route_recheck']=None if not planned else {'invalid_edge_indices':invalid,'passed':not invalid,
            'seconds':time.perf_counter()-begin,'performed_after_execution':True}
        return row,{'planned_joint_route':planned,'executed_joint_route':joints,'executed_panel_positions':positions}
    finally:env.close()


def counts(rows):
    result={'episodes':len(rows),**{reason:sum(r['reason']==reason for r in rows) for reason in REASONS}}
    if sum(result[k] for k in REASONS)!=len(rows):raise ValueError('Unknown/nonexclusive termination reason')
    return result


def feasibility(primary,retries):
    if len({r['seed'] for r in primary})!=len(primary):raise ValueError('Duplicate primary scene')
    by_seed={r['seed']:[r] for r in primary}
    for row in retries:
        if row['seed'] not in by_seed:raise ValueError('Retry of non-primary scene')
        by_seed[row['seed']].append(row)
    return {'scenes':len(primary),'primary':counts(primary),'retry_trials':counts(retries),
        'scenes_with_any_found_plan':sum(any(r['planning_success'] for r in rows) for rows in by_seed.values()),
        'scenes_with_any_successful_execution':sum(any(r['reason']=='success' for r in rows) for rows in by_seed.values()),
        'unresolved_scene_seeds':[seed for seed,rows in by_seed.items() if not any(r['reason']=='success' for r in rows)],
        'failure_is_not_a_proof_of_infeasibility':True}
