"""Offline expert collection and actor-only imitation, explicitly invoked by CLI.

The expert sees prescribed motion; the policy sees only the normal 77 inputs.
No expert path is substituted for the obstacle-free policy reference.
"""
import copy
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.rl.avoidance_curriculum import (
    StagedAvoidanceEnv,stage_scene,initialize_gated,INTERFACE_CONTRACT,ReferencePreservingPolicy)
from wall_panel_rl.rl.autonomous_dynamic_ppo import without_obstacles


def save_teacher(path,observations,actions,raw_actions,offsets,seeds,tasks,
                 terminal_obs=None,terminal_actions=None):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,obs=np.asarray(observations,dtype=np.float32).reshape(-1,77),
            actions=np.asarray(actions,dtype=np.float32).reshape(-1,7),
            raw_actions=np.asarray(raw_actions,dtype=np.float32).reshape(-1,7),
            offsets=np.asarray(offsets,dtype=np.int64),seeds=np.asarray(seeds,dtype=np.int64),
            tasks=np.asarray(tasks,dtype='U16'),
            terminal_obs=np.asarray(terminal_obs if terminal_obs is not None else [],dtype=np.float32).reshape(-1,77),
            terminal_actions=np.asarray(terminal_actions if terminal_actions is not None else [],dtype=np.float32).reshape(-1,7))
    temp.replace(path)


class AlignmentTeacher:
    """Follow the policy's own reference/terminal servo; never run independent IK."""
    planning_success=True
    planning_time=0.
    replans=0
    timed=False

    def __init__(self,model):self.model=model

    def action(self,observation):
        with torch.no_grad():
            return self.model.policy.reference_mean(torch.as_tensor(observation[None])).numpy()[0]


def make_teacher(env,model,group,scene,seed):
    if group=='alignment':return AlignmentTeacher(model)
    return ClassicalController(env,seed,goal_q=scene['reference_joint_path'][-1])


def collect_teacher(config,scenes,s,output):
    if s.get('interface_contract')!=INTERFACE_CONTRACT:
        raise ValueError('Recollect teacher with the current reference-preserving contract')
    spec=s['teacher']
    corrections=int(spec.get('correction_episodes',0))
    minimum_avoidance=int(spec.get('minimum_unique_avoidance_scenes',0))
    save_every=int(spec.get('save_every_attempts',1))
    if save_every<1:raise ValueError('Invalid demonstration checkpoint interval')
    if not 0<=corrections<=spec['avoidance_episodes']:
        raise ValueError('Corrections must be included in the avoidance quota')
    if minimum_avoidance>min(spec['avoidance_episodes'],len({v['seed'] for v in scenes})):
        raise ValueError('Too few layouts/episodes for requested avoidance coverage')
    prefix_model=None
    if corrections:
        if not 1<=spec['correction_prefix_min']<=spec['correction_prefix_max']:
            raise ValueError('Invalid correction prefix range')
        prefix_model=PPO.load(spec['correction_model'],device='cpu')
        if not isinstance(prefix_model.policy,ReferencePreservingPolicy):
            raise ValueError('Wrong correction actor interface')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    quotas={'alignment':spec['alignment_episodes'],
            'avoidance':spec['avoidance_episodes']-corrections,'correction':corrections}
    counts={k:0 for k in quotas};rng=np.random.default_rng(s['seed'])
    obs_rows=[];actions=[];raw=[];offsets=[0];seeds=[];tasks=[];attempts=[]
    terminals=[];terminal_labels=[]
    model=initialize_gated(s['source_model'],s)
    env=StagedAvoidanceEnv(config,scenes,s)
    try:
        for attempt in range(spec['maximum_attempts']):
            remaining={k:quotas[k]-counts[k] for k in quotas}
            if not sum(remaining.values()):break
            group=str(rng.choice(list(remaining),p=np.array(list(remaining.values()))/sum(remaining.values())))
            pool=scenes
            used_avoidance={seed for seed,task in zip(seeds,tasks) if task=='avoidance'}
            if group!='alignment' and len(used_avoidance)<minimum_avoidance:
                pool=[v for v in scenes if v['seed'] not in used_avoidance]
            original=pool[int(rng.integers(len(pool)))]
            scene=stage_scene(original,s['active_stage'],rng)
            task='static' if s['active_stage']['mode']=='static' else 'dynamic'
            if group=='alignment':
                scene=without_obstacles(scene,rng.uniform(.02,.15) if rng.random()<.5 else None)
                task='alignment'
            obs,info=env.reset(options={'scenario':scene,'task':task,
                'strength':[s['active_stage']['amplitude'],s['active_stage']['speed']]})
            warmup=[];prefix_failure=None
            if group=='correction':
                limit=int(rng.integers(spec['correction_prefix_min'],spec['correction_prefix_max']+1))
                from wall_panel_rl.rl.avoidance_curriculum import clearance_feedback
                for _ in range(limit):
                    _,distance=clearance_feedback(env.env,spec['correction_handoff_clearance'],0.)
                    if distance is not None and distance<spec['correction_handoff_clearance']:break
                    action=prefix_model.predict(obs,deterministic=True)[0]
                    obs,_,done,truncated,info=env.step(action)
                    warmup.append(info['executed_action'])
                    if done or truncated:
                        prefix_failure='prefix_'+info['termination_reason'];break
                if not warmup:prefix_failure='prefix_no_safe_step'
            expert=(make_teacher(env.env,model,group,scene,int(original['seed'])+attempt)
                    if prefix_failure is None else None)
            episode_obs=[];episode_actions=[];episode_raw=[]
            if expert is not None and expert.planning_success:
                while True:
                    # obs contains the ORIGINAL reference, never expert.path.
                    action=expert.action(obs)
                    next_obs,_,done,truncated,info=env.step(action)
                    episode_obs.append(obs.copy());episode_raw.append(np.asarray(action).copy())
                    episode_actions.append(info['executed_action']);obs=next_obs
                    if done or truncated:break
                reason=info['termination_reason']
            else:reason=prefix_failure or 'planning_failure'
            success=expert is not None and expert.planning_success and reason=='success'
            attempts.append({'attempt':attempt,'group':group,'seed':original['seed'],'reason':reason,
                'steps':env.env.steps,'teacher_timed':bool(expert.timed) if expert else False,
                'planning_seconds':float(expert.planning_time) if expert else 0.,'replans':expert.replans if expert else 0,
                'warmup_actions':warmup,'warmup_steps':len(warmup),
                'teacher_kind':'reference_servo' if group=='alignment' else 'fixed_endpoint_rrt',
                'reference_recoveries':getattr(env.cursor,'recoveries',0),
                'scene':scene})
            if success:
                obs_rows.extend(episode_obs);actions.extend(episode_actions);raw.extend(episode_raw)
                offsets.append(len(obs_rows));seeds.append(original['seed'])
                tasks.append('alignment' if group=='alignment' else 'avoidance');counts[group]+=1
                # Successful post-step states are supervision anchors only, never
                # extra environment transitions or actions executed after done.
                terminals.append(obs.copy())
                terminal_labels.append(AlignmentTeacher(model).action(obs))
            if (attempt==0 or (attempt+1)%save_every==0 or counts==quotas or
                attempt+1==spec['maximum_attempts']):
                save_teacher(output/'teacher.npz',obs_rows,actions,raw,offsets,seeds,tasks,terminals,terminal_labels)
                log=output/'attempts.json.tmp';log.write_text(json.dumps(attempts,indent=2))
                log.replace(output/'attempts.json')
            print('teacher',attempt+1,counts,reason,flush=True)
        avoidance_seeds={seed for seed,task in zip(seeds,tasks) if task=='avoidance'}
        complete=counts==quotas and len(avoidance_seeds)>=minimum_avoidance
        report={'completed':complete,
            'counts':{'alignment':counts['alignment'],'avoidance':counts['avoidance']+counts['correction']},
            'group_counts':counts,'quotas':quotas,'attempts':len(attempts),
            'transitions':len(obs_rows),'unique_scene_seeds':len(set(seeds)),
            'settings':s,'reference_kind':'obstacle_free','action_labels':'executed_joint_commands',
            'interface_contract':INTERFACE_CONTRACT,'terminal_anchors':len(terminals),
            'alignment_teacher':'reference_servo','avoidance_goal':'original_reference_endpoint',
            'expert_runtime':'offline_only','validation_used_for_teacher':False,
            'environment_config':config,
            'teacher_sha256':hashlib.sha256((output/'teacher.npz').read_bytes()).hexdigest()}
        report['unique_avoidance_scene_seeds']=len(avoidance_seeds)
        report['source_sha256']=hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest()
        if prefix_model is not None:
            report['correction_model_sha256']=hashlib.sha256(Path(spec['correction_model']).read_bytes()).hexdigest()
            report['correction_prefix_not_action_labels']=True
        (output/'metadata.json').write_text(json.dumps(report,indent=2))
        if not complete:raise RuntimeError('Teacher quota not met; attempts retained; no BC training permitted')
    finally:env.close()


def imitation_split(offsets,seeds,seed):
    unique=np.unique(seeds)
    if len(unique)<2:raise ValueError('At least two distinct scene seeds required')
    rng=np.random.default_rng(seed);rng.shuffle(unique)
    held=set(unique[:max(1,int(np.ceil(.2*len(unique))))].tolist())
    train=np.array([i for i,v in enumerate(seeds) if v not in held],dtype=int)
    validation=np.array([i for i,v in enumerate(seeds) if v in held],dtype=int)
    if not len(train) or not len(validation):raise ValueError('Empty imitation split')
    return train,validation


def teacher_reuse_contract(meta,s,config):
    """Permit optimization/output changes while binding physical/data semantics.

    Older v3 data has its source hash attested by its original BC metadata. Never
    rewrite that metadata or infer provenance from a currently mutable filename.
    """
    from wall_panel_rl.utils.config import ROOT
    keys=('interface_contract','active_stage','reference_recovery','terminal_radius',
          'joint_margin','progress_weight','clearance_distance','clearance_weight','seed',
          'residual_scale','initial_prior_gate')
    if meta.get('interface_contract')!=INTERFACE_CONTRACT or meta['environment_config']!=config:
        raise ValueError('Data interface/physical configuration changed')
    if any(meta['settings'].get(k)!=s.get(k) for k in keys):
        raise ValueError('Data reference/control contract changed')
    source_sha=hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest()
    recorded=meta.get('source_sha256')
    if recorded is None:
        proof=json.loads((ROOT/meta['settings']['stage_output']/'bc/metadata.json').read_text())
        if proof['teacher_sha256']!=meta['teacher_sha256']:
            raise ValueError('Original BC metadata does not attest this teacher data')
        recorded=proof['source_sha256']
    if recorded!=source_sha:raise ValueError('Data source weights changed')
    return {'interface_contract':INTERFACE_CONTRACT,'source_sha256':source_sha,
        'teacher_sha256':meta['teacher_sha256'],'frozen_settings':{k:s[k] for k in keys},
        'environment_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}


def closed_loop_score(result,loss):
    checks=result['checks'];target=result['metrics']['dynamic_ppo']['stage_target']
    retention=all(checks[k] for k in ('alignment_samples','alignment_retained',
                                     'alignment_collision_free','goal_convergence'))
    return (int(result['passed']),int(retention),target['success'],-target['collision'],
            -target['joint_limit'],-target['timeout'],-float(loss))


def pretrain_actor(model,teacher_dir,s,output,*,reuse_environment=None,evaluator=None,
                   resume_state=None,heldout_seeds=None,include_initial=False):
    teacher_dir=Path(teacher_dir);output=Path(output)
    if output.exists():raise FileExistsError(output)
    meta=json.loads((teacher_dir/'metadata.json').read_text())
    teacher_path=teacher_dir/'teacher.npz'
    if not meta['completed'] or hashlib.sha256(teacher_path.read_bytes()).hexdigest()!=meta['teacher_sha256']:
        raise ValueError('Teacher data incomplete or changed')
    if (s.get('interface_contract')!=INTERFACE_CONTRACT or
        meta.get('interface_contract')!=INTERFACE_CONTRACT or
        not isinstance(model.policy,ReferencePreservingPolicy)):
        raise ValueError('Teacher/policy contract is obsolete; recollect v3 demonstrations')
    reuse=None
    if reuse_environment is None:
        if meta['settings']!=s:raise ValueError('Teacher settings changed; recollect explicitly')
    else:reuse=teacher_reuse_contract(meta,s,reuse_environment)
    if meta['settings']['active_stage']!=s['active_stage']:
        raise ValueError('Teacher stage mismatch')
    if meta['unique_scene_seeds']<s['teacher']['minimum_unique_scenes']:
        raise ValueError('Too few unique teacher scenes')
    with np.load(teacher_path,allow_pickle=False) as d:
        observations=torch.as_tensor(d['obs'].copy());actions=torch.as_tensor(d['actions'].copy())
        offsets=d['offsets'].copy();seeds=d['seeds'].copy()
        tasks=d['tasks'].copy()
        terminal_obs=torch.as_tensor(d['terminal_obs'].copy())
        terminal_actions=torch.as_tensor(d['terminal_actions'].copy())
    if (observations.ndim!=2 or observations.shape[1]!=77 or actions.shape!=(len(observations),7)
        or len(offsets)!=len(seeds)+1 or offsets[0]!=0 or offsets[-1]!=len(observations)
        or np.any(np.diff(offsets)<=0) or not torch.isfinite(observations).all()
        or not torch.isfinite(actions).all() or torch.any(actions.abs()>1.000001)):
        raise ValueError('Malformed teacher arrays')
    if (tasks.shape!=seeds.shape or set(tasks.tolist())!={'alignment','avoidance'} or
        terminal_obs.shape!=(len(seeds),77) or terminal_actions.shape!=(len(seeds),7) or
        not torch.isfinite(terminal_obs).all() or not torch.isfinite(terminal_actions).all() or
        torch.any(terminal_actions.abs()>1.000001)):
        raise ValueError('Missing or malformed task/terminal supervision')
    if len(np.unique(seeds))<s['teacher']['minimum_unique_scenes']:
        raise ValueError('Actual unique teacher scene count below minimum')
    if heldout_seeds is None:
        train,held=imitation_split(offsets,seeds,s['seed'])
    else:
        held_set=set(map(int,heldout_seeds))
        if not held_set or not held_set.issubset(set(seeds.tolist())):
            raise ValueError('Fixed held-out seeds absent from teacher data')
        held=np.flatnonzero(np.isin(seeds,list(held_set)))
        train=np.flatnonzero(~np.isin(seeds,list(held_set)))
    partitions={name:{task:episodes[tasks[episodes]==task] for task in ('alignment','avoidance')}
                for name,episodes in [('train',train),('heldout',held)]}
    if any(not len(v) for group in partitions.values() for v in group.values()):
        raise ValueError('Both tasks must be represented in train and held-out scenes')
    held_indices={task:np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in episodes])
                  for task,episodes in partitions['heldout'].items()}
    params=[p for n,p in model.policy.named_parameters() if n.startswith(('residual.','prior_gate.'))]
    before={n:p.detach().clone() for n,p in model.policy.named_parameters()
            if not n.startswith(('residual.','prior_gate.'))}
    optimizer=torch.optim.Adam(params,lr=s['teacher']['learning_rate'])
    rng=np.random.default_rng(s['seed']);best=None;best_loss=float('inf');history=[]
    previous_updates=0
    if resume_state is not None:
        state=torch.load(resume_state,map_location='cpu',weights_only=False)
        previous_updates=int(state['updates'])
        if previous_updates<=0:raise ValueError('Invalid BC continuation counter')
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
        # A continuation may explicitly change LR, while retaining Adam moments.
        for group in optimizer.param_groups:group['lr']=s['teacher']['learning_rate']
    spec=s['teacher']
    checkpoint_steps=set(spec.get('closed_loop_updates',[spec['updates']]))
    if evaluator is not None:
        if spec['updates'] not in checkpoint_steps or any(type(v) is not int or v<=0 or v>spec['updates'] or v%100 for v in checkpoint_steps):
            raise ValueError('Closed-loop checkpoints must be logged 100-update boundaries including final')
        output.mkdir(parents=True)
    best_score=None;selected_update=None;selected_metrics=None
    def checkpoint(update,losses,validation):
        nonlocal best_score,best_loss,best,selected_update,selected_metrics
        path=output/'checkpoints'/f'bc_{update}.zip';path.parent.mkdir(exist_ok=True)
        model.save(path)
        torch.save({'updates':previous_updates+update,'optimizer':optimizer.state_dict(),
            'numpy_rng':rng.bit_generator.state,'torch_rng':torch.get_rng_state()},
            path.with_suffix('.bc_state.pt'))
        torch_rng=torch.get_rng_state();numpy_rng=np.random.get_state();python_rng=random.getstate()
        try:metrics=evaluator(PPO.load(path,device='cpu'),update,losses)
        finally:
            torch.set_rng_state(torch_rng);np.random.set_state(numpy_rng);random.setstate(python_rng)
        score=closed_loop_score(metrics,validation)
        event={'update':update,'cumulative_bc_updates':previous_updates+update,
               'score':score,'validation':metrics,'losses':losses}
        with (output/'closed_loop_history.jsonl').open('a') as stream:
            stream.write(json.dumps(event,allow_nan=False)+'\n')
        if best_score is None or score>best_score:
            best_score=score;best_loss=validation;best=copy.deepcopy(model.policy.state_dict())
            selected_update=update;selected_metrics=metrics
        print('BC checkpoint',update,'passed',metrics['passed'],
              'target',metrics['metrics']['dynamic_ppo']['stage_target'],flush=True)
    if include_initial:
        if evaluator is None:raise ValueError('Initial retention requires a closed-loop evaluator')
        with torch.no_grad():
            mse={task:float(torch.nn.functional.mse_loss(model.policy.tracking_mean(observations[idx]),actions[idx]))
                 for task,idx in held_indices.items()}
            tmse=float(torch.nn.functional.mse_loss(model.policy.tracking_mean(terminal_obs[held]),terminal_actions[held]))
        validation=float(np.mean(list(mse.values())))+spec['terminal_weight']*tmse
        history.append({'update':0,'heldout_scene_mse':float(np.mean(list(mse.values()))),
            'heldout_task_mse':mse,'heldout_terminal_mse':tmse,'selection_objective':validation})
        checkpoint(0,history[-1],validation)
    def objective(indices,terminal_indices):
        prediction=model.policy.tracking_mean(observations[indices])
        imitation=torch.nn.functional.mse_loss(prediction,actions[indices])
        terminal=torch.nn.functional.mse_loss(model.policy.tracking_mean(terminal_obs[terminal_indices]),
                                              terminal_actions[terminal_indices])
        strength,gate,residual=model.policy.correction_components(observations[indices])
        # Mild priors discourage wholesale shutdown while retaining freedom to
        # wait/deviate where imitation demonstrates that this is necessary.
        regularization=(spec['gate_retention_weight']*(strength*(1-gate).square()).mean()+
                        spec['residual_weight']*(strength*residual.square()).mean())
        return imitation+spec['terminal_weight']*terminal+regularization
    for step in range(s['teacher']['updates']):
        half=s['teacher']['batch_size']//2
        episodes=np.r_[rng.choice(partitions['train']['alignment'],size=half),
                       rng.choice(partitions['train']['avoidance'],size=s['teacher']['batch_size']-half)]
        lo,hi=offsets[episodes],offsets[episodes+1]
        indices=lo+(rng.random(len(episodes))*(hi-lo)).astype(int)
        loss=objective(indices,episodes)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if (step+1)%100==0 or step+1==s['teacher']['updates']:
            with torch.no_grad():
                task_mse={task:float(torch.nn.functional.mse_loss(
                    model.policy.tracking_mean(observations[idx]),actions[idx])) for task,idx in held_indices.items()}
                terminal_mse=float(torch.nn.functional.mse_loss(model.policy.tracking_mean(terminal_obs[held]),terminal_actions[held]))
                validation=float(np.mean(list(task_mse.values())))+spec['terminal_weight']*terminal_mse
            history.append({'update':step+1,'sampled_train_objective':float(loss.detach()),
                'heldout_scene_mse':float(np.mean(list(task_mse.values()))),'heldout_task_mse':task_mse,
                'heldout_terminal_mse':terminal_mse,'selection_objective':validation})
            if evaluator is None:
                if validation<best_loss:
                    best_loss=validation;best=copy.deepcopy(model.policy.state_dict());selected_update=step+1
            elif step+1 in checkpoint_steps:
                checkpoint(step+1,history[-1],validation)
            if evaluator is not None:
                (output/'progress.json').write_text(json.dumps({'updates':step+1,'budget':spec['updates']}))
                (output/'loss_history.json').write_text(json.dumps(history,indent=2))
    if best is None:raise RuntimeError('No BC candidate')
    model.policy.load_state_dict(best)
    for name,param in model.policy.named_parameters():
        if name in before and not torch.equal(param.detach(),before[name]):raise AssertionError(f'BC changed {name}')
    if evaluator is None:output.mkdir(parents=True)
    model.save(output/'model.zip')
    report={'bc_updates':s['teacher']['updates'],'ppo_environment_steps':0,
        'previous_bc_updates':previous_updates,'cumulative_bc_updates':previous_updates+s['teacher']['updates'],
        'selected_cumulative_bc_updates':previous_updates+selected_update,
        'bc_optimizer_resumed':resume_state is not None,
        'resume_state_sha256':hashlib.sha256(Path(resume_state).read_bytes()).hexdigest() if resume_state else None,
        'source_steps':model.num_timesteps,'heldout_selection_objective':best_loss,
        'heldout_scene_mse':next(r for r in history if r['update']==selected_update)['heldout_scene_mse'],
        'selected_update':selected_update,'selection':'closed_loop_then_loss' if evaluator else 'heldout_loss',
        'selected_closed_loop':selected_metrics,'data_reuse_contract':reuse,
        'interface_contract':INTERFACE_CONTRACT,'task_sampling':'balanced_alignment_avoidance',
        'source_sha256':hashlib.sha256(Path(s['source_model']).read_bytes()).hexdigest(),
        'train_scene_seeds':sorted(set(seeds[train].tolist())),
        'heldout_scene_seeds':sorted(set(seeds[held].tolist())),
        'teacher_sha256':meta['teacher_sha256'],'actor_only_updates':True,
        'critic_and_prior_unchanged':True,'history':history,'settings':s}
    (output/'metadata.json').write_text(json.dumps(report,indent=2))
