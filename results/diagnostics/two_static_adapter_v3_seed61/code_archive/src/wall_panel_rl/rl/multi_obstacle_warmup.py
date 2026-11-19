"""Actor-only BC warm-up with task balancing and obstacle-order augmentation."""
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch


GROUPS=('two_static','one_static','alignment')


def swap_two_slots(observations,rng,probability=.5):
    """Swap only two present obstacle blocks, leaving q/goal/ref/empty slots intact."""
    result=observations.clone();slots=result[:,20:70].reshape(-1,5,10)
    active=(slots[:,:,9]>.5).sum(1)==2
    selected=active & torch.as_tensor(rng.random(len(result))<probability,device=result.device)
    indices=torch.where(selected)[0]
    # Canonical two-static scenes pack obstacles in slots 0 and 1.
    if len(indices):
        if not torch.all((slots[indices,0,9]>.5)&(slots[indices,1,9]>.5)):
            raise ValueError('Expected packed two-obstacle observations')
        first=slots[indices,0,:].clone()
        slots[indices,0,:]=slots[indices,1,:];slots[indices,1,:]=first
    return result


def teacher_arrays(path,heldout):
    with np.load(path,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    n=len(data['obs']);episodes=len(data['seeds'])
    if (data['obs'].shape!=(n,77) or data['actions'].shape!=(n,7) or
        data['offsets'].shape!=(episodes+1,) or data['offsets'][0]!=0 or data['offsets'][-1]!=n or
        np.any(np.diff(data['offsets'])<=0) or data['terminal_obs'].shape!=(episodes,77) or
        data['terminal_actions'].shape!=(episodes,7) or set(data['tasks'].tolist())!=set(GROUPS)):
        raise ValueError('Malformed multi-task teacher arrays')
    for key in ('obs','actions','terminal_obs','terminal_actions'):
        if not np.isfinite(data[key]).all():raise ValueError('Non-finite teacher data')
    if np.max(np.abs(data['actions']))>1.000001:raise ValueError('Unbounded action labels')
    heldout=set(map(int,heldout))
    if not heldout or not heldout.issubset(set(data['seeds'].tolist())):raise ValueError('Held-out layouts missing')
    partitions={}
    for name,mask in [('train',~np.isin(data['seeds'],list(heldout))),('heldout',np.isin(data['seeds'],list(heldout)))]:
        partitions[name]={group:np.flatnonzero(mask&(data['tasks']==group)) for group in GROUPS}
        if any(not len(v) for v in partitions[name].values()):raise ValueError('Missing group in scene split')
    return data,partitions


def warmup_actor(model,teacher_path,heldout,spec,output,evaluator):
    """The evaluator ranks only fixed validation cases, never held-out test scenes."""
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    data,parts=teacher_arrays(teacher_path,heldout)
    obs=torch.as_tensor(data['obs']);actions=torch.as_tensor(data['actions'])
    terminal=torch.as_tensor(data['terminal_obs']);terminal_actions=torch.as_tensor(data['terminal_actions'])
    actor_prefixes=getattr(model.policy,'imitation_prefixes',('residual.','prior_gate.'))
    params=[p for name,p in model.policy.named_parameters() if name.startswith(actor_prefixes) and p.requires_grad]
    if not params:raise ValueError('Expected reference-preserving actor')
    unchanged={name:p.detach().clone() for name,p in model.policy.named_parameters()
               if not name.startswith(actor_prefixes)}
    optimizer=torch.optim.Adam(params,lr=spec['learning_rate']);rng=np.random.default_rng(spec.get('seed',61))
    weights=np.array([spec['task_weights'][group] for group in GROUPS],dtype=float)
    if not np.isclose(weights.sum(),1) or np.any(weights<=0):raise ValueError('Invalid task weights')
    held_rows={group:np.concatenate([np.arange(data['offsets'][i],data['offsets'][i+1]) for i in ids])
               for group,ids in parts['heldout'].items()}
    best=None;selected=None;history=[]
    def checkpoint(update):
        nonlocal best,selected
        with torch.no_grad():
            held_loss={group:float(torch.nn.functional.mse_loss(model.policy.tracking_mean(obs[idx]),actions[idx]))
                       for group,idx in held_rows.items()}
        state=(torch.get_rng_state(),np.random.get_state(),random.getstate())
        try:result=evaluator(model,update)
        finally:torch.set_rng_state(state[0]);np.random.set_state(state[1]);random.setstate(state[2])
        event={'update':update,'heldout_mse':held_loss,**result}
        path=output/'checkpoints'/f'bc_{update}.zip';path.parent.mkdir(exist_ok=True)
        model.save(path)
        torch.save({'updates':update,'optimizer':optimizer.state_dict(),'rng':rng.bit_generator.state,
            'torch_rng':torch.get_rng_state(),'heldout_seeds':sorted(map(int,heldout))},path.with_suffix('.bc_state.pt'))
        if best is None or tuple(event['score'])>best:
            best=tuple(event['score']);selected=copy.deepcopy(event);model.save(output/'model.zip')
            torch.save(torch.load(path.with_suffix('.bc_state.pt'),map_location='cpu',weights_only=False),output/'selected_bc_state.pt')
        history.append(event)
        (output/'history.json').write_text(json.dumps(history,indent=2))
        print('WARMUP_CHECK',json.dumps(event),flush=True)
    checkpoint(0)
    for update in range(1,spec['updates']+1):
        chosen=rng.choice(len(GROUPS),size=spec['batch_size'],p=weights);ids=np.empty(len(chosen),dtype=int)
        for i,group in enumerate(GROUPS):
            mask=chosen==i;pool=parts['train'][group]
            options={}
            if 'priority_episode_start' in spec:
                priority=np.where(pool>=spec['priority_episode_start'],
                                  spec.get('priority_episode_weight',1.),1.)
                options['p']=priority/priority.sum()
            ids[mask]=rng.choice(pool,size=int(mask.sum()),**options)
        lo,hi=data['offsets'][ids],data['offsets'][ids+1]
        indices=lo+(rng.random(len(ids))*(hi-lo)).astype(int)
        x=swap_two_slots(obs[indices],rng,spec['swap_two_obstacles_probability'])
        prediction=model.policy.tracking_mean(x)
        with torch.no_grad():
            baseline=getattr(model.policy,'base_mean',model.policy.reference_mean)
            deviation=(baseline(x)-actions[indices]).square().mean(1).clamp(0,1)
            importance=1+spec.get('correction_weight',0.)*deviation
        imitation=(importance*(prediction-actions[indices]).square().mean(1)).sum()/importance.sum()
        tx=swap_two_slots(terminal[ids],rng,spec['swap_two_obstacles_probability'])
        anchor=torch.nn.functional.mse_loss(model.policy.tracking_mean(tx),terminal_actions[ids])
        strength,gate,residual=model.policy.correction_components(x)
        loss=imitation+spec['terminal_weight']*anchor+spec['gate_retention_weight']*(strength*(1-gate).square()).mean()+spec['residual_weight']*(strength*residual.square()).mean()
        if hasattr(model.policy,'adapter_correction'):
            loss=loss+spec.get('adapter_weight',.001)*model.policy.adapter_correction(x).square().mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if update%100==0:
            (output/'progress.json').write_text(json.dumps({'updates':update,'budget':spec['updates'],'loss':float(loss.detach())}))
        if update in spec['checkpoints']:checkpoint(update)
    for name,p in model.policy.named_parameters():
        if name in unchanged and not torch.equal(p.detach(),unchanged[name]):raise AssertionError(f'BC changed {name}')
    metadata={'updates':spec['updates'],'selected':selected,'history':history,
        'train_scene_seeds':sorted(set(data['seeds'][np.concatenate(list(parts['train'].values()))].tolist())),
        'heldout_scene_seeds':sorted(map(int,heldout)),'prior_critic_and_std_unchanged':True,
        'ppo_steps_during_warmup':0,'task_weights':spec['task_weights'],'augmentation':'swap_two_present_slots'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    return output/'model.zip',metadata
