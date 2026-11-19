"""Finish interrupted DAgger collection and exit, with no optimizer calls.

Replay saved direct-policy traces to reconstruct unflushed labels. Only states
where the original teacher changed the action need a fresh predictive query.
Trace mismatches cause a full collection-only rerun, never a parameter update.
"""
import _bootstrap
from concurrent.futures import ProcessPoolExecutor
import copy,json,multiprocessing as mp,os,shutil
from pathlib import Path
import numpy as np
from run_standalone import collect,worker_model
from run_adaptation import WORK,initialize_worker,Status,jobs,save,digest
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv,planning_snapshot,assert_planning_paused
from wall_panel_rl.rl.predictive_safety import PredictiveSafetyController
from wall_panel_rl.rl.waypoint_tracking import bounded_action
from wall_panel_rl.submission import load_json
from wall_panel_rl.utils.config import ROOT


def reconstruct(job):
    model_path,case,old=job
    if old is None:return collect((model_path,case,False))
    model=worker_model(model_path);rt=WORK['runtime']
    env=SlowDynamicFourEnv(rt['configuration'],[],rt['settings'])
    xs=[];ys=[];weights=[]
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        predictor=PredictiveSafetyController(model,env,12,rt['controller']['gains'])
        for index,expected in enumerate(old['trace']):
            student=model.predict(obs,deterministic=True)[0]
            valid=expected['teacher_safe_horizon'];changed=expected['teacher_changed_action']
            if valid:
                if changed:
                    before=planning_snapshot(env);no_safe=predictor.no_safe_action
                    target=predictor.action(obs)
                    assert_planning_paused(env,before)
                    if predictor.no_safe_action!=no_safe:raise ValueError('Teacher no longer safe on reconstructed state')
                else:target=student
                q,_=env.env.joint_state()
                target=bounded_action(q,np.clip(target,-1,1),env.env.lower,env.env.upper,env.env.max_increment,env.margin)
                xs.append(obs.copy());ys.append(target);weights.append(4. if changed else .5)
            obs,_,done,truncated,info=env.step(student)
            if abs(float(info['position_error'])-expected['position_m'])>1e-6:
                raise ValueError('Physical trace mismatch')
            if bool(done or truncated)!=(index==len(old['trace'])-1):raise ValueError('Episode boundary mismatch')
        if info['termination_reason']!=old['reason']:raise ValueError('Outcome mismatch')
        result=([old],np.asarray(xs,dtype=np.float32).reshape(-1,77),
                np.asarray(ys,dtype=np.float32).reshape(-1,7),np.asarray(weights,dtype=np.float32))
    except ValueError as exc:
        print('Recollecting scene',case['scene']['seed'],str(exc),flush=True)
        result=None
    finally:env.close()
    return result if result is not None else collect((model_path,case,False))


def main():
    out=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics/standalone_ppo_v22_seed20261112')
    request=load_json(out/'stop_after_collection_request.json')
    path=request['collection_model'];source_hash=digest(path)
    checkpoint_hashes={p.name:digest(p) for p in (out/'models').glob('*.zip')}
    initial_hash=digest(out/'initial_teacher_labels.npz')
    rows=load_json(out/'dagger_episodes.json')
    backup=out/'dagger_interrupted_episodes.json'
    if not backup.exists():shutil.copy2(out/'dagger_episodes.json',backup)
    existing={r['seed']:r for r in rows}
    bank=load_json(ROOT/'data/train_cases.json')
    assert len(bank)==100
    status=Status(out);status.state.update(load_json(out/'status.json'))
    status.state.update(pid=os.getpid(),status='running',optimizer_updates_disabled=True,ppo_updates_started=False)
    logs=[];xs=[];ys=[];weights=[];scene_ids=[]
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
        for start in range(0,len(bank),4):
            batch=bank[start:start+4]
            result=jobs(pool,reconstruct,[(path,c,existing.get(c['scene']['seed'])) for c in batch],status,'finish_collection_only')
            for case,(record,x,y,w) in zip(batch,result):
                logs.extend(record);xs.extend(x);ys.extend(y);weights.extend(w);scene_ids.extend([case['scene']['seed']]*len(x))
            save(out/'dagger_episodes.json',logs)
            target=out/'dagger_labels.npz';temp=target.with_suffix('.npz.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,observations=np.asarray(xs,dtype=np.float32).reshape(-1,77),
                    actions=np.asarray(ys,dtype=np.float32).reshape(-1,7),weights=np.asarray(weights,dtype=np.float32),scene_ids=np.asarray(scene_ids))
            temp.replace(target)
            status.update('finish_collection_only',collected_scenes=min(start+4,len(bank)),collection_scenes=len(bank),labels=len(xs))
    assert len(logs)==100 and len({r['seed'] for r in logs})==100
    assert digest(path)==source_hash and digest(out/'initial_teacher_labels.npz')==initial_hash
    assert checkpoint_hashes=={p.name:digest(p) for p in (out/'models').glob('*.zip')}
    assert not list((out/'models').glob('ppo_*.zip'))
    assert not list(out.glob('arm*_episodes*'))
    initial=np.load(out/'initial_teacher_labels.npz')
    extra=np.load(out/'dagger_labels.npz')
    assert np.isfinite(extra['observations']).all() and np.isfinite(extra['actions']).all()
    save(out/'collection_complete.json',{'initial_training_scenes':100,'initial_labels':len(initial['actions']),
        'dagger_training_scenes':100,'dagger_labels':len(extra['actions']),
        'already_completed_supervised_updates_before_stop_request':request['already_completed_supervised_updates'],
        'new_supervised_updates_since_stop_request':0,'ppo_steps':0,
        'collection_model':path,'collection_model_sha256':source_hash,
        'all_existing_checkpoints_unchanged':True,'requires_explicit_instruction_before_training':True})
    status.update('collection_complete_training_paused',status='paused',collected_scenes=100,
        optimizer_updates_disabled=True,ppo_steps=0,background_work_finished=True)
    print('COLLECTION COMPLETE; NO TRAINING PROCESS WILL BE STARTED',flush=True)
if __name__=='__main__':main()
