"""Resume v22 from completed label files; no scene/expert collection is run.

The paused 3,000-update stage is retained. Resume at aggregate-data fitting,
then the two predeclared PPO arms and direct-policy-only evaluation.
"""
import _bootstrap
import argparse,json,multiprocessing as mp,os,shutil,traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from run_adaptation import Status,save,digest,initialize_worker
from run_standalone import Supervised,AnchoredPPO,converted,evaluate,score
from wall_panel_rl.submission import load_json,CaseBankEnv
from wall_panel_rl.utils.config import ROOT


def inputs(output):
    finished=load_json(output/'collection_complete.json')
    if not finished['requires_explicit_instruction_before_training']:
        raise ValueError('Unexpected collection contract')
    source=Path(finished['collection_model'])
    assert digest(source)==finished['collection_model_sha256']
    train=load_json(ROOT/'data/train_cases.json')
    validation=load_json(ROOT/'data/selection_cases.json')
    regression=load_json(ROOT/'data/validation_cases.json')
    test=load_json(output/'frozen_test_cases.json')
    ids=lambda bank:{int(c['scene']['seed']) for c in bank}
    train_ids=ids(train);val_ids=ids(validation);old_ids=ids(regression);test_ids=ids(test)
    assert len(train)==len(train_ids)==100
    assert len(validation)==len(val_ids)==16
    assert len(test)==len(test_ids)==32
    assert not train_ids&(val_ids|old_ids|test_ids)
    assert not test_ids&(val_ids|old_ids)
    arrays=[]
    for filename,expected in [('initial_teacher_labels.npz',finished['initial_labels']),('dagger_labels.npz',finished['dagger_labels'])]:
        with np.load(output/filename,allow_pickle=False) as archive:
            data={key:archive[key].copy() for key in ('observations','actions','weights','scene_ids')}
        n=len(data['actions'])
        assert n==expected and data['observations'].shape==(n,77) and data['actions'].shape==(n,7)
        assert data['weights'].shape==data['scene_ids'].shape==(n,)
        assert all(np.isfinite(v).all() for v in data.values())
        assert (data['weights']>0).all() and np.max(np.abs(data['actions']))<=1.000001
        assert set(data['scene_ids']).issubset(train_ids)
        arrays.append(data)
    merged={key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    selection=load_json(output/'selection.json')
    best=Path(selection['model'])
    baseline=load_json(output/'evaluations/source_validation.json')
    label=selection['selected']
    selected=baseline if label=='source' else load_json(output/'evaluations'/f'{label}.json')
    assert digest(best)==selected['model_sha256']
    assert not (output/'frozen_selection.json').exists(),'Final selection already frozen; inspect before resuming'
    assert not list((output/'models').glob('ppo_arm*.zip')),'PPO has already started; explicit checkpoint resume required'
    assert not (output/'models/dagger_500.zip').exists(),'Aggregate fitting already started'
    original=load_json(output/'protocol.json')
    for name,sha in original['code_hashes'].items():assert digest(ROOT/name)==sha,f'Original source changed: {name}'
    return dict(source=source,train=train,validation=validation,regression=regression,test=test,
                data=merged,best=best,best_label=label,best_result=selected,baseline=baseline,finished=finished)


def run(args):
    output=args.output.resolve();prepared=inputs(output)
    if args.check_only:
        print(json.dumps({'check_passed':True,'aggregate_labels':len(prepared['data']['actions']),
            'training_scenes':len(prepared['train']),'validation_scenes':len(prepared['validation']),
            'frozen_test_scenes':len(prepared['test']),'resume_model':str(prepared['best']),
            'supervised_updates_to_skip':prepared['finished']['already_completed_supervised_updates_before_stop_request'],
            'remaining_supervised_updates':1500,'ppo_steps_per_arm':args.steps,'arms':2},indent=2));return
    old=load_json(output/'status.json')
    if old['status']!='paused':raise RuntimeError('Experiment is not paused; inspect before resuming')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    lock=output/'resume_active.lock'
    with lock.open('x') as f:f.write(str(os.getpid())+'\n')
    status=Status(output)
    status.state.update(previous_status='paused',resumed_at=status.state['started_at'],
        started_at=old['started_at'],optimizer_updates_disabled=False,background_work_finished=False,
        initial_collection_scenes=100,dagger_collection_scenes=100,aggregate_labels=len(prepared['data']['actions']),
        completed_supervised_before_resume=3000,resumed_supervised_updates=0,total_ppo_steps_completed=0,
        total_ppo_steps_planned=args.steps*2)
    paused=output/'status_paused_20261112.json'
    if not paused.exists():save(paused,old)
    names=['initial_teacher_labels.npz','dagger_labels.npz','frozen_test_cases.json','collection_complete.json']
    hashes={str(output/name):digest(output/name) for name in names}
    hashes.update({str(prepared['source']):digest(prepared['source']),
        str(ROOT/'data/train_cases.json'):digest(ROOT/'data/train_cases.json'),
        str(ROOT/'data/selection_cases.json'):digest(ROOT/'data/selection_cases.json'),
        str(ROOT/'data/validation_cases.json'):digest(ROOT/'data/validation_cases.json')})
    save(output/'resume_protocol_20261113.json',{'seed':args.seed,'input_hashes':hashes,
        'resume_script_sha256':digest(Path(__file__)),'prior_supervised_updates_not_repeated':3000,
        'remaining_aggregate_fit_updates':1500,'ppo_steps_per_arm':args.steps,'ppo_learning_rates':[3e-6,1e-5],
        'direct_policy_selection_only':True,'no_scene_generation_or_expert_collection':True})
    best_path=prepared['best'];best_result=prepared['best_result'];best_label=prepared['best_label']
    data=prepared['data'];train=prepared['train'];validation=prepared['validation']
    baseline=prepared['baseline'];source=prepared['source']
    try:
        status.update('resume_verified')
        np.savez_compressed(output/'aggregate_labels.npz',**data)
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            def select(path,result,label):
                nonlocal best_path,best_result,best_label
                if score(result)>score(best_result):best_path,best_result,best_label=path,result,label
                save(output/'selection.json',{'selected':best_label,'model':str(best_path),
                    'direct_policy_metrics':best_result['metrics'],'selection_uses_enhanced_execution':False})
            student=Supervised(best_path,data,1e-5,args.seed+1)
            for update in range(1,1501):
                loss=student.update()
                if update%100==0:
                    student.verify();status.update('dagger_fit',resumed_supervised_updates=update,
                        resumed_supervised_planned=1500,loss=loss)
                    with (output/'supervised_loss.csv').open('a') as f:f.write(f'dagger,{update},{loss}\n')
                if update in (500,1500):
                    path=output/'models'/f'dagger_{update}.zip';student.model.save(path)
                    select(path,evaluate(pool,path,validation,output,status,f'dagger_{update}'),f'dagger_{update}')
            ppo_start=best_path
            for arm,lr in enumerate((3e-6,1e-5)):
                env=DummyVecEnv([lambda:Monitor(CaseBankEnv(train,reward_scale=.01),
                    filename=str(output/f'arm{arm}_episodes'),info_keywords=('is_success','termination_reason'))])
                try:
                    env.seed(args.seed)
                    model=converted(ppo_start,env=env,n_steps=256,batch_size=64,n_epochs=3,learning_rate=lr,
                        clip_range=.05,target_kl=.005,gamma=.99,gae_lambda=.95,ent_coef=0.,vf_coef=.1,max_grad_norm=.5,
                        tensorboard_log=None,seed=args.seed)
                    model.policy.configure_learning(train_critic=True)
                    with torch.no_grad():model.policy.log_std.fill_(np.log(.05))
                    model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),lr=lr,**model.policy.optimizer_kwargs)
                    frozen={n:p.detach().clone() for n,p in model.policy.named_parameters() if not p.requires_grad}
                    model.ep_info_buffer=None;model.ep_success_buffer=None;model._episode_num=0
                    model.set_logger(configure(str(output/f'ppo_arm{arm}'),['stdout','csv']))
                    initial=model.num_timesteps
                    for target in range(8192,args.steps+1,8192):
                        status.update('ppo_training',ppo_arm=arm,learning_rate=lr,
                            arm_steps_completed=target-8192,arm_steps_planned=args.steps,
                            total_ppo_steps_completed=arm*args.steps+target-8192)
                        callback=AnchoredPPO(data,status,output/f'arm{arm}_anchor_updates.jsonl',args.seed+target)
                        model.learn(8192,reset_num_timesteps=False,callback=callback)
                        assert model.num_timesteps-initial==target
                        assert all(torch.equal(p,frozen[n]) for n,p in model.policy.named_parameters() if n in frozen)
                        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                        path=output/'models'/f'ppo_arm{arm}_{target}.zip';model.save(path)
                        status.update('ppo_checkpoint_saved',arm_steps_completed=target,
                            total_ppo_steps_completed=arm*args.steps+target)
                        label=f'ppo_arm{arm}_{target}'
                        select(path,evaluate(pool,path,validation,output,status,label),label)
                finally:env.close()
            shutil.copy2(best_path,output/'models/selected_model.zip')
            save(output/'frozen_selection.json',{'selected':best_label,
                'sha256':digest(output/'models/selected_model.zip'),'validation_metrics':best_result['metrics'],
                'fresh_test_not_yet_evaluated':True})
            old_result=evaluate(pool,output/'models/selected_model.zip',prepared['regression'],output,status,'selected_regression64',False)
            test_source=evaluate(pool,source,prepared['test'],output,status,'source_fresh_test',False)
            test_selected=evaluate(pool,output/'models/selected_model.zip',prepared['test'],output,status,'selected_fresh_test',False)
            for name,sha in hashes.items():assert digest(name)==sha,f'Input changed: {name}'
            save(output/'summary.json',{'selected':best_label,'validation':best_result['metrics'],
                'source_validation':baseline['metrics'],'regression64':old_result['metrics'],
                'fresh_test_source':test_source['metrics'],'fresh_test_selected':test_selected['metrics'],
                'pure_policy_execution':True,'total_ppo_steps':args.steps*2,'source_unchanged':True,
                'resumed_using_saved_labels':True,'aggregate_labels':len(data['actions'])})
            status.update('complete',status='completed',background_work_finished=True,
                          total_ppo_steps_completed=args.steps*2)
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--steps',type=int,default=32768);p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.steps<=0 or a.steps%8192 or a.workers<1:p.error('Invalid training budget')
    run(a)
if __name__=='__main__':main()
