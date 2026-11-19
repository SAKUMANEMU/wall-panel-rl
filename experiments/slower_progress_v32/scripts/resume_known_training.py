"""Resume a completed PPO update checkpoint after paused processes are lost.

Weights, optimizer and cumulative counters resume from disk. Physical rollout
environments and their RNG streams restart explicitly; this is not bitwise
continuation of the lost in-memory episode. Completed evaluations are reused.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import fcntl
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import traceback
import zipfile

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from run_adaptation import Status, save, digest
from run_variable_training import (initialize, evaluate_one, summary, score,
    TrainingProgress, make_training_env, plot_history)
from wall_panel_rl.submission import load_json, load_runtime, load_agent
from wall_panel_rl.utils.config import ROOT


def checkpoint_data(path):
    with zipfile.ZipFile(path) as archive:
        if 'policy.optimizer.pth' not in archive.namelist():
            raise ValueError('The checkpoint has no saved PPO optimizer')
        return json.loads(archive.read('data'))


def resume_preflight(output, checkpoint):
    """Read-only checks; the frozen run specification overrides project defaults."""
    protocol = load_json(output/'protocol.json')
    for name, sha in protocol['source_files'].items():
        if digest(ROOT/name) != sha:
            raise ValueError('Original training source changed: '+name)
    scenes = output.parent/'scenes'
    for split in ('train','validation'):
        if digest(scenes/f'{split}_cases.json') != protocol[f'{split}_bank_sha256']:
            raise ValueError('A frozen case bank changed: '+split)
    source = ROOT/'models/current_ppo.zip'
    if digest(source) != protocol['source_model_sha256']:
        raise ValueError('Original source model changed')
    spec = copy.deepcopy(protocol['settings'])
    frozen_config = output.parent/'training_config.yaml'
    if frozen_config.exists() and yaml.safe_load(frozen_config.read_text()) != spec:
        raise ValueError('Frozen training settings changed')
    checkpoint = checkpoint.resolve()
    saved, initial = checkpoint_data(checkpoint), checkpoint_data(source)
    completed = int(saved['num_timesteps'])-int(initial['num_timesteps'])
    interval = spec['training']['checkpoint_interval']
    if checkpoint != (output/'models'/f'ppo_{completed}.zip').resolve():
        raise ValueError('Checkpoint must belong to this run')
    if not 0 < completed <= spec['training']['steps'] or completed % interval:
        raise ValueError('Resume requires a completed original PPO update boundary')
    frozen_runtime = output.parent/'runtime.json'
    runtime = load_json(frozen_runtime) if frozen_runtime.exists() else load_runtime()
    return protocol, spec, runtime, saved, initial, completed


def initialize_resume(runtime):
    """Evaluation workers use the same resolved runtime saved before reboot."""
    from run_variable_training import CACHE
    torch.set_num_threads(1)
    CACHE.update(model=load_agent(), runtime=copy.deepcopy(runtime), path=None)


def reject_original_processes(output):
    """Avoid a second optimizer while the original stopped run still exists."""
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():
            continue
        try:
            command = (directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if str(output.parent) in command and any(
            f'/scripts/{name}' in command for name in ('run_site_pipeline.py','run_variable_training.py')):
            raise RuntimeError('Original pipeline still exists. Resume it in memory or reboot before checkpoint recovery.')


def cached_evaluate(pool, path, cases, output, status, label, record=False):
    """Keep finished rows and submit only missing scene seeds for this model."""
    destination = output/'evaluations'/f'{label}.json'
    rows = []
    expected = {int(c['scene']['seed']) for c in cases}
    model_hash = digest(path)
    if destination.exists():
        old = load_json(destination)
        if Path(old['model']).resolve() != path.resolve() or old['mode'] != 'enhanced':
            raise ValueError('Cached evaluation has a different model or execution mode')
        if old.get('model_sha256', model_hash) != model_hash:
            raise ValueError('Cached evaluation checkpoint hash differs')
        rows = list(old['rows'])
        ids = [int(r['seed']) for r in rows]
        if len(ids) != len(set(ids)) or not set(ids) <= expected:
            raise ValueError('Invalid cached evaluation scene identifiers')
    done = {int(r['seed']) for r in rows}
    status.update('evaluate_'+label, evaluation_completed=len(rows), evaluation_planned=len(cases),
                  reused_evaluation_rows=len(rows))
    futures = []
    for case in cases:
        if int(case['scene']['seed']) in done:
            continue
        target = str(output/'recordings'/label/f"{case['scene']['seed']}.json") if record else None
        futures.append(pool.submit(evaluate_one, (str(path), case, 'enhanced', target)))
    def write():
        result = {'model':str(path), 'model_sha256':model_hash, 'mode':'enhanced',
                  'complete':len(rows)==len(cases), 'metrics':summary(rows),
                  'rows':sorted(rows,key=lambda r:r['seed'])}
        save(destination,result)
        return result
    for future in as_completed(futures):
        row = future.result()
        if int(row['seed']) not in expected or int(row['seed']) in done:
            raise ValueError('Unexpected or duplicate completed evaluation row')
        rows.append(row);done.add(int(row['seed']))
        write()
        status.update('evaluate_'+label,evaluation_completed=len(rows),evaluation_planned=len(cases))
    return write()


def run(args):
    output = args.output.resolve()
    if args.check_only:
        torch.set_num_threads(1)
        _,spec,_,saved,_,completed = resume_preflight(output,args.checkpoint)
        model = load_agent(args.checkpoint)
        assert model.num_timesteps == saved['num_timesteps']
        assert model._n_updates == saved['_n_updates']
        assert model.policy.optimizer.state, 'Saved optimizer state not restored'
        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
        print(json.dumps({'check_only':True,'training_started':False,
            'checkpoint':str(args.checkpoint.resolve()),'completed_ppo_steps':completed,
            'remaining_ppo_steps':spec['training']['steps']-completed,
            'optimizer_restored':True,'finite_parameters':True,'frozen_settings_verified':True}),flush=True)
        return
    reject_original_processes(output)
    with (output/'resume.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        _run(args,output)


def _run(args,output):
    torch.set_num_threads(1)
    original_status = load_json(output/'status.json')
    protocol,spec,runtime,saved,initial,completed = resume_preflight(output,args.checkpoint)
    scenes = output.parent/'scenes'
    for split in ('train','validation'):
        if digest(scenes/f'{split}_cases.json') != protocol[f'{split}_bank_sha256']:
            raise ValueError('A frozen case bank changed: '+split)
    source = ROOT/'models/current_ppo.zip'
    assert digest(source) == protocol['source_model_sha256']
    settings = spec['training']
    checkpoint = args.checkpoint.resolve()
    saved = checkpoint_data(checkpoint)
    initial = checkpoint_data(source)
    base_steps = int(initial['num_timesteps'])
    completed = int(saved['num_timesteps'])-base_steps
    interval = settings['checkpoint_interval']
    if not 0 < completed <= settings['steps'] or completed % interval:
        raise ValueError('Resume requires a completed original PPO update boundary')
    stamp = datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
    status = Status(output)
    status.state.update({k:v for k,v in original_status.items()
                         if k not in ('status','pid','updated_at','paused_at','pause_method')})
    status.state.update(status='running',pid=os.getpid(),resumed_at=datetime.now().astimezone().isoformat())
    def pipeline(state,phase,**extra):
        save(output.parent/'pipeline_status.json',{'status':state,'phase':phase,'pid':os.getpid(),
             'child_pid':os.getpid(),'updated_at':datetime.now().astimezone().isoformat(),**extra})
    pipeline('running','resume_train_and_evaluate',resumed_from_ppo_steps=completed)
    status.update('resume_checkpoint',ppo_steps_completed=completed,ppo_steps_planned=settings['steps'])
    resume_record = {'resumed_at':status.state['resumed_at'],'pid':os.getpid(),'checkpoint':str(checkpoint),
        'checkpoint_sha256':digest(checkpoint),'completed_ppo_steps':completed,
        'remaining_ppo_steps':settings['steps']-completed,'optimizer_restored':True,
        'environment_rng_reinitialized':True,'bitwise_continuation':False,
        'training_seed_after_resume':spec['seed']+completed,'original_protocol_preserved':True}
    save(output/f'resume_{stamp}.json',resume_record)
    try:
        train = load_json(scenes/'train_cases.json')
        validation = load_json(scenes/'validation_cases.json')
        warm = load_json(output/'warm_start_selection.json')
        best_path = Path(warm['model'])
        best_label = warm['label']
        best_result = load_json(output/'evaluations'/f'{best_label}.json')
        history = [{'label':best_label,'added_steps':0,'metrics':best_result['metrics']}]
        with ProcessPoolExecutor(spec['workers'],mp_context=mp.get_context('spawn'),
                                 initializer=initialize_resume,initargs=(runtime,)) as pool:
            # Reconstruct completed selection events, completing a partial one.
            for step in range(interval,completed+1,interval):
                path = output/'models'/f'ppo_{step}.zip'
                result = cached_evaluate(pool,path,validation,output,status,f'ppo_{step}')
                history.append({'label':f'ppo_{step}','added_steps':step,'metrics':result['metrics']})
                if score(result)>score(best_result):
                    best_path,best_result,best_label = path,result,f'ppo_{step}'
            save(output/'checkpoint_history.json',history)
            plot_history(output,history)
            save(output/'selection.json',{'label':best_label,'model':str(best_path),'metrics':best_result['metrics'],
                 'selection_uses_validation_only':True,'completed_ppo_steps':completed})
            if completed >= settings['pilot_steps'] and not (output/'pilot_check.json').exists():
                save(output/'pilot_check.json',{'passed':True,'additional_ppo_steps':settings['pilot_steps'],
                    'finite_parameters':True,'optimizer_updated':True,'frozen_tracking_prior_unchanged':True,
                    'training_counts':original_status['training_counts'],'convergence_claim':False})
            env = SubprocVecEnv([partial(make_training_env,runtime,train,settings,
                    spec['seed']+completed+rank*1000,str(output/f'worker{rank}_episodes_resume_{stamp}'))
                    for rank in range(spec['workers'])],start_method='spawn')
            try:
                env.seed(spec['seed']+completed)
                # PPO.load restores policy.optimizer.pth. Do not recreate Adam.
                model = PPO.load(checkpoint,env=env,device='cpu',tensorboard_log=None,
                                 seed=spec['seed']+completed)
                model.policy.configure_learning(train_critic=True)
                assert model.num_timesteps==saved['num_timesteps']
                assert model._n_updates==saved['_n_updates']
                assert model.policy.optimizer.state, 'Saved optimizer state not restored'
                frozen = {n:p.detach().clone() for n,p in model.policy.named_parameters() if not p.requires_grad}
                model.set_logger(configure(str(output/f'ppo_resume_{stamp}'),['stdout','csv']))
                callback = TrainingProgress(status,output,base_steps,settings['steps'])
                callback.episodes = int(original_status.get('ppo_episodes',0))
                callback.counts = {int(k):copy.deepcopy(v) for k,v in original_status['training_counts'].items()}
                for added in range(completed+interval,settings['steps']+1,interval):
                    status.update('ppo_training',ppo_steps_completed=added-interval,ppo_steps_planned=settings['steps'])
                    model.learn(interval,reset_num_timesteps=False,callback=callback)
                    assert model.num_timesteps-base_steps==added
                    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                    assert all(torch.equal(dict(model.policy.named_parameters())[n],p) for n,p in frozen.items())
                    path = output/'models'/f'ppo_{added}.zip'
                    model.save(path)
                    result = cached_evaluate(pool,path,validation,output,status,f'ppo_{added}')
                    history.append({'label':f'ppo_{added}','added_steps':added,'metrics':result['metrics']})
                    save(output/'checkpoint_history.json',history);plot_history(output,history)
                    if score(result)>score(best_result):
                        best_path,best_result,best_label = path,result,f'ppo_{added}'
                    save(output/'selection.json',{'label':best_label,'model':str(best_path),'metrics':best_result['metrics'],
                        'selection_uses_validation_only':True,'completed_ppo_steps':added})
                model.save(output/'models/final_ppo.zip')
                save(output/'training_summary.json',{'additional_ppo_steps':model.num_timesteps-base_steps,
                    'optimizer_epoch_updates':model._n_updates-initial['_n_updates'],
                    'optimizer_epoch_updates_after_resume':model._n_updates-saved['_n_updates'],
                    'ppo_episodes':callback.episodes,'training_counts':callback.counts,
                    'resumed_from_ppo_steps':completed,'environment_rng_reinitialized':True})
            finally:
                env.close()
            selected = output/'models/selected_model.zip'
            shutil.copy2(best_path,selected)
            test = load_json(scenes/'test_cases.json')
            assert not {c['scene']['seed'] for c in test}&{c['scene']['seed'] for c in train+validation}
            result = cached_evaluate(pool,selected,test,output,status,'independent_test_enhanced',record=True)
            assert digest(source)==protocol['source_model_sha256']
            save(output/'summary.json',{'selected_label':best_label,'validation':best_result['metrics'],
                'evaluation_mode':'enhanced_only','independent_test_enhanced':result['metrics'],
                'training_performed':True,'source_model_unchanged':True,'resumed_from_ppo_steps':completed})
            status.update('complete',status='completed',ppo_steps_completed=settings['steps'],ppo_steps_planned=settings['steps'])
            pipeline('completed','complete')
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc())
        pipeline('failed','failed',error=repr(exc))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--check-only',action='store_true',help='Verify disk recovery without starting training or evaluation')
    run(parser.parse_args())


if __name__=='__main__':
    main()
