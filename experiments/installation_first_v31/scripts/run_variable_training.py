"""BC warm start then genuine on-policy PPO on balanced 2--5 dynamic scenes.

Teacher actions are used only by a separate BC optimizer. PPO buffers contain
only sampled PPO actions and their original log probabilities. Validation cases
select checkpoints; the independent test bank is read only after selection.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback

import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from run_adaptation import Status, save, digest
from wall_panel_rl.submission import load_agent, load_json, load_runtime, run_episode, make_controller
from wall_panel_rl.rl.variable_weaving import VariableDynamicEnv, VariableCaseBankEnv
from wall_panel_rl.rl.known_route_policy import KnownRoutePolicy as StandalonePolicy
from wall_panel_rl.envs.known_route_motion import OBSERVATIONS
from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot, assert_planning_paused
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.envs.known_route_motion import upgrade_case
from wall_panel_rl.rl.timeout_diagnostics import control_frame,diagnose
from wall_panel_rl.visualization.episode_replay import recording_payload

CACHE = {}


def initialize(output=None):
    torch.set_num_threads(1)
    CACHE.update(model=load_agent(), runtime=load_runtime(ROOT/'configs/installation_runtime.json'), path=None,
                 training=yaml.safe_load((ROOT/'configs/variable_weaving.yaml').read_text())['training'],
                 teacher_output=str(Path(output)/'recordings/teachers') if output else None)


def worker_model(path):
    if CACHE.get('path') != str(path):
        CACHE.update(evaluated=load_agent(path), path=str(path))
    return CACHE['evaluated']


def collect_teacher(case):
    """Retry only training motion; keep failed attempts and accept success labels."""
    spec=CACHE['training'];attempt_rows=[]
    for attempt in range(spec.get('teacher_phase_retries',0)+1):
        selected=copy.deepcopy(case)
        route_seed=int(case['scene']['seed'])+31000000+attempt
        if attempt:
            selected=upgrade_case(selected,np.random.default_rng(route_seed),
                CACHE['runtime']['configuration']['obstacles']['dynamic']['speed_range_m_s'],preserve_conflict=False)
        row,x,y,w=teacher_attempt(selected,attempt,route_seed)
        attempt_rows.append(row)
        if row['accepted_for_bc']:
            row['attempts']=copy.deepcopy(attempt_rows)
            return row,x,y,w
    row['attempts']=copy.deepcopy(attempt_rows)
    return row,x,y,w


def teacher_attempt(case,attempt,route_seed):
    rt = CACHE['runtime']
    env = VariableDynamicEnv(rt['configuration'], [], rt['settings'])
    xs, ys, weights = [], [], []
    begin = time.perf_counter()
    try:
        obs, info = env.reset(options={'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']})
        frames=[control_frame(env,info)]
        path = env.cursor.path.copy()
        c = rt['controller']
        agent = make_controller(CACHE['model'],env,c)
        while True:
            before = planning_snapshot(env)
            action = agent.action(obs)
            assert_planning_paused(env, before)
            xs.append(obs.copy())
            spec=CACHE['training']
            weight=spec.get('multi_obstacle_teacher_weight',1.) if len(case['scene']['obstacles']) in (3,4) else 1.
            if info['position_error']<.12:weight*=spec.get('terminal_teacher_weight',1.)
            weights.append(weight)
            obs, _, done, truncated, info = env.step(action)
            ys.append(np.asarray(info['executed_action'], dtype=np.float32))
            frames.append(control_frame(env,info,action))
            if done or truncated:
                break
        assert np.array_equal(path, env.cursor.path)
        accepted = info['termination_reason'] == 'success'
        row = {'seed': case['scene']['seed'], 'count': len(case['scene']['obstacles']),
               'reason': info['termination_reason'], 'steps': env.env.steps, 'accepted_for_bc': accepted,
               'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
               'attempt':attempt,'route_seed':route_seed,
               'wall_seconds': time.perf_counter()-begin, 'uses_global_rrt': False}
        recording=recording_payload(rt['configuration'],case,row,frames)
        diagnosis=diagnose(recording);row['diagnosis']=diagnosis
        recording.update(timeout_diagnosis=diagnosis,execution_controller=rt['controller'])
        save(Path(CACHE['teacher_output'])/f"{case['scene']['seed']}_attempt{attempt}.json",recording)
        return (row,np.asarray(xs,np.float32) if accepted else np.empty((0,OBSERVATIONS),np.float32),
                np.asarray(ys,np.float32) if accepted else np.empty((0,7),np.float32),
                np.asarray(weights,np.float32) if accepted else np.empty(0,np.float32))
    finally:
        env.close()


def evaluate_one(job):
    path, case, mode, record_path, *variants = job
    runtime=CACHE['runtime']
    if variants and variants[0]=='v29_controller':
        runtime=copy.deepcopy(runtime)
        runtime['controller']=load_json(ROOT/'configs/v29_baseline_controller.json')
    row, recording = run_episode(worker_model(path), case, mode, runtime)
    row['execution_variant']=variants[0] if variants and variants[0] else 'installation_controller'
    row['obstacle_count'] = len(case['scene']['obstacles'])
    if record_path:
        save(record_path, recording)
    return row


def summary(rows):
    def group(items):
        result = {'episodes': len(items), **{reason: sum(r['reason']==reason for r in items)
                  for reason in ('success', 'collision', 'timeout', 'joint_limit', 'planning_failure')}}
        assert sum(result[k] for k in ('success', 'collision', 'timeout', 'joint_limit', 'planning_failure')) == len(items)
        result.update(success_rate=result['success']/max(1,len(items)),
            mean_passage_count=float(np.mean([r['passage_count'] for r in items])) if items else 0.,
            success_with_weaving=sum(r['reason']=='success' and r['passage_count']>=min(3,r['obstacle_count']) for r in items),
            mean_task_error=float(np.mean([r['position_m']+.12*np.deg2rad(r['orientation_deg']) for r in items])) if items else 0.)
        return result
    return {'all': group(rows), 'by_count': {str(n): group([r for r in rows if r['obstacle_count']==n]) for n in (2,3,4,5)}}


def evaluate(pool, path, cases, output, status, label, mode='enhanced', record=False,controller_variant=None):
    status.update('evaluate_'+label, evaluation_completed=0, evaluation_planned=len(cases))
    rows = []
    futures = [pool.submit(evaluate_one, (str(path), case, mode,
                str(output/'recordings'/label/f"{case['scene']['seed']}.json") if record else None,
                controller_variant)) for case in cases]
    for future in as_completed(futures):
        rows.append(future.result())
        result = {'model': str(path), 'mode': mode, 'complete': len(rows)==len(cases),
                  'metrics': summary(rows), 'rows': sorted(rows,key=lambda r:r['seed'])}
        save(output/'evaluations'/f'{label}.json', result)
        status.update('evaluate_'+label, evaluation_completed=len(rows), evaluation_planned=len(cases))
    return result


def score(result):
    m = result['metrics']['all']
    return (m['success'],-m['joint_limit']-m['collision'],-m['mean_task_error'])


class TrainingProgress(BaseCallback):
    def __init__(self, status, output, start, planned):
        super().__init__()
        self.status, self.output, self.start, self.planned = status, output, start, planned
        self.episodes = 0
        self.counts = {n: {k:0 for k in ('success','collision','timeout','joint_limit')} for n in (2,3,4,5)}

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                self.episodes += 1
                reason = info['termination_reason']
                self.counts[info['obstacle_count']][reason] += 1
                with (self.output/'training_episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps({'ppo_added_steps': self.num_timesteps-self.start,
                        'obstacle_count': info['obstacle_count'], 'reason': reason,
                        'position_m': float(info['position_error']), 'orientation_deg': float(np.rad2deg(info['orientation_error'])),
                        'vertical_crossings': info['vertical_crossings'], 'episode': info.get('episode')})+'\n')
        if self.n_calls % 128 == 0:
            self.status.update('ppo_training', ppo_steps_completed=self.num_timesteps-self.start,
                ppo_steps_planned=self.planned, ppo_episodes=self.episodes, training_counts=self.counts)
        return True


def make_training_env(runtime, cases, spec, seed, monitor_path):
    return Monitor(VariableCaseBankEnv(runtime, cases, spec, seed), monitor_path,
                   info_keywords=('is_success','termination_reason','obstacle_count','vertical_crossings'))


def plot_history(output, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=list(range(len(history)))
    fig, axes = plt.subplots(1, 2, figsize=(10,3.5))
    for count in (2,3,4,5):
        axes[0].plot(x, [r['metrics']['by_count'][str(count)]['success_rate'] for r in history], marker='o', label=f'{count} obstacles')
    axes[0].set(xlabel='Validation stage',ylabel='Enhanced installation success rate',ylim=(-.05,1.05))
    axes[0].legend(fontsize=8)
    for key in ('success','collision','timeout'):
        axes[1].plot(x, [r['metrics']['all'][key] for r in history], marker='o', label=key)
    axes[1].set(xlabel='Validation stage',ylabel='Validation episode count')
    for axis in axes:
        axis.set_xticks(x,[r['label'] for r in history],rotation=25,ha='right',fontsize=7)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'validation_curves.png', dpi=160)
    plt.close(fig)


def run(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a fresh training directory')
    out.mkdir(parents=True)
    (out/'models').mkdir()
    status = Status(out)
    torch.set_num_threads(1)
    spec = yaml.safe_load(args.config.read_text())
    s = spec['training']
    source = ROOT/'models/current_ppo.zip'
    source_hash = digest(source)
    runtime = load_runtime(ROOT/'configs/installation_runtime.json')
    # The test bank remains unread until all checkpoint selection has finished.
    train = load_json(args.scenes/'train_cases.json')
    validation = load_json(args.scenes/'validation_cases.json')
    assert not {c['scene']['seed'] for c in train} & {c['scene']['seed'] for c in validation}
    assert spec['counts']==[2,3,4,5] and spec['count_probabilities']==[.25]*4
    save(out/'protocol.json', {'settings':spec, 'source_model_sha256':source_hash,
        'train_cases':len(train), 'validation_cases':len(validation), 'test_unread_until_selection':True, 'formal_evaluation_mode':'enhanced_only',
        'observations':OBSERVATIONS, 'known_geometric_routes':True, 'future_random_schedule_exposed':False,
        'count_sampling':'uniform per episode, 25% each', 'all_scenes_dynamic':True,
        'bc_and_ppo_data_separated':True, 'ppo_rollouts_use_raw_sampled_actions':True,
        'source_files':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src').rglob('*.py')},
        'train_bank_sha256':digest(args.scenes/'train_cases.json'),
        'validation_bank_sha256':digest(args.scenes/'validation_cases.json')})
    history = []
    try:
        with ProcessPoolExecutor(spec['workers'],mp_context=mp.get_context('spawn'),
                                 initializer=initialize,initargs=(str(out),)) as pool:
            restored=evaluate(pool,source,validation,out,status,'v29_controller_on_closer',
                     controller_variant='v29_controller',record=True)
            baseline = evaluate(pool, source, validation, out, status, 'source_enhanced', record=True)
            best_path, best_result, best_label = source, baseline, 'source_enhanced'
            best_variant=None
            if score(restored)>=score(best_result):
                best_result,best_label,best_variant=restored,'v29_controller_on_closer','v29_controller'
            initial_result=copy.deepcopy(best_result)
            history=[{'label':'v29_controller','added_steps':0,'metrics':restored['metrics']},
                     {'label':'installation_init','added_steps':0,'metrics':baseline['metrics']}]
            save(out/'checkpoint_history.json',history);plot_history(out,history)
            chosen = []
            for count in spec['counts']:
                quotas=s.get('teacher_cases_by_count',{})
                quota=quotas.get(str(count),quotas.get(count,s['teacher_cases_per_count']))
                chosen.extend([c for c in train if len(c['scene']['obstacles'])==count][:quota])
            futures = [pool.submit(collect_teacher, case) for case in chosen]
            xs, ys, ws, logs = [], [], [], []
            status.update('collect_bc_teachers', teacher_completed=0, teacher_planned=len(chosen))
            for future in as_completed(futures):
                row, x, y, w = future.result()
                logs.append(row)
                xs.append(x); ys.append(y);ws.append(w)
                save(out/'teacher_episodes.json', logs)
                status.update('collect_bc_teachers', teacher_completed=len(logs), teacher_planned=len(chosen),
                    successful_teacher_episodes=sum(r['accepted_for_bc'] for r in logs), teacher_transitions=sum(len(a) for a in xs))
            x, y = np.concatenate(xs), np.concatenate(ys)
            weights=np.concatenate(ws)
            np.savez_compressed(out/'teacher_labels.npz',observations=x,actions=y,sampling_weights=weights)
            bc_updates_performed=0
            if len(x) >= 64:
                model = PPO.load(source, device='cpu', custom_objects={'policy_class':StandalonePolicy})
                model.policy.configure_learning(train_critic=False)
                parameters = [p for p in model.policy.parameters() if p.requires_grad]
                optimizer = torch.optim.Adam(parameters, lr=s['bc_learning_rate'])
                rng = np.random.default_rng(spec['seed'])
                probabilities=weights.astype(np.float64);probabilities/=probabilities.sum()
                for update in range(1, s['bc_updates']+1):
                    indices=rng.choice(len(x),size=s['batch_size'],replace=True,p=probabilities)
                    obs = torch.as_tensor(x[indices]); target = torch.as_tensor(y[indices])
                    prediction = model.policy.imitation_mean(obs)
                    loss = (prediction-target).square().mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite BC loss')
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters,.5); optimizer.step()
                    bc_updates_performed=update
                    if update % 100 == 0:
                        status.update('bc_warm_start', bc_updates=update, bc_updates_planned=s['bc_updates'], bc_loss=float(loss.detach()))
                    if update in (250, 500, s['bc_updates']):
                        path = out/'models'/f'bc_{update}.zip'
                        model.save(path)
                        result=evaluate(pool,path,validation,out,status,f'bc_{update}',record=True)
                        history.append({'label':f'bc_{update}','added_steps':0,'metrics':result['metrics']})
                        save(out/'checkpoint_history.json',history);plot_history(out,history)
                        if score(result) > score(best_result):
                            best_path, best_result, best_label = path, result, f'bc_{update}'
                            best_variant=None
            save(out/'warm_start_selection.json', {'model':str(best_path), 'label':best_label,
                 'metrics':best_result['metrics'],'successful_teacher_transitions':len(x),
                 'controller_variant':best_variant})
            ppo_ran=score(best_result)>score(initial_result)
            save(out/'bc_gate.json',{'passed':ppo_ran,'criterion':'validation success, then collisions, then task error',
                'initial_metrics':initial_result['metrics'],'selected_metrics':best_result['metrics'],
                'ppo_steps_if_passed':s['steps'],'no_test_outcomes_used':True})
            if ppo_ran:
                env = SubprocVecEnv([partial(make_training_env, runtime, train, s, spec['seed']+rank*1000,
                                str(out/f'worker{rank}_episodes')) for rank in range(spec['workers'])], start_method='spawn')
                try:
                    env.seed(spec['seed'])
                    model = PPO.load(best_path, env=env, device='cpu', custom_objects={'policy_class':StandalonePolicy},
                        n_steps=s['n_steps'], batch_size=s['batch_size'], n_epochs=s['n_epochs'],
                        learning_rate=s['learning_rate'], clip_range=s['clip_range'], target_kl=s['target_kl'],
                        gamma=.99, gae_lambda=.95, ent_coef=0., vf_coef=.1, max_grad_norm=.5,
                        tensorboard_log=None, seed=spec['seed'])
                    model.policy.configure_learning(train_critic=True)
                    with torch.no_grad(): model.policy.log_std.fill_(np.log(s['action_std']))
                    model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(), lr=s['learning_rate'], **model.policy.optimizer_kwargs)
                    frozen = {n:p.detach().clone() for n,p in model.policy.named_parameters() if not p.requires_grad}
                    model.ep_info_buffer=None; model.ep_success_buffer=None; model._episode_num=0
                    model.set_logger(configure(str(out/'ppo'), ['stdout','csv']))
                    start = model.num_timesteps
                    updates = model._n_updates
                    callback = TrainingProgress(status, out, start, s['steps'])
                    for added in range(s['checkpoint_interval'], s['steps']+1, s['checkpoint_interval']):
                        status.update('ppo_training', ppo_steps_completed=added-s['checkpoint_interval'], ppo_steps_planned=s['steps'])
                        model.learn(s['checkpoint_interval'], reset_num_timesteps=False, callback=callback)
                        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                        assert all(torch.equal(dict(model.policy.named_parameters())[n], p) for n,p in frozen.items())
                        assert model.num_timesteps-start==added
                        assert model._n_updates>updates
                        path = out/'models'/f'ppo_{added}.zip'
                        model.save(path)
                        result = evaluate(pool, path, validation, out, status, f'ppo_{added}',record=True)
                        history.append({'label':f'ppo_{added}', 'added_steps':added, 'metrics':result['metrics']})
                        save(out/'checkpoint_history.json', history)
                        plot_history(out, history)
                        if score(result)>score(best_result):
                            best_path, best_result, best_label = path, result, f'ppo_{added}'
                            best_variant=None
                        save(out/'selection.json', {'label':best_label, 'model':str(best_path), 'metrics':best_result['metrics'],
                             'selection_uses_validation_only':True,'completed_ppo_steps':added,
                             'controller_variant':best_variant})
                        if added==s['pilot_steps']:
                            save(out/'pilot_check.json', {'passed':True, 'additional_ppo_steps':added,
                                 'finite_parameters':True, 'optimizer_updated':True, 'frozen_tracking_prior_unchanged':True,
                                 'training_counts':callback.counts, 'convergence_claim':False})
                    model.save(out/'models/final_ppo.zip')
                    save(out/'training_summary.json', {'additional_ppo_steps':model.num_timesteps-start,
                        'optimizer_epoch_updates':model._n_updates-updates, 'ppo_episodes':callback.episodes,
                        'training_counts':callback.counts})
                finally:
                    env.close()
            else:
                status.update('ppo_skipped_no_bc_improvement',ppo_steps_completed=0,ppo_steps_planned=s['steps'])
                save(out/'training_summary.json',{'additional_ppo_steps':0,'optimizer_epoch_updates':0,
                    'ppo_episodes':0,'ppo_skipped':True,'reason':'BC did not improve the best initialization validation score'})
                save(out/'selection.json',{'label':best_label,'model':str(best_path),'metrics':best_result['metrics'],
                    'selection_uses_validation_only':True,'completed_ppo_steps':0,'controller_variant':best_variant})
            # Freeze the selected model before opening the independent test bank.
            import shutil
            shutil.copy2(best_path, out/'models/selected_model.zip')
            test = load_json(args.scenes/'test_cases.json')
            assert not {c['scene']['seed'] for c in test} & {c['scene']['seed'] for c in train+validation}
            selected = out/'models/selected_model.zip'
            selected_runtime=copy.deepcopy(runtime)
            if best_variant=='v29_controller':selected_runtime['controller']=load_json(ROOT/'configs/v29_baseline_controller.json')
            save(out/'selected_runtime.json',selected_runtime)
            enhanced=evaluate(pool,selected,test,out,status,'independent_test_enhanced',
                              mode='enhanced',record=True,controller_variant=best_variant)
            assert digest(source)==source_hash
            save(out/'summary.json', {'selected_label':best_label, 'validation':best_result['metrics'],
                'evaluation_mode':'enhanced_only', 'independent_test_enhanced':enhanced['metrics'],
                'training_performed':bc_updates_performed>0 or ppo_ran,'source_model_unchanged':True,
                'controller_variant':best_variant,'bc_updates_performed':bc_updates_performed,
                'ppo_started':ppo_ran})
            status.update('complete',status='completed',ppo_steps_completed=s['steps'] if ppo_ran else 0,ppo_steps_planned=s['steps'],ppo_skipped=not ppo_ran)
    except BaseException as exc:
        status.update('failed', status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/variable_weaving.yaml')
    parser.add_argument('--scenes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
