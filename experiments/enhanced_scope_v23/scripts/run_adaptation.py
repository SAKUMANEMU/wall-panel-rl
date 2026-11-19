"""Auditable dynamic-scene BC -> short PPO study, isolated from submission files.

All labels come from physically successful training episodes. Evaluation cases
never enter either optimizer. Original-reference observations are retained.
"""
import _bootstrap
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import yaml
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.submission import CaseBankEnv, load_agent, load_json, load_runtime, run_episode
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv, planning_snapshot, assert_planning_paused
from wall_panel_rl.rl.episode_imitation import make_rrt
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
from wall_panel_rl.rl.vertical_corridor import fresh_vertical_corridor_case

WORK = {}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    """Atomic files allow a status reader to run during long physics trials."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class Status:
    def __init__(self, output):
        self.output = Path(output)
        self.state = {'status': 'running', 'pid': __import__('os').getpid(),
                      'started_at': datetime.now().astimezone().isoformat()}

    def update(self, phase, **extra):
        self.state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        save(self.output/'status.json', self.state)
        print(json.dumps(self.state), flush=True)


def initialize_worker():
    torch.set_num_threads(1)
    WORK.update(model=load_agent(), runtime=load_runtime(),
                profile=yaml.safe_load((ROOT/'configs/scene.yaml').read_text()))


def generate_one(seed):
    try:
        case = fresh_vertical_corridor_case(WORK['runtime']['configuration'],
            WORK['runtime']['settings'], WORK['profile'], seed, WORK['model'],
            WORK['runtime']['controller'])
        return {'seed': seed, 'geometry_valid': True, 'case': case}
    except (RuntimeError, ValueError) as exc:
        return {'seed': seed, 'geometry_valid': False, 'reason': str(exc)}


def teacher_episode(case, mode):
    """Store pre-action observations and the shared actuator's executed targets."""
    rt = WORK['runtime']; env = SlowDynamicFourEnv(rt['configuration'], [], rt['settings'])
    observations, actions, diagnostics = [], [], []
    start = time.perf_counter()
    try:
        obs, info = env.reset(options={'scenario': case['scene'], 'task': case['task'],
                                       'strength': case['strength']})
        original_reference = env.cursor.path.copy()
        paused = planning_snapshot(env)
        expert = make_rrt(env, int(case['scene']['seed'])+100000) if mode == 'rrt' else None
        controller = None
        if mode == 'enhanced':
            c = rt['controller']
            controller = ProgressRecoveryController(WORK['model'], env, c['selected']['horizon'],
                                                     c['gains'], c['recovery'])
        assert_planning_paused(env, paused)
        reason = 'planning_failure' if expert is not None and not expert.planning_success else None
        while reason is None:
            previous_obs = obs.copy(); q, _ = env.env.joint_state()
            reference_index = env.cursor.index
            paused = planning_snapshot(env)
            command = expert.action(obs) if expert else controller.action(obs)
            assert_planning_paused(env, paused)
            obs, reward, done, truncated, info = env.step(command)
            target = np.asarray(info['executed_action'], dtype=np.float32)
            observations.append(previous_obs); actions.append(target)
            new_q, _ = env.env.joint_state()
            diagnostics.append({'step': env.env.steps, 'reference_index': reference_index,
                'next_reference_index': env.cursor.index,
                'reference_distance_rad': float(np.linalg.norm(env.ref-new_q)),
                'zero_command': bool(np.max(np.abs(target)) < 1e-8),
                'stationary': bool(np.max(np.abs(q-new_q)) < 1e-4),
                'clearance_m': float(info['obstacle_clearance_capped_m']),
                'position_m': float(info['position_error']),
                'orientation_deg': float(np.rad2deg(info['orientation_error']))})
            if done or truncated:
                reason = info['termination_reason']
        assert np.array_equal(original_reference, env.cursor.path)
        row = {'seed': int(case['scene']['seed']), 'controller': mode, 'reason': reason,
            'steps': env.env.steps, 'position_m': float(info['position_error']),
            'orientation_deg': float(np.rad2deg(info['orientation_error'])),
            'wall_seconds': time.perf_counter()-start,
            'accepted_for_imitation': reason == 'success',
            'reference_preserved': True, 'diagnostics': diagnostics,
            'recovery_events': controller.recovery_events if controller else []}
        return row, np.asarray(observations, dtype=np.float32).reshape(-1,77), np.asarray(actions,dtype=np.float32).reshape(-1,7)
    finally:
        env.close()


def collect_one(case):
    attempts = []
    for mode in ('rrt', 'enhanced'):
        row, x, y = teacher_episode(case, mode)
        attempts.append(row)
        if row['accepted_for_imitation']:
            return attempts, x, y
    return attempts, np.empty((0,77),np.float32), np.empty((0,7),np.float32)


def evaluate_one(job):
    model_path, case, mode, early = job
    key = str(model_path)
    if WORK.get('loaded_path') != key:
        WORK['evaluated_model'] = load_agent(model_path); WORK['loaded_path'] = key
    rt = copy.deepcopy(WORK['runtime'])
    if early:
        # A separate controller ablation; never changes PPO actions or task rules.
        rt['controller']['recovery'].update(minimum_step=60, window=16, cooldown=16,
                                            maximum_rescues=5, commit_steps=6)
    row, recording = run_episode(WORK['evaluated_model'], case, mode, rt)
    row.update(controller_early_recovery=early,
               waiting_fraction=row['zero_command_steps']/max(row['steps'],1),
               stationary_fraction=row['stationary_steps']/max(row['steps'],1))
    return row


def jobs(pool, function, items, status, phase):
    """Return in submission order; execution completion order never selects cases."""
    futures = {pool.submit(function, item): i for i,item in enumerate(items)}
    results = [None]*len(items); completed = 0
    while futures:
        ready, _ = wait(futures, timeout=15, return_when=FIRST_COMPLETED)
        for future in ready:
            results[futures.pop(future)] = future.result(); completed += 1
        status.update(phase, batch_completed=completed, batch_total=len(items))
    return results


def metrics(rows):
    result = {'episodes':len(rows)}
    for reason in ('success','collision','timeout','joint_limit','planning_failure'):
        result[reason] = sum(r['reason']==reason for r in rows)
    assert sum(result[k] for k in ('success','collision','timeout','joint_limit','planning_failure'))==len(rows)
    result['success_rate'] = result['success']/len(rows)
    result['mean_waiting_fraction'] = float(np.mean([r['waiting_fraction'] for r in rows]))
    result['mean_stationary_fraction'] = float(np.mean([r['stationary_fraction'] for r in rows]))
    timed = [r for r in rows if r['reason']=='timeout']
    result['timeout_mean_position_m'] = float(np.mean([r['position_m'] for r in timed])) if timed else None
    result['timeout_mean_orientation_deg'] = float(np.mean([r['orientation_deg'] for r in timed])) if timed else None
    return result


def selection_score(evaluation):
    """Reject safety regressions before comparing success and waiting."""
    p = evaluation['ppo']['metrics']; e = evaluation['enhanced']['metrics']
    return (-(e['joint_limit']+e['collision']), e['success'], p['success'],
            -(p['joint_limit']+p['collision']), -e['mean_waiting_fraction'])


def evaluate(pool, path, bank, output, status, label, early=False):
    destination = output/'evaluations'/f'{label}.json'
    if destination.exists(): return load_json(destination)
    result = {'model':str(path), 'model_sha256':digest(path), 'selection_only':label!='regression64',
              'controller_early_recovery':early}
    for mode in ('ppo','enhanced'):
        status.update(f'evaluate_{label}_{mode}', evaluation_model=label)
        rows = jobs(pool, evaluate_one, [(str(path),c,mode,early) for c in bank],
                    status, f'evaluate_{label}_{mode}')
        result[mode] = {'metrics':metrics(rows),'rows':rows}
        save(destination, result)
    return result


def bc_step(policy, optimizer, x, y):
    prediction = policy.tracking_mean(x)
    loss = (prediction-y).square().mean()
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite BC loss')
    optimizer.zero_grad(); loss.backward()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    torch.nn.utils.clip_grad_norm_(parameters, .5); optimizer.step()
    return float(loss.detach())


def train_bc(source, labels, output, seed, status, updates=2000):
    model = load_agent(source); model.policy.configure_learning(train_critic=False)
    frozen = {k:v.detach().clone() for k,v in model.policy.state_dict().items() if not k.startswith('multi_adapter.')}
    optimizer = torch.optim.Adam([p for p in model.policy.parameters() if p.requires_grad],lr=1e-4)
    x = torch.as_tensor(labels['observations']); y = torch.as_tensor(labels['actions'])
    scene_ids = labels['scene_ids']; unique = np.unique(scene_ids)
    indices = {s:np.flatnonzero(scene_ids==s) for s in unique}
    rng = np.random.default_rng(seed); losses=[]
    # Every successful scene gets equal sampling probability. Keep valid waiting
    # labels; do not teach unsafe motion by deleting zero commands.
    for update in range(1,updates+1):
        chosen=rng.choice(unique,256); batch=np.array([rng.choice(indices[s]) for s in chosen])
        losses.append(bc_step(model.policy,optimizer,x[batch],y[batch]))
        if update%100==0 or update==updates:
            assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in frozen.items())
            assert all(torch.isfinite(p).all() for p in model.policy.parameters())
            save(output/'bc_progress.json',{'updates':update,'planned_updates':updates,
                'mean_recent_loss':float(np.mean(losses[-100:])), 'frozen_parameters_unchanged':True})
            with (output/'bc_loss.csv').open('a') as log:
                if log.tell()==0: log.write('update,mean_recent_loss\n')
                log.write(f'{update},{float(np.mean(losses[-100:]))}\n')
            status.update('bc_training',bc_updates=update,bc_updates_planned=updates)
    # PPO starts with fresh Adam moments after a supervised optimization stage.
    model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),lr=3e-6,**model.policy.optimizer_kwargs)
    model.save(output/'models/bc_model.zip')
    save(output/'bc_summary.json',{'updates':updates,'transitions':len(x),'successful_scenes':len(unique),
        'initial_loss':float(np.mean(losses[:100])),'final_loss':float(np.mean(losses[-100:])),
        'prior_critic_std_frozen':True,'ppo_steps_added':0,'new_ppo_optimizer':True})
    return output/'models/bc_model.zip'


def run(args):
    torch.set_num_threads(1); output=args.output.resolve()
    if output.exists(): raise FileExistsError('Choose a new experiment output directory')
    output.mkdir(parents=True);(output/'models').mkdir()
    status=Status(output); source=ROOT/'models/current_ppo.zip'; source_hash=digest(source)
    protocol={'seed':args.seed,'train_scenes':args.train_scenes,'validation_scenes':args.validation_scenes,
        'ppo_additional_steps':args.steps,'workers':args.workers,'source_model_sha256':source_hash,
        'scene_start_seed':args.start_seed,'scene_attempt_budget':args.max_attempts,
        'bc_updates':args.bc_updates,'teacher_rule':'RRT first; enhanced teacher if that episode fails. All attempts logged; only complete successes become BC labels.',
        'selection_rule':'enhanced collision/limit count first, then enhanced success, direct PPO success, direct failures and enhanced waiting fraction; source retained on exact ties',
        'final_test_performed':False,'reward_or_success_tolerance_changed':False,
        'code_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src').rglob('*.py'))}}
    protocol['code_hashes']['scripts/run_adaptation.py']=digest(Path(__file__))
    save(output/'protocol.json',protocol)
    try:
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            split=load_json(ROOT/'data/split.json');excluded=set(split['train']+split['validation']+split['demo'])
            bank=[];attempts=[]; next_seed=args.start_seed
            goal=args.train_scenes+args.validation_scenes
            while len(bank)<goal and len(attempts)<args.max_attempts:
                seeds=[]
                while len(seeds)<min(args.workers*2,args.max_attempts-len(attempts)):
                    seed=next_seed;next_seed+=1
                    if seed not in excluded: seeds.append(seed)
                status.update('generating_scenes',valid_scenes=len(bank),target_scenes=goal,attempts=len(attempts))
                batch=jobs(pool,generate_one,seeds,status,'generating_scenes')
                for item in batch:
                    case=item.pop('case',None);attempts.append(item)
                    if case is not None:bank.append(case)
                save(output/'scene_attempts.json',attempts);save(output/'all_generated_cases.json',bank)
            if len(bank)<goal: raise RuntimeError('Scene generation budget exhausted; all partial data retained')
            train=bank[:args.train_scenes]; validation=bank[args.train_scenes:goal]
            save(output/'train_cases.json',train);save(output/'validation_cases.json',validation)
            status.update('scenes_ready',valid_scenes=goal,training_scenes=len(train),validation_scenes=len(validation))
            save(output/'split.json',{'train':[c['scene']['seed'] for c in train],
                'validation':[c['scene']['seed'] for c in validation],
                'excluded_existing_split':sorted(excluded),'unused_valid':len(bank)-goal})
            status.update('collecting_teachers',teacher_completed=0,teacher_planned=len(train))
            x=[];y=[];ids=[];rows=[]
            for start in range(0,len(train),args.workers):
                results=jobs(pool,collect_one,train[start:start+args.workers],status,'collecting_teachers')
                for case,(logs,observations,actions) in zip(train[start:start+args.workers],results):
                    rows.extend(logs)
                    if len(observations):
                        x.extend(observations);y.extend(actions);ids.extend([case['scene']['seed']]*len(actions))
                    save(output/'teacher_attempts.json',rows)
                    np.savez_compressed(output/'demonstrations.npz',observations=np.asarray(x,dtype=np.float32).reshape(-1,77),
                        actions=np.asarray(y,dtype=np.float32).reshape(-1,7),scene_ids=np.asarray(ids))
                status.update('collecting_teachers',teacher_completed=min(start+args.workers,len(train)),
                    teacher_successes=len(set(ids)),teacher_transitions=len(x))
                if start==0 and not x:
                    raise RuntimeError('Teacher preflight found no success; inspect all logged failures before continuing')
            if len(set(ids))<max(2,args.train_scenes//10):
                raise RuntimeError('Too few successful demonstration scenes for BC; failed attempts remain logged')
            labels=dict(np.load(output/'demonstrations.npz'))
            baseline=evaluate(pool,source,validation,output,status,'source')
            bc_path=train_bc(source,labels,output,args.seed,status,args.bc_updates)
            bc_result=evaluate(pool,bc_path,validation,output,status,'bc')
            best_path,best_result,best_label=source,baseline,'source'
            if selection_score(bc_result)>selection_score(best_result):
                best_path,best_result,best_label=bc_path,bc_result,'bc'
            # Adapt the BC model even if BC alone loses validation; the unchanged
            # source remains eligible for final selection and cannot be replaced silently.
            env=DummyVecEnv([lambda:Monitor(CaseBankEnv(train,reward_scale=.01),
                filename=str(output/'ppo_episodes'),info_keywords=('is_success','termination_reason'))])
            try:
                env.seed(args.seed)
                model=load_agent(bc_path,env=env,n_steps=256,batch_size=64,n_epochs=3,
                    learning_rate=3e-6,clip_range=.05,target_kl=.005,gamma=.99,gae_lambda=.95,
                    ent_coef=0.,vf_coef=.1,max_grad_norm=.5,tensorboard_log=None,seed=args.seed)
                model.policy.configure_learning(train_critic=True)
                frozen_ppo={n:p.detach().clone() for n,p in model.policy.named_parameters() if not p.requires_grad}
                model.ep_info_buffer=None;model.ep_success_buffer=None;model._episode_num=0
                model.set_logger(configure(str(output/'ppo_logs'),['stdout','csv']))
                initial_steps=model.num_timesteps
                for target in range(args.checkpoint_every,args.steps+1,args.checkpoint_every):
                    status.update('ppo_training',ppo_completed_steps=model.num_timesteps-initial_steps,ppo_planned_steps=args.steps)
                    model.learn(args.checkpoint_every,reset_num_timesteps=False)
                    assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                    assert all(torch.equal(p,frozen_ppo[n]) for n,p in model.policy.named_parameters() if n in frozen_ppo)
                    path=output/'models'/f'ppo_{target}.zip';model.save(path)
                    evaluation=evaluate(pool,path,validation,output,status,f'ppo_{target}')
                    if selection_score(evaluation)>selection_score(best_result):
                        best_path,best_result,best_label=path,evaluation,f'ppo_{target}'
                    save(output/'selection.json',{'selected':best_label,'model':str(best_path),
                        'metrics':{m:best_result[m]['metrics'] for m in ('ppo','enhanced')}})
            finally: env.close()
            early=evaluate(pool,best_path,validation,output,status,'early_recovery',early=True)
            selected_early=selection_score(early)>selection_score(best_result)
            if selected_early:best_result=early
            __import__('shutil').copy2(best_path,output/'models/selected_model.zip')
            rt=load_runtime()
            if selected_early:rt['controller']['recovery'].update(minimum_step=60,window=16,cooldown=16,maximum_rescues=5,commit_steps=6)
            save(output/'selected_runtime.json',rt)
            # Report old64 as regression only; no new claim of independent test.
            regression=load_json(ROOT/'data/validation_cases.json')
            final=evaluate(pool,output/'models/selected_model.zip',regression,output,status,'regression64',early=selected_early)
            assert digest(source)==source_hash
            save(output/'summary.json',{'selected':best_label,'early_recovery_selected':selected_early,
                'source_model_unchanged':True,'ppo_additional_steps':args.steps,
                'training_scenes':len(train),'validation_scenes':len(validation),
                'teacher_successes':len(set(ids)),'teacher_transitions':len(x),
                'source_validation':{m:baseline[m]['metrics'] for m in ('ppo','enhanced')},
                'selected_validation':{m:best_result[m]['metrics'] for m in ('ppo','enhanced')},
                'regression64':{m:final[m]['metrics'] for m in ('ppo','enhanced')},
                'final_test_performed':False})
            status.update('complete',status='completed',ppo_completed_steps=args.steps)
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--train-scenes',default=100,type=int)
    parser.add_argument('--validation-scenes',default=16,type=int)
    parser.add_argument('--start-seed',default=1212000,type=int)
    parser.add_argument('--max-attempts',default=2000,type=int)
    parser.add_argument('--workers',default=4,type=int)
    parser.add_argument('--seed',default=20261112,type=int)
    parser.add_argument('--bc-updates',default=2000,type=int)
    parser.add_argument('--steps',default=32768,type=int)
    parser.add_argument('--checkpoint-every',default=8192,type=int)
    args=parser.parse_args()
    if (min(args.train_scenes,args.validation_scenes,args.workers,args.bc_updates,args.checkpoint_every,args.steps)<=0
        or args.checkpoint_every%256 or args.steps%args.checkpoint_every):
        parser.error('Positive budgets required; checkpoint interval must divide steps and be a multiple of 256')
    run(args)

if __name__=='__main__':main()
