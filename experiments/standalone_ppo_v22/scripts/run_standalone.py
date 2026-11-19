"""Pure-policy selection: corrective distillation, DAgger states and PPO arms.

Experts label training data only. Validation/test execution uses direct policy
inference with the original environment wrapper, budget, rewards and tolerances.
"""
import _bootstrap
import argparse,copy,csv,json,multiprocessing as mp,time,traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from run_adaptation import (WORK, Status, save, digest, jobs, metrics, initialize_worker,
                            teacher_episode, generate_one)
from wall_panel_rl.rl.standalone_policy import StandalonePolicy,ACTOR_PREFIXES
from wall_panel_rl.rl.slow_dynamic_four import SlowDynamicFourEnv,planning_snapshot,assert_planning_paused
from wall_panel_rl.rl.predictive_safety import PredictiveSafetyController
from wall_panel_rl.rl.waypoint_tracking import bounded_action
from wall_panel_rl.submission import load_json,load_agent,load_runtime,CaseBankEnv,run_episode
from wall_panel_rl.utils.config import ROOT


def converted(path, **kwargs):
    return PPO.load(path,device='cpu',custom_objects={'policy_class':StandalonePolicy},**kwargs)


def worker_model(path):
    if WORK.get('current')!=str(path):
        WORK['model']=load_agent(path);WORK['current']=str(path)
    return WORK['model']


def collect(job):
    """Successful full teacher data plus labels at autonomous visited states.

    Prefix corrective labels need only be safe in the declared 12-step model
    horizon. They are NOT misrepresented as successful full demonstrations.
    """
    path,case,include_full=job;model=worker_model(path)
    logs=[];all_x=[];all_y=[];all_w=[];seed=case['scene']['seed']
    if include_full:
        row,x,y=teacher_episode(case,'enhanced')
        logs.append(dict(kind='full_enhanced_teacher',**{k:v for k,v in row.items() if k not in ('diagnostics','recovery_events')}))
        if row['accepted_for_imitation']:
            all_x.extend(x);all_y.extend(y);all_w.extend(np.ones(len(y)))
    rt=WORK['runtime'];env=SlowDynamicFourEnv(rt['configuration'],[],rt['settings'])
    corrected=0;no_safe=0;steps=0;trace=[]
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
        reference=env.cursor.path.copy()
        predictor=PredictiveSafetyController(model,env,12,rt['controller']['gains'])
        while True:
            student=model.predict(obs,deterministic=True)[0]
            before=planning_snapshot(env);old_no_safe=predictor.no_safe_action
            target=predictor.action(obs)
            assert_planning_paused(env,before)
            valid=predictor.no_safe_action==old_no_safe
            changed=np.max(np.abs(student-target))>.02
            q,_=env.env.joint_state()
            projected=bounded_action(q,np.clip(target,-1,1),env.env.lower,env.env.upper,env.env.max_increment,env.margin)
            if valid:
                all_x.append(obs.copy());all_y.append(projected)
                all_w.append(4. if changed else .5)
                corrected+=int(changed)
            else:no_safe+=1
            obs,reward,done,truncated,info=env.step(student);steps+=1
            trace.append({'step':steps,'teacher_changed_action':bool(changed),'teacher_safe_horizon':bool(valid),
                          'position_m':float(info['position_error']),'reference_index':env.cursor.index,
                          'clearance_m':float(info['obstacle_clearance_capped_m'])})
            if done or truncated:break
        assert np.array_equal(reference,env.cursor.path)
        logs.append({'kind':'student_visited_states','seed':seed,'reason':info['termination_reason'],
            'steps':steps,'corrective_labels':corrected,'no_safe_label_states':no_safe,
            'position_m':float(info['position_error']),'orientation_deg':float(np.rad2deg(info['orientation_error'])),
            'trace':trace,'reference_preserved':True})
    finally:env.close()
    return logs,np.asarray(all_x,dtype=np.float32).reshape(-1,77),np.asarray(all_y,dtype=np.float32).reshape(-1,7),np.asarray(all_w,dtype=np.float32)


def evaluate_one(job):
    path,case=job;model=worker_model(path)
    row,_=run_episode(model,case,'ppo')
    assert not row['uses_global_rrt'] and row['local_rescues']==0
    row.update(waiting_fraction=row['zero_command_steps']/max(1,row['steps']),
               stationary_fraction=row['stationary_steps']/max(1,row['steps']))
    return row


def score(result):
    m=result['metrics']
    error=np.mean([r['position_m']+.12*np.deg2rad(r['orientation_deg']) for r in result['rows']])
    return (m['success'],-m['joint_limit'],-m['collision'],-float(error))


def evaluate(pool,path,cases,out,status,label,selection=True):
    status.update('evaluate_'+label,evaluating=label)
    rows=jobs(pool,evaluate_one,[(str(path),c) for c in cases],status,'evaluate_'+label)
    result={'model':str(path),'model_sha256':digest(path),'mode':'direct_ppo',
        'prediction_or_recovery_during_execution':False,'selection_data':selection,
        'metrics':metrics(rows),'rows':rows}
    save(out/'evaluations'/f'{label}.json',result)
    return result


def weighted_bc_loss(policy,x,y,w):
    per=(policy.imitation_mean(x)-y).square().mean(dim=1)
    return (per*w).sum()/w.sum().clamp_min(1e-8)


class Supervised:
    def __init__(self,path,data,lr,seed):
        self.model=converted(path);self.model.policy.configure_learning(train_critic=False)
        self.x=torch.as_tensor(data['observations']);self.y=torch.as_tensor(data['actions'])
        self.w=torch.as_tensor(data['weights']);self.ids=data['scene_ids']
        self.scenes=np.unique(self.ids);self.indices={s:np.flatnonzero(self.ids==s) for s in self.scenes}
        self.rng=np.random.default_rng(seed)
        self.params=[p for p in self.model.policy.parameters() if p.requires_grad]
        self.frozen={n:p.detach().clone() for n,p in self.model.policy.named_parameters() if not p.requires_grad}
        self.optimizer=torch.optim.Adam(self.params,lr=lr)
    def batch(self,size=256):
        chosen=self.rng.choice(self.scenes,size)
        return np.array([self.rng.choice(self.indices[s]) for s in chosen])
    def update(self):
        i=self.batch();loss=weighted_bc_loss(self.model.policy,self.x[i],self.y[i],self.w[i])
        if not torch.isfinite(loss):raise FloatingPointError('Invalid supervised loss')
        self.optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.params,.5);self.optimizer.step()
        return float(loss.detach())
    def verify(self):
        assert all(torch.equal(p,self.frozen[n]) for n,p in self.model.policy.named_parameters() if n in self.frozen)
        assert all(torch.isfinite(p).all() for p in self.model.policy.parameters())


class AnchoredPPO(BaseCallback):
    """Small KL-guarded BC updates occur BEFORE the next on-policy rollout."""
    def __init__(self,data,status,out,seed):
        super().__init__();self.data=data;self.status=status;self.out=out
        self.rng=np.random.default_rng(seed);self.accepted=0;self.rejected=0
    def _on_training_start(self):
        self.params=[p for n,p in self.model.policy.named_parameters() if n.startswith(ACTOR_PREFIXES)]
        self.optimizer=torch.optim.Adam(self.params,lr=1e-6)
        self.x=torch.as_tensor(self.data['observations']);self.y=torch.as_tensor(self.data['actions']);self.w=torch.as_tensor(self.data['weights'])
    def _on_step(self):return True
    def _on_rollout_start(self):
        p=self.model.policy;p.set_training_mode(True)
        ids=self.rng.integers(0,len(self.x),256);x=self.x[ids]
        with torch.no_grad():
            d=p.get_distribution(x).distribution
            old=torch.distributions.Normal(d.loc.clone(),d.scale.clone())
        weights=[v.detach().clone() for v in self.params];state=copy.deepcopy(self.optimizer.state_dict())
        # Only actor parameter tensors are passed to autograd. The critic stays
        # outside this supervised optimizer and PPO retains its own Adam state.
        loss=.2*weighted_bc_loss(p,x,self.y[ids],self.w[ids])
        grads=torch.autograd.grad(loss,self.params,allow_unused=False)
        self.optimizer.zero_grad()
        for parameter,grad in zip(self.params,grads):parameter.grad=grad
        torch.nn.utils.clip_grad_norm_(self.params,.5);self.optimizer.step()
        with torch.no_grad():
            new=p.get_distribution(x).distribution
            kl=float(torch.distributions.kl_divergence(old,new).sum(-1).mean())
        keep=np.isfinite(kl) and kl<=.005
        if keep:self.accepted+=1
        else:
            with torch.no_grad():
                for parameter,old_value in zip(self.params,weights):parameter.copy_(old_value)
            self.optimizer.load_state_dict(state);self.rejected+=1
        with self.out.open('a') as f:
            f.write(json.dumps({'step':self.num_timesteps,'loss':float(loss.detach()),'kl':kl,'accepted':bool(keep)})+'\n')


def collect_bank(pool,path,bank,out,status,label,full):
    xs=[];ys=[];weights=[];scene_ids=[];logs=[]
    for start in range(0,len(bank),4):
        batch=bank[start:start+4]
        result=jobs(pool,collect,[(str(path),c,full) for c in batch],status,'collect_'+label)
        for c,(rows,x,y,w) in zip(batch,result):
            logs.extend(rows);xs.extend(x);ys.extend(y);weights.extend(w);scene_ids.extend([c['scene']['seed']]*len(x))
        save(out/f'{label}_episodes.json',logs)
        status.update('collect_'+label,collected_scenes=min(start+4,len(bank)),collection_scenes=len(bank),labels=len(xs))
    data={'observations':np.asarray(xs,dtype=np.float32).reshape(-1,77),
        'actions':np.asarray(ys,dtype=np.float32).reshape(-1,7),'weights':np.asarray(weights,dtype=np.float32),
        'scene_ids':np.asarray(scene_ids)}
    assert len(xs)>0 and np.isfinite(data['observations']).all() and np.isfinite(data['actions']).all()
    np.savez_compressed(out/f'{label}_labels.npz',**data)
    return data


def run(args):
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    out=args.output.resolve()
    if out.exists():raise FileExistsError('Choose a new output directory')
    out.mkdir(parents=True);(out/'models').mkdir();status=Status(out)
    source=ROOT/'models/current_ppo.zip';sha=digest(source)
    train=load_json(ROOT/'data/train_cases.json');validation=load_json(ROOT/'data/selection_cases.json')
    regression=load_json(ROOT/'data/validation_cases.json')
    excluded={c['scene']['seed'] for c in train+validation+regression}
    assert len({c['scene']['seed'] for c in train})==100
    assert not {c['scene']['seed'] for c in train}&{c['scene']['seed'] for c in validation+regression}
    save(out/'protocol.json',{'objective':'Direct PPO installation success only; no predictive action filter or RRT at evaluation',
        'source_sha256':sha,'train_scenes':len(train),'selection_validation_scenes':len(validation),
        'fresh_test_scenes':args.test_scenes,'seed':args.seed,
        'distillation_checkpoints':[500,1500,3000],'ppo_steps_per_arm':args.steps,
        'ppo_learning_rates':[3e-6,1e-5],'action_std':.05,'teacher_horizon':12,
        'data_separation':'All expert labels use training cases only; new test frozen before adaptation and evaluated only after model selection',
        'code_hashes':{str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'src').rglob('*.py'))},
        'script_sha256':digest(Path(__file__)),'task_reward_and_termination_unchanged':True})
    try:
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize_worker) as pool:
            # Freeze a fresh geometry-valid test bank before learning. Do not use
            # it for checkpoint/controller selection; include every final outcome.
            test=[];attempts=[];seed=args.test_start_seed
            while len(test)<args.test_scenes and len(attempts)<1000:
                seeds=[]
                while len(seeds)<args.workers*2:
                    if seed not in excluded:seeds.append(seed)
                    seed+=1
                result=jobs(pool,generate_one,seeds,status,'freeze_fresh_test')
                for item in result:
                    case=item.pop('case',None);attempts.append(item)
                    if case is not None:test.append(case)
                save(out/'test_generation_attempts.json',attempts)
                save(out/'generated_test_candidates.json',test)
                status.update('freeze_fresh_test',valid_test_scenes=min(len(test),args.test_scenes),test_target=args.test_scenes)
            if len(test)<args.test_scenes:raise RuntimeError('Fresh test geometry budget exhausted')
            test=test[:args.test_scenes];save(out/'frozen_test_cases.json',test)
            baseline=evaluate(pool,source,validation,out,status,'source_validation')
            best_path,best_result,best_label=source,baseline,'source'
            def select(path,result,label):
                nonlocal best_path,best_result,best_label
                if score(result)>score(best_result):best_path,best_result,best_label=path,result,label
                save(out/'selection.json',{'selected':best_label,'model':str(best_path),
                     'direct_policy_metrics':best_result['metrics'],'selection_uses_enhanced_execution':False})
            data=collect_bank(pool,source,train,out,status,'initial_teacher',True)
            supervised=Supervised(source,data,3e-5,args.seed)
            for update in range(1,3001):
                loss=supervised.update()
                if update%100==0:
                    supervised.verify();status.update('distillation',supervised_updates=update,supervised_planned=3000,loss=loss)
                    with (out/'supervised_loss.csv').open('a') as f:f.write(f'initial,{update},{loss}\n')
                if update in (500,1500,3000):
                    path=out/'models'/f'distill_{update}.zip';supervised.model.save(path)
                    select(path,evaluate(pool,path,validation,out,status,f'distill_{update}'),f'distill_{update}')
            # Aggregate labels on the best autonomous actor's visited states.
            new=collect_bank(pool,best_path,train,out,status,'dagger',False)
            data={key:np.concatenate([data[key],new[key]]) for key in data}
            np.savez_compressed(out/'aggregate_labels.npz',**data)
            supervised=Supervised(best_path,data,1e-5,args.seed+1)
            for update in range(1,1501):
                loss=supervised.update()
                if update%100==0:
                    supervised.verify();status.update('dagger_fit',supervised_updates=update,supervised_planned=1500,loss=loss)
                    with (out/'supervised_loss.csv').open('a') as f:f.write(f'dagger,{update},{loss}\n')
                if update in (500,1500):
                    path=out/'models'/f'dagger_{update}.zip';supervised.model.save(path)
                    select(path,evaluate(pool,path,validation,out,status,f'dagger_{update}'),f'dagger_{update}')
            ppo_start=best_path
            for arm,lr in enumerate((3e-6,1e-5)):
                env=DummyVecEnv([lambda:Monitor(CaseBankEnv(train,reward_scale=.01),filename=str(out/f'arm{arm}_episodes'),info_keywords=('is_success','termination_reason'))])
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
                    model.set_logger(configure(str(out/f'ppo_arm{arm}'),['stdout','csv']))
                    for target in range(8192,args.steps+1,8192):
                        status.update('ppo_training',ppo_arm=arm,learning_rate=lr,arm_steps_completed=target-8192,arm_steps_planned=args.steps)
                        callback=AnchoredPPO(data,status,out/f'arm{arm}_anchor_updates.jsonl',args.seed+target)
                        model.learn(8192,reset_num_timesteps=False,callback=callback)
                        assert all(torch.equal(p,frozen[n]) for n,p in model.policy.named_parameters() if n in frozen)
                        assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                        path=out/'models'/f'ppo_arm{arm}_{target}.zip';model.save(path)
                        label=f'ppo_arm{arm}_{target}'
                        select(path,evaluate(pool,path,validation,out,status,label),label)
                finally:env.close()
            __import__('shutil').copy2(best_path,out/'models/selected_model.zip')
            # Checkpoints are frozen before any test execution. No later model
            # selection or teacher data collection can depend on these outcomes.
            save(out/'frozen_selection.json',{'selected':best_label,'sha256':digest(out/'models/selected_model.zip'),
                'validation_metrics':best_result['metrics'],'fresh_test_not_yet_evaluated':True})
            old=evaluate(pool,out/'models/selected_model.zip',regression,out,status,'selected_regression64',False)
            test_source=evaluate(pool,source,test,out,status,'source_fresh_test',False)
            test_new=evaluate(pool,out/'models/selected_model.zip',test,out,status,'selected_fresh_test',False)
            assert digest(source)==sha
            save(out/'summary.json',{'selected':best_label,'validation':best_result['metrics'],
                'source_validation':baseline['metrics'],'regression64':old['metrics'],
                'fresh_test_source':test_source['metrics'],'fresh_test_selected':test_new['metrics'],
                'pure_policy_execution':True,'total_ppo_steps':args.steps*2,'source_unchanged':True})
            status.update('complete',status='completed')
    except BaseException as exc:
        status.update('failed',status='failed',error=repr(exc),traceback=traceback.format_exc());raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261112)
    p.add_argument('--steps',type=int,default=32768);p.add_argument('--test-scenes',type=int,default=32)
    p.add_argument('--test-start-seed',type=int,default=1225000);a=p.parse_args()
    if a.steps<=0 or a.steps%8192 or min(a.workers,a.test_scenes)<=0:p.error('Invalid budget')
    run(a)
if __name__=='__main__':main()
