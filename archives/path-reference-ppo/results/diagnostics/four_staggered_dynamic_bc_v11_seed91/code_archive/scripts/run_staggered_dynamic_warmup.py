"""Train-only collision diagnosis, verified dynamic RRT labels, bounded BC warmup.

The independent test bank is never opened here. All policy updates use only
verified TRAIN episodes; validation selects checkpoints without supplying labels.
"""
import _bootstrap
import argparse,copy,json,multiprocessing as mp,shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np
import pybullet as pb
import torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv,case_at_speed,evaluate_ppo,verified_expert_episode,counts
from wall_panel_rl.rl.four_ppo_continuation import hard_probabilities,state_digest
from wall_panel_rl.rl.four_priority_bc import sample_rows
from run_targeted_bc import read,dump,sha

WORK={}


def initialize(spec):
    torch.set_num_threads(1)
    prep=read(ROOT/spec['preparation_report']/'protocol.json')
    WORK.update(spec=spec,config=prep['configuration'],settings=prep['evaluation_settings'],
                model=PPO.load(ROOT/spec['source_model'],device='cpu'))


def collision_pairs(e):
    """Geometric/contact evidence at termination; unknown stays unknown."""
    rows=[];margin=e.config['collision']['margin'];kw={'physicsClientId':e.client}
    targets=[('obstacle_'+str(i),b) for i,b in enumerate(e.obstacle_ids)]+[('wall',e.wall)]
    queries=[(name,body,other,body2) for name,body in [('arm',e.robot),('panel',e.panel)] for other,body2 in targets]
    queries += [('arm',e.robot,'arm',e.robot),('arm',e.robot,'panel',e.panel)]
    for name,a,other,b in queries:
        hits=pb.getClosestPoints(a,b,margin,**kw)
        for hit in hits:
            if a==b and abs(hit[3]-hit[4])<=1:continue
            if a==e.robot and b==e.panel and hit[3]==e.ee:continue
            rows.append({'body':name,'other':other,'link':hit[3],'other_link':hit[4],'distance_m':float(hit[8])})
    return rows


def diagnose(case):
    env=HybridFourEnv(copy.deepcopy(WORK['config']),[],copy.deepcopy(WORK['settings']))
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        while True:
            obs,_,done,truncated,info=env.step(WORK['model'].predict(obs,deterministic=True)[0])
            if done or truncated:break
        row={'seed':case['scene']['seed'],'reason':info['termination_reason'],'steps':env.env.steps,
             'time_s':env.env.sim_time,'position_m':info['position_error'],'orientation_deg':float(np.rad2deg(info['orientation_error'])),
             'reference_index':int(env.cursor.index),'reference_last':len(env.cursor.path)-1,
             'stage':'early' if env.env.steps<=WORK['spec']['warmup']['head_steps'] else ('terminal' if env.cursor.index>=len(env.cursor.path)-2 else 'middle'),
             'collision_pairs':collision_pairs(env.env) if info['termination_reason']=='collision' else []}
        return row
    finally:env.close()


def teacher(job):
    case,phase_seed,planner_seed=job
    spec=WORK['spec'];case=case_at_speed(case,spec['validation_target_speed'],phase_rng=np.random.default_rng(phase_seed),amplitude=spec['validation_target_amplitude'])
    env=HybridFourEnv(copy.deepcopy(WORK['config']),[],copy.deepcopy(WORK['settings']))
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        row,pairs=verified_expert_episode(env,planner_seed,obs)
        row.update(phase_seed=phase_seed,amplitude_m=case['strength'][0],speed_m_s=case['strength'][1])
        return {'row':row,'case':case,'pairs':pairs if row['accepted_for_imitation'] else []}
    finally:env.close()


def selection(m,baseline):
    return [int(m['joint_limit']==0 and m['collision']<=baseline['collision']),m['success'],-m['collision'],-m['timeout']]


def run(spec,config_file):
    torch.set_num_threads(1);torch.manual_seed(spec['seed']);warm=spec['warmup']
    report=ROOT/warm['report_root'];models=ROOT/warm['model_root']
    if report.exists() or models.exists():raise FileExistsError('Refusing warmup overwrite')
    if not spec.get('dynamic_only') or spec['static_probability']!=0:raise ValueError('100% dynamic run required')
    prep=read(ROOT/spec['preparation_report']/'protocol.json')
    for path,digest in prep['input_hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Prepared input changed: '+path)
    cases=read(ROOT/spec['data_root']/'train_cases.json');validation=read(ROOT/spec['data_root']/'validation_cases.json')
    split=read(ROOT/spec['data_root']/'split.json');train_ids=set(split['train']);excluded=set(split['validation']+split['test']+split['previously_excluded'])
    assert not train_ids&excluded and all(c['task']=='dynamic' for c in cases+validation)
    source_hash=sha(ROOT/spec['source_model']);assert source_hash==read(ROOT/spec['source_report']/'candidate_manifest.json')['model_sha256']
    report.mkdir(parents=True);models.mkdir(parents=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    state={'status':'running','ppo_steps':0,'dynamic_probability':1.}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('DYNAMIC_WARMUP',json.dumps(state),flush=True)
    code=['scripts/run_staggered_dynamic_warmup.py',str(config_file.relative_to(ROOT)),
          'src/wall_panel_rl/rl/dynamic_four_training.py','src/wall_panel_rl/rl/slow_dynamic_four.py',
          'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/rl/conditional_recovery.py',
          'src/wall_panel_rl/rl/four_priority_bc.py','src/wall_panel_rl/envs/wall_panel_env.py',
          'src/wall_panel_rl/envs/collision.py','src/wall_panel_rl/baselines/classical.py',
          'src/wall_panel_rl/planning/timed_rrt.py','src/wall_panel_rl/rl/multi_obstacle_adapter.py']
    hashes={f:sha(ROOT/f) for f in code};inputs=dict(prep['input_hashes']);inputs[spec['source_model']]=source_hash
    dump(report/'protocol.json',{'spec':spec,'configuration':prep['configuration'],'evaluation_settings':prep['evaluation_settings'],
        'code_hashes':hashes,'input_hashes':inputs,'train_seeds':sorted(train_ids),'excluded_seeds':sorted(excluded),
        'source_sha256':source_hash,'selection':'Dynamic target success first, collision non-increase, zero joint violations; earlier wins ties',
        'teacher_rule':'TRAIN only; random phases; timed pre/post checks and independent physical replay; no evaluation labels',
        'static_episodes':0,'old_bc_pool_reused':False,'ppo_optimizer_preserved':True,'ppo_steps':0,
        'independent_test_evaluated':False})
    for f in code:
        dest=report/'code_archive'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    try:
        with ProcessPoolExecutor(max_workers=warm['teacher_workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
            status('diagnosing_training_cases',processed=0,total=len(cases));diagnostics=[]
            for start in range(0,len(cases),24):
                diagnostics.extend(pool.map(diagnose,cases[start:start+24]));dump(report/'training_diagnosis.json',diagnostics)
                status('diagnosing_training_cases',processed=len(diagnostics),total=len(cases),outcomes=dict(Counter(r['reason'] for r in diagnostics)))
            failed=[r['seed'] for r in diagnostics if r['reason']!='success'];collisions=[r for r in diagnostics if r['reason']=='collision']
            dump(report/'diagnosis_summary.json',{'outcomes':dict(Counter(r['reason'] for r in diagnostics)),
                'collision_stages':dict(Counter(r['stage'] for r in collisions)),
                'collision_components':dict(Counter(r['body']+' / '+r['other'] for x in collisions for r in x['collision_pairs'])),
                'collision_component_unit':'terminal contact pairs; multiple pairs may belong to one episode',
                'collision_episodes_without_pair_evidence':sum(not x['collision_pairs'] for x in collisions),
                'source':'training cases only','evaluation_cases_used':False})
            probs=hard_probabilities(cases,failed,.5);dump(report/'sampling.json',{'hard_training_seeds':sorted(failed),'probabilities':probs.tolist(),'hard_fraction':.5,'diagnosis_only_on_train':True})
            rng=np.random.default_rng(spec['seed']+20000000);by_id={c['scene']['seed']:c for c in cases};accepted=[];attempt_rows=[];attempt=0;layout_attempts=Counter()
            status('collecting_verified_dynamic_teachers',accepted=0,target=warm['teacher_episodes'],attempts=0)
            while len(accepted)<warm['teacher_episodes'] and attempt<warm['teacher_attempts']:
                jobs=[]
                for _ in range(min(8,warm['teacher_attempts']-attempt)):
                    # Repeat failed training layouts with distinct phases, while retaining route coverage.
                    available=[i for i in by_id if layout_attempts[i]<warm['phase_variants']]
                    hard=[i for i in failed if i in available]
                    ids=hard if hard and rng.random()<.7 else available
                    seed=int(rng.choice(ids));layout_attempts[seed]+=1
                    jobs.append((by_id[seed],int(rng.integers(1,2**31-1)),int(rng.integers(1,2**31-1))))
                for result in pool.map(teacher,jobs):
                    attempt_rows.append(result['row'])
                    if result['row']['accepted_for_imitation']:accepted.append(result)
                attempt+=len(jobs);dump(report/'teacher_attempts.json',attempt_rows)
                status('collecting_verified_dynamic_teachers',accepted=len(accepted),target=warm['teacher_episodes'],attempts=attempt)
            if len(accepted)<max(16,warm['teacher_episodes']//2):raise RuntimeError('Insufficient physically verified dynamic teacher episodes')
        assert all(x['row']['seed'] in train_ids and x['row']['seed'] not in excluded and x['case']['task']=='dynamic' for x in accepted)
        torch.save({'episodes':accepted,'train_only':True,'dynamic_only':True},models/'teacher.pt')
        offsets=np.r_[0,np.cumsum([len(x['pairs']) for x in accepted])]
        observations=np.stack([pair[0] for x in accepted for pair in x['pairs']]).astype(np.float32)
        actions=np.stack([pair[1] for x in accepted for pair in x['pairs']]).astype(np.float32)
        assert observations.shape[1:]==(77,) and actions.shape[1:]==(7,) and np.isfinite(observations).all() and np.isfinite(actions).all() and np.max(np.abs(actions))<=1.000001
        data={'offsets':offsets};model=PPO.load(ROOT/spec['source_model'],device='cpu');counter=model.num_timesteps
        initial=copy.deepcopy(model.policy.state_dict());optimizer_hash=state_digest(model.policy.optimizer.state_dict())
        model.policy.configure_learning(train_critic=False)
        parameters=[p for name,p in model.policy.named_parameters() if name.startswith('multi_adapter.') and p.requires_grad]
        assert model.policy.imitation_prefixes==('multi_adapter.',) and parameters
        optimizer=torch.optim.Adam(parameters,lr=warm['learning_rate']);history=[];best=None;selected=None;accepted_updates=0;rejected_updates=0
        rng=np.random.default_rng(spec['seed']+30000000)
        guard=torch.as_tensor(observations[rng.choice(len(observations),min(512,len(observations)),replace=False)])
        with torch.no_grad():
            dist=model.policy.get_distribution(guard).distribution
            original_dist=torch.distributions.Normal(dist.loc.detach().clone(),dist.scale.detach().clone())
        def check(update):
            nonlocal best,selected
            status('validating_bc',bc_updates=update)
            result=evaluate_ppo(model,prep['configuration'],prep['evaluation_settings'],validation,spec['validation_target_speed'],amplitude=spec['validation_target_amplitude'])
            if update==0:dump(report/'baseline_validation.json',result)
            metrics=result['metrics'];baseline=history[0]['metrics'] if history else metrics
            event={'bc_updates':update,'metrics':metrics,'score':selection(metrics,baseline),'accepted_updates':accepted_updates,'rejected_updates':rejected_updates}
            model.save(models/f'bc_{update}.zip');torch.save(optimizer.state_dict(),models/f'bc_{update}_optimizer.pt')
            for name,value in initial.items():
                if not name.startswith('multi_adapter.'):assert torch.equal(value,model.policy.state_dict()[name])
            assert model.num_timesteps==counter and state_digest(model.policy.optimizer.state_dict())==optimizer_hash
            if best is None or tuple(event['score'])>best:
                best=tuple(event['score']);selected=copy.deepcopy(event);model.save(models/'selected_model.zip')
            history.append(event);dump(report/'validation'/f'bc_{update}.json',result);dump(report/'history.json',history);status('bc_checkpoint',**event,selected_bc_updates=selected['bc_updates'])
        check(0)
        for update in range(1,warm['updates']+1):
            ids=rng.integers(0,len(accepted),warm['batch_size']);rows=sample_rows(data,ids,rng,warm['head_probability'],warm['head_steps'])
            x=torch.as_tensor(observations[rows]);target=torch.as_tensor(actions[rows])
            with torch.no_grad():
                d=model.policy.get_distribution(guard).distribution;before_dist=torch.distributions.Normal(d.loc.detach().clone(),d.scale.detach().clone())
            weights=[p.detach().clone() for p in parameters];previous_optimizer=copy.deepcopy(optimizer.state_dict())
            loss=(model.policy.tracking_mean(x)-target).square().mean()+.001*model.policy.adapter_correction(x).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite BC loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(parameters,.5);optimizer.step()
            with torch.no_grad():
                d=model.policy.get_distribution(guard).distribution
                step_kl=float(torch.distributions.kl_divergence(before_dist,d).sum(-1).mean())
                source_kl=float(torch.distributions.kl_divergence(original_dist,d).sum(-1).mean())
            keep=bool(np.isfinite([step_kl,source_kl]).all() and step_kl<=warm['maximum_step_kl'] and source_kl<=warm['maximum_source_kl'])
            if not keep:
                with torch.no_grad():
                    for p,w in zip(parameters,weights):p.copy_(w)
                optimizer.load_state_dict(previous_optimizer);rejected_updates+=1
            else:accepted_updates+=1
            with (report/'loss.jsonl').open('a') as f:f.write(json.dumps({'bc_update':update,'loss':float(loss.detach()),'step_kl':step_kl,'source_kl':source_kl,'accepted':keep})+'\n')
            if update in warm['checkpoints']:check(update)
        model.save(models/'final_model.zip');chosen=PPO.load(models/'selected_model.zip',device='cpu')
        assert state_digest(chosen.policy.optimizer.state_dict())==optimizer_hash
        # New auxiliary optimizer deliberately starts fresh after offline BC.
        # Only verified dynamic labels are carried into PPO, never the old mixed pool.
        fresh=torch.optim.Adam([p for n,p in chosen.policy.named_parameters() if n.startswith('multi_adapter.')],lr=spec['auxiliary_bc']['learning_rate'])
        pairs=[pair for episode in accepted for pair in episode['pairs']]
        torch.save({'optimizer':fresh.state_dict(),'rng':np.random.default_rng(spec['seed']+40000000).bit_generator.state,
            'accepted_updates':0,'rejected_updates':0,'pool':pairs,'ppo_num_timesteps':counter},models/'selected_auxiliary_state.pt')
        shutil.copy2(models/'selected_model.zip',report/'selected_model.zip')
        dump(report/'candidate_manifest.json',{'configuration':prep['configuration'],'evaluation_settings':prep['evaluation_settings'],
            'model_sha256':sha(report/'selected_model.zip'),'dynamic_only_labels':True,'selected_bc_updates':selected['bc_updates'],
            'auxiliary_state':str((models/'selected_auxiliary_state.pt').relative_to(ROOT)),'default_model_preserved':True})
        summary={'selected_bc_updates':selected['bc_updates'],'selected':selected,'source_validation':history[0]['metrics'],
            'teacher_episodes':len(accepted),'teacher_transitions':len(pairs),'teacher_attempts':attempt,'teacher_outcomes':dict(Counter(r['reason'] for r in attempt_rows)),
            'bc_updates':warm['updates'],'accepted_bc_updates':accepted_updates,'rejected_bc_updates':rejected_updates,
            'diagnosed_training_layouts':len(cases),'training_diagnosis':counts(diagnostics),'head_sampling_probability':warm['head_probability'],
            'static_episodes':0,'static_labels':0,'independent_test_evaluated':False,'ppo_optimizer_preserved':True,
            'fresh_auxiliary_optimizer_after_warmup':True,'source_unchanged':sha(ROOT/spec['source_model'])==source_hash,'ppo_steps':0}
        for f,digest in hashes.items():assert sha(ROOT/f)==digest,'Runtime code changed: '+f
        for f,digest in inputs.items():assert sha(ROOT/f)==digest,'Runtime input changed: '+f
        dump(report/'summary.json',summary);status('complete',status='completed',summary=summary)
        (report/'README.md').write_text('# Dynamic-only diagnosis and BC warmup\n\n'+json.dumps(summary,indent=2)+'\n\nNo static episode or old mixed-label pool is used. All teacher labels come from training cases after time checks and independent physical replay. The reserved test remains unused. PPO counters, optimizer, critic, base actor and standard deviation are unchanged during BC.\n')
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_staggered_dynamic_preparation.yaml');p.add_argument('--run',action='store_true')
    args=p.parse_args();spec=yaml.safe_load(args.config.read_text());print({'mode':'run' if args.run else 'check_only','static_probability':spec['static_probability'],'warmup':spec['warmup']},flush=True)
    if args.run:run(spec,args.config.resolve())

if __name__=='__main__':main()
