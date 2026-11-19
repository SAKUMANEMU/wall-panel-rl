"""Compare progress recovery against frozen v13; use a newly frozen test bank."""
import _bootstrap
import argparse,copy,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from collections import Counter
import numpy as np,torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv,counts,compose_mixture
from wall_panel_rl.rl.episode_imitation import episode_record
from wall_panel_rl.rl.predictive_safety import PredictiveSafetyController
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
from run_targeted_bc import read,dump,sha
WORK={}

def initialize(spec):
    torch.set_num_threads(1);p=read(ROOT/spec['preparation_report']/'protocol.json');model=PPO.load(ROOT/spec['source_model'],device='cpu')
    WORK.update(spec=spec,config=p['configuration'],settings=p['evaluation_settings'],model=model,weights=copy.deepcopy(model.policy.state_dict()))

def episode(job):
    mode,case=job;s=WORK['spec'];env=HybridFourEnv(copy.deepcopy(WORK['config']),[],copy.deepcopy(WORK['settings']));begin=time.perf_counter()
    try:
        assert case['task']=='dynamic' and case['strength']==[.18,.12]
        obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']});reference=env.cursor.path.copy()
        controller=(ProgressRecoveryController(WORK['model'],env,s['horizon'],s['gains'],s['recovery']) if mode=='recovery' else PredictiveSafetyController(WORK['model'],env,s['horizon'],s['gains']));reward=0.
        while True:
            before=(env.env.steps,env.env.sim_time,env.used);action=controller.action(obs);assert before==(env.env.steps,env.env.sim_time,env.used)
            obs,r,done,truncated,info=env.step(action);reward+=r
            if done or truncated:break
        assert np.array_equal(reference,env.cursor.path) and env.env.config['simulation']['max_steps']==240
        row=episode_record(env,info,'ppo',reward)
        row.update(execution_controller='predictive_progress_recovery' if mode=='recovery' else 'predictive_safety_v13',
            online_avoidance_planner=True,uses_global_rrt=False,model_weights_changed=False,
            interventions=controller.interventions,no_safe_action=controller.no_safe_action,gain_counts=controller.gain_counts,
            discarded_prediction_steps=controller.prediction_steps,prediction_wall_s=controller.seconds,
            total_wall_s=time.perf_counter()-begin,physics_elapsed_s=env.env.sim_time,
            reference_index=int(env.cursor.index),reference_last=len(reference)-1,
            recovery_actions=getattr(controller,'recovery_actions',0),rescues=getattr(controller,'rescues',0),
            baseline_protected=getattr(controller,'protected',False),forecast_calls=getattr(controller,'forecast_calls',0),
            forecast_cache_hits=getattr(controller,'forecast_cache_hits',0),recovery_events=getattr(controller,'recovery_events',[]))
        assert all(torch.equal(v,WORK['model'].policy.state_dict()[k]) for k,v in WORK['weights'].items())
        return row
    finally:env.close()

def preserve(before,after):
    assert [r['seed'] for r in before]==[r['seed'] for r in after]
    for a,b in zip(before,after):
        if a['reason']=='success':
            assert b['reason']=='success' and a['steps']==b['steps'] and b['recovery_actions']==0,'Changed an already successful baseline episode'
            assert abs(a['position_m']-b['position_m'])<=1e-6 and abs(a['orientation_deg']-b['orientation_deg'])<=1e-6

def run(spec,config_file):
    root=ROOT/spec['report_root']
    if root.exists():raise FileExistsError('Refusing recovery study overwrite')
    prep=read(ROOT/spec['preparation_report']/'protocol.json');fresh=read(ROOT/spec['preparation_root']/'protocol.json')
    assert sha(ROOT/spec['fresh_test_root']/'test_cases.json')==fresh['test_cases_sha256']
    assert sha(ROOT/spec['source_model'])==fresh['source_model_sha256']
    root.mkdir(parents=True);visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    if visible.resolve()!=root.resolve():visible.symlink_to(root,target_is_directory=True)
    state={'status':'running','optimizer_updates':0,'static_episodes':0}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra);dump(root/'status.json',state);print('PROGRESS_RECOVERY',json.dumps(state),flush=True)
    files=[str(config_file.relative_to(ROOT)),'scripts/evaluate_progress_recovery.py','src/wall_panel_rl/rl/progress_recovery.py','src/wall_panel_rl/rl/predictive_safety.py',
        'src/wall_panel_rl/rl/dynamic_four_training.py','src/wall_panel_rl/rl/conditional_recovery.py','src/wall_panel_rl/envs/wall_panel_env.py']
    hashes={f:sha(ROOT/f) for f in files};inputs=dict(prep['input_hashes'])
    for f in [spec['source_model'],spec['old_report']+'/validation_safety.json',spec['old_report']+'/tuning_horizon_12.json',spec['old_report']+'/tuning_train_seeds.json',spec['fresh_test_root']+'/test_cases.json']:
        inputs[f]=sha(ROOT/f)
    dump(root/'protocol.json',{'spec':spec,'configuration':prep['configuration'],'evaluation_settings':prep['evaluation_settings'],'code_hashes':hashes,'input_hashes':inputs,
        'scope':'PPO + predictive safety + local progress recovery; no neural updates or global RRT fallback',
        'rule':'Fixed training-screened controller; unchanged240-step budget, scene strength and collision/success rules; protect successful baseline continuations',
        'validation_gate':'More successes, fewer timeouts, no increase in collisions, all old successes unchanged',
        'test_rule':'New100-case paired test only after validation gate; freeze controller first; no tuning after test',
        'old_test_usage':'Prior v13 aggregate/terminal rows used only as development diagnosis, never as an independent v14 result'})
    for f in files:
        dst=root/'code_archive'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    def evaluate(pool,jobs,phase):
        rows=[];status(phase,processed=0,total=len(jobs))
        for row in pool.map(episode,jobs):
            rows.append(row)
            if len(rows)%4==0 or len(rows)==len(jobs):status(phase,processed=len(rows),total=len(jobs),outcomes=counts(rows))
        return {'rows':rows,'metrics':counts(rows)}
    try:
        with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
            train=read(ROOT/spec['data_root']/'train_cases.json');by_seed={c['scene']['seed']:c for c in train};ids=read(ROOT/spec['old_report']/'tuning_train_seeds.json')
            baseline_train=read(ROOT/spec['old_report']/'tuning_horizon_12.json');new_train=evaluate(pool,[('recovery',by_seed[i]) for i in ids],'training_screen');preserve(baseline_train['rows'],new_train['rows']);dump(root/'training_recovery.json',new_train)
            if new_train['metrics']['collision']>baseline_train['metrics']['collision']:raise RuntimeError('Training collision gate failed')
            baseline=read(ROOT/spec['old_report']/'validation_safety.json');cases=read(ROOT/spec['data_root']/'validation_cases.json')
            current=evaluate(pool,[('recovery',c) for c in cases],'validation');preserve(baseline['rows'],current['rows'])
            dump(root/'validation_baseline.json',baseline);dump(root/'validation_recovery.json',current)
            cache=read(ROOT/'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation_expert_cache.json')
            mixed=compose_mixture(current,cache);old_mixed=compose_mixture(baseline,cache);dump(root/'validation_mixture_recovery.json',mixed);dump(root/'validation_mixture_baseline.json',old_mixed)
            a=baseline['metrics'];b=current['metrics'];gate=(b['success']>a['success'] and b['timeout']<a['timeout'] and b['collision']<=a['collision'] and b['joint_limit']==0)
            dump(root/'validation_gate.json',{'passed':gate,'baseline':a,'recovery':b,'old_successes_preserved':True,'locked_before_test':True})
            test={'performed':False,'reason':'Validation gate not passed; fresh test unused'}
            if gate:
                tests=read(ROOT/spec['fresh_test_root']/'test_cases.json');old=evaluate(pool,[('baseline',c) for c in tests],'fresh_test_baseline');new=evaluate(pool,[('recovery',c) for c in tests],'fresh_test_recovery');preserve(old['rows'],new['rows'])
                old['independent_test']=new['independent_test']=True;dump(root/'independent_baseline.json',old);dump(root/'independent_recovery.json',new)
                test={'performed':True,'baseline':old['metrics'],'recovery':new['metrics'],'paired':dict(Counter(a['reason']+' -> '+b['reason'] for a,b in zip(old['rows'],new['rows']))),'old_successes_preserved':True,'no_tuning_after_test':True}
            dump(root/'independent_test.json',test)
        for f,h in hashes.items():assert sha(ROOT/f)==h,'Runtime code changed'
        for f,h in inputs.items():assert sha(ROOT/f)==h,'Frozen input changed'
        summary={'validation_baseline':a,'validation_recovery':b,'validation_mixture_baseline':old_mixed['metrics'],'validation_mixture_recovery':mixed['metrics'],
            'training_baseline':baseline_train['metrics'],'training_recovery':new_train['metrics'],'validation_gate_passed':gate,'independent_test':test,
            'old_successes_preserved':True,'source_unchanged':True,'optimizer_updates':0,'neural_weights_changed':False,'pure_ppo':False,'global_rrt_fallback':False,
            'default_old_controller_preserved':True,'real_step_limit':240}
        dump(root/'summary.json',summary);status('complete',status='completed',summary=summary)
    except Exception as e:status('failed',status='failed',error=repr(e));raise

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_progress_recovery.yaml');p.add_argument('--run',action='store_true');args=p.parse_args();spec=yaml.safe_load(args.config.read_text());print({'mode':'run' if args.run else 'check_only','controller':'predictive progress recovery','maximum_real_steps':240},flush=True)
    if args.run:run(spec,args.config.resolve())
if __name__=='__main__':main()
