"""Controlled PPO+prediction study; no learning and no pure-PPO relabelling."""
import _bootstrap
import argparse,copy,json,multiprocessing as mp,shutil,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
import numpy as np,torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv,counts,compose_mixture
from wall_panel_rl.rl.episode_imitation import episode_record
from wall_panel_rl.rl.predictive_safety import PredictiveSafetyController
from run_targeted_bc import read,dump,sha

WORK={}

def initialize(spec):
    torch.set_num_threads(1);p=read(ROOT/spec['preparation_report']/'protocol.json');model=PPO.load(ROOT/spec['source_model'],device='cpu')
    WORK.update(spec=spec,config=p['configuration'],settings=p['evaluation_settings'],model=model,
        weights=copy.deepcopy(model.policy.state_dict()),counter=model.num_timesteps,updates=model._n_updates)

def episode(job):
    mode,horizon,case=job;spec=WORK['spec'];env=HybridFourEnv(copy.deepcopy(WORK['config']),[],copy.deepcopy(WORK['settings']));start=time.perf_counter()
    try:
        if case['task']!='dynamic' or not np.allclose(case['strength'],spec['motion']):raise ValueError('Fixed stronger dynamic target required')
        obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']});reference=env.cursor.path.copy();reward=0.
        controller=PredictiveSafetyController(WORK['model'],env,horizon=horizon,gains=spec['gains']) if mode=='predictive' else None
        while True:
            before=(env.env.steps,env.env.sim_time,env.used)
            action=controller.action(obs) if controller else WORK['model'].predict(obs,deterministic=True)[0]
            assert before==(env.env.steps,env.env.sim_time,env.used),'Prediction committed virtual steps'
            obs,r,done,truncated,info=env.step(action);reward+=r
            if done or truncated:break
        assert np.array_equal(reference,env.cursor.path)
        row=episode_record(env,info,'ppo',reward)
        row.update(execution_controller='ppo_with_predictive_safety' if controller else 'pure_ppo',online_avoidance_planner=bool(controller),
            uses_global_rrt=False,weights_changed=False,prediction_horizon=int(horizon) if controller else 0,
            interventions=controller.interventions if controller else 0,no_safe_action=controller.no_safe_action if controller else 0,
            discarded_prediction_steps=controller.prediction_steps if controller else 0,prediction_wall_s=controller.seconds if controller else 0.,
            gain_counts=controller.gain_counts if controller else {},physics_elapsed_s=env.env.sim_time,
            total_wall_s=time.perf_counter()-start,reference_index=int(env.cursor.index),reference_last=len(reference)-1)
        assert WORK['model'].num_timesteps==WORK['counter'] and WORK['model']._n_updates==WORK['updates']
        assert all(torch.equal(v,WORK['model'].policy.state_dict()[k]) for k,v in WORK['weights'].items())
        return row
    finally:env.close()


def preserved_successes(before,after):
    assert [r['seed'] for r in before]==[r['seed'] for r in after]
    for a,b in zip(before,after):
        if a['reason']=='success':
            assert b['reason']=='success' and a['steps']==b['steps'],'Prediction lost an original success'
            assert b['interventions']==0 and abs(a['position_m']-b['position_m'])<=1e-6 and abs(a['orientation_deg']-b['orientation_deg'])<=1e-6
    return True


def run(spec,config_file):
    report=ROOT/spec['report_root']
    if report.exists():raise FileExistsError('Refusing safety study overwrite')
    p=read(ROOT/spec['preparation_report']/'protocol.json')
    for f,h in p['input_hashes'].items():assert sha(ROOT/f)==h
    original=read(ROOT/spec['source_report']/'candidate_manifest.json');source_hash=sha(ROOT/spec['source_model']);assert source_hash==original['model_sha256']
    report.mkdir(parents=True);visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    state={'status':'running','optimizer_updates':0,'static_episodes':0,'controller':'PPO + short-horizon predictive safety'}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra);dump(report/'status.json',state);print('PREDICTIVE_SAFETY',json.dumps(state),flush=True)
    files=[str(config_file.relative_to(ROOT)),'scripts/evaluate_predictive_safety.py','src/wall_panel_rl/rl/predictive_safety.py',
        'src/wall_panel_rl/rl/dynamic_four_training.py','src/wall_panel_rl/rl/conditional_recovery.py','src/wall_panel_rl/rl/avoidance_curriculum.py',
        'src/wall_panel_rl/rl/waypoint_ppo.py','src/wall_panel_rl/envs/wall_panel_env.py','src/wall_panel_rl/control/joint_controller.py']
    hashes={f:sha(ROOT/f) for f in files};inputs=dict(p['input_hashes']);inputs[spec['source_model']]=source_hash
    for f in ['results/diagnostics/four_staggered_dynamic_bc_v11_seed91/training_diagnosis.json',
              'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation/dynamic_0.120_0.json',
              'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation_expert_cache.json']:
        inputs[f]=sha(ROOT/f)
    diagnosis=read(ROOT/'results/diagnostics/four_staggered_dynamic_bc_v11_seed91/training_diagnosis.json')
    train=read(ROOT/spec['data_root']/'train_cases.json');by_seed={c['scene']['seed']:c for c in train};rng=np.random.default_rng(spec['seed'])
    selected=[]
    for reason,key in [('success','train_tuning_success_cases'),('collision','train_tuning_collision_cases')]:
        candidates=[r['seed'] for r in diagnosis if r['reason']==reason];rng.shuffle(candidates);selected.extend(candidates[:spec[key]])
    tuning=[by_seed[s] for s in selected];dump(report/'tuning_train_seeds.json',selected)
    dump(report/'protocol.json',{'spec':spec,'source_sha256':source_hash,'configuration':p['configuration'],'evaluation_settings':p['evaluation_settings'],
        'code_hashes':hashes,'input_hashes':inputs,'tuning_train_seeds':selected,'parameter_selection':'Train subset only: success first, then fewer collisions/timeouts, then smaller horizon; fixed before validation',
        'validation_gate':'Strict success gain, collision non-increase, zero joint violations, every original success unchanged',
        'test_rule':'Only after validation gate; lock all parameters before one paired source vs safety evaluation on reserved100 cases; no tuning afterwards',
        'comparison_label':'PPO + predictive execution controller, not pure PPO; model weights unchanged; known dynamics; virtual simulation time discarded',
        'no_global_rrt_fallback':True,'optimizer_updates':0})
    for f in files:
        dest=report/'code_archive'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    def evaluate(pool,jobs,phase):
        rows=[];status(phase,processed=0,total=len(jobs))
        for row in pool.map(episode,jobs):
            rows.append(row)
            if len(rows)%4==0 or len(rows)==len(jobs):status(phase,processed=len(rows),total=len(jobs),outcomes=counts(rows))
        return rows
    try:
        with ProcessPoolExecutor(max_workers=spec['workers'],mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(spec,)) as pool:
            base_train=evaluate(pool,[('pure',0,c) for c in tuning],'tuning_source_train');dump(report/'tuning_source.json',{'rows':base_train,'metrics':counts(base_train)})
            candidates=[]
            for horizon in spec['horizons']:
                rows=evaluate(pool,[('predictive',horizon,c) for c in tuning],f'tuning_horizon_{horizon}');preserved_successes(base_train,rows)
                m=counts(rows);value=[m['success'],-m['collision'],-m['timeout'],-horizon];candidates.append({'horizon':horizon,'metrics':m,'score':value})
                dump(report/f'tuning_horizon_{horizon}.json',{'rows':rows,'metrics':m})
            chosen=max(candidates,key=lambda r:r['score']);dump(report/'selection.json',{'selected':chosen,'all_train_candidates':candidates,'locked_before_validation':True})
            horizon=chosen['horizon'];validation=read(ROOT/spec['data_root']/'validation_cases.json')
            source=read(ROOT/'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation/dynamic_0.120_0.json')
            rows=evaluate(pool,[('predictive',horizon,c) for c in validation],'validation_selected_controller');preserved_successes(source['rows'],rows)
            safe={'rows':rows,'metrics':counts(rows),'mode':'ppo_with_predictive_safety','pure_ppo':False};dump(report/'validation_safety.json',safe);dump(report/'validation_source.json',source)
            cache=read(ROOT/'results/diagnostics/four_staggered_dynamic_ppo_v11_seed91/validation_expert_cache.json');mixed=compose_mixture(safe,cache);source_mixed=compose_mixture(source,cache)
            dump(report/'validation_mixture_safety.json',mixed);dump(report/'validation_mixture_source.json',source_mixed)
            a=source['metrics'];b=safe['metrics'];gate=b['success']>a['success'] and b['collision']<=a['collision'] and b['joint_limit']==0
            dump(report/'validation_gate.json',{'passed':gate,'source':a,'safety':b,'original_successes_preserved':True,'horizon':horizon,'locked_before_test':True})
            test={'performed':False,'reason':'Validation gate not passed; reserved bank remains unused'}
            if gate and spec['evaluate_independent_test_after_validation_gain']:
                tests=read(ROOT/spec['data_root']/'test_cases.json')
                before=evaluate(pool,[('pure',0,c) for c in tests],'independent_source_test')
                after=evaluate(pool,[('predictive',horizon,c) for c in tests],'independent_safety_test');preserved_successes(before,after)
                test={'performed':True,'source':counts(before),'safety':counts(after),'paired':dict(Counter(a['reason']+' -> '+b['reason'] for a,b in zip(before,after))),'no_tuning_after_test':True,'original_successes_preserved':True}
                dump(report/'independent_source.json',{'rows':before,'metrics':counts(before),'independent_test':True});dump(report/'independent_safety.json',{'rows':after,'metrics':counts(after),'independent_test':True})
            dump(report/'independent_test.json',test)
        for f,h in hashes.items():assert sha(ROOT/f)==h,'Runtime code changed'
        for f,h in inputs.items():assert sha(ROOT/f)==h,'Frozen input changed'
        summary={'selected_horizon':horizon,'train_candidates':candidates,'validation_source':a,'validation_safety':b,
            'validation_mixture_source':source_mixed['metrics'],'validation_mixture_safety':mixed['metrics'],'validation_gate_passed':gate,
            'validation_original_successes_preserved':True,'independent_test':test,'model_weights_changed':False,'source_unchanged':sha(ROOT/spec['source_model'])==source_hash,
            'optimizer_updates':0,'controller_is_pure_ppo':False,'global_rrt_fallback':False,'default_pure_ppo_preserved':True,
            'validation_prediction_steps':sum(r['discarded_prediction_steps'] for r in rows),'validation_interventions':sum(r['interventions'] for r in rows)}
        dump(report/'summary.json',summary);status('complete',status='completed',summary=summary)
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'configs/four_predictive_safety.yaml');parser.add_argument('--run',action='store_true');args=parser.parse_args();spec=yaml.safe_load(args.config.read_text())
    print({'mode':'run' if args.run else 'check_only','controller':'PPO+predictive safety','pure_ppo':False},flush=True)
    if args.run:run(spec,args.config.resolve())

if __name__=='__main__':main()
