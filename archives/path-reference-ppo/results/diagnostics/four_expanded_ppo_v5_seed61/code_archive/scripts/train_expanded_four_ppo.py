"""20k PPO continuation on192 train layouts, fresh validation/test, noise audit."""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import random
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from run_targeted_bc import read, dump, sha
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import evaluate_controller
from wall_panel_rl.rl.four_priority_bc import score
from wall_panel_rl.rl.four_priority_ppo import FourPolicyTrainingEnv
from wall_panel_rl.rl.four_ppo_continuation import load_continuation, hard_probabilities, stochastic_evaluation
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES


def run(spec):
    torch.set_num_threads(1)
    root, report = Path(spec['model_root']), Path(spec['report_root'])
    prep, data = Path(spec['preparation_report']), Path(spec['data_root'])
    if root.exists() or report.exists(): raise FileExistsError('Refusing continuation overwrite')
    if read(prep/'status.json')['status'] != 'completed': raise RuntimeError('Scene bank incomplete')
    pp = read(prep/'protocol.json'); config, settings = pp['configuration'], pp['evaluation_settings']
    for path, expected in read(prep/'data_hashes.json').items(): assert sha(path)==expected
    source_report = Path(spec['source_report']); sp = read(source_report/'protocol.json')
    assert sha(spec['source_model'])==read(source_report/'candidate_manifest.json')['model_sha256']
    assert spec['reward_scale']==sp['spec']['reward_scale']
    splits = read(data/'split.json')
    names = ['train','validation','test','old_validation','old_test_regression']
    assert all(not set(splits[a])&set(splits[b]) for a in names for b in names if a!=b)
    train_cases = read(data/'train_cases.json'); cases = read(data/'validation_cases.json')
    old_validation = read(data/'old_validation_cases.json')
    root.mkdir(parents=True); report.mkdir(parents=True)
    state = {'status':'running','started_at':datetime.now().astimezone().isoformat(),'bc_updates':0}
    def status(phase, **extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json', state); print('EXPANDED_PPO',json.dumps(state),flush=True)
    source = PPO.load(spec['source_model'], device='cpu')
    status('mine_training_failures', train_layouts=len(train_cases))
    mine = evaluate_controller(source,config,settings,train_cases,mode='ppo')
    dump(report/'source_training_layout_scan.json',mine)
    hard = [r['seed'] for r in mine['rows'] if r['reason']!='success']
    probabilities = hard_probabilities(train_cases,hard,spec['hard_training_fraction'])
    def worker(rank):
        return FourPolicyTrainingEnv(ConditionalRecoveryEnv(copy.deepcopy(config),[],settings,rank), train_cases,
            excluded=splits['validation']+splits['test']+splits['old_validation']+splits['old_test_regression'],
            seed=spec['seed']+rank*1000,reward_scale=spec['reward_scale'],
            journal=root/f'episodes_worker_{rank}.jsonl',sampling_probabilities=probabilities)
    env = DummyVecEnv([lambda i=i:worker(i) for i in range(spec['workers'])])
    try:
        model, continuation = load_continuation(spec['source_model'],env,spec)
        original = copy.deepcopy(model.policy.state_dict()); start = model.num_timesteps
        model.set_logger(configure(str(root/'training'),['csv']))
        code = ['scripts/train_expanded_four_ppo.py','configs/four_expanded_ppo.yaml',
            'src/wall_panel_rl/rl/four_ppo_continuation.py','src/wall_panel_rl/rl/four_priority_ppo.py',
            'src/wall_panel_rl/rl/multi_obstacle_adapter.py','src/wall_panel_rl/rl/conditional_recovery.py']
        inputs = {str(p):sha(p) for p in [Path(spec['source_model']),*(data/f'{name}_cases.json' for name in names)]}
        dump(report/'protocol.json', {'spec':spec,'configuration':config,'evaluation_settings':settings,
            'continuation':continuation,'source_sha256':sha(spec['source_model']),
            'code_hashes':{p:sha(p) for p in code},'input_hashes':inputs,'splits':splits,
            'hard_training_seeds':hard,'sampling_probabilities':probabilities.tolist(),
            'selection':'fresh64 validation success first, collisions not above baseline, old32 tie-break only; old64 not used',
            'reward_scale_preserved':True,'critic_not_reset':True,'optimizer_not_reset':True,
            'test_and_noise_conditions_predeclared':True,'default_model_preserved':True})
        history=[]; baseline=None; best=None; selected=None
        def evaluate(added):
            nonlocal baseline,best,selected
            rng=(torch.get_rng_state(),np.random.get_state(),random.getstate())
            try:
                result=evaluate_controller(model,config,settings,cases,mode='ppo')
                old=evaluate_controller(model,config,settings,old_validation,mode='ppo')
            finally: torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2])
            metrics=result['metrics']['stage_target']; regression=old['metrics']['stage_target']
            if baseline is None:
                baseline=metrics; assert regression==read(source_report/'summary.json')['selected']['four']
            rank=score(metrics,baseline['collision'])+[regression['success'],-regression['collision']]
            event={'additional_ppo_steps':added,'new_validation':metrics,'old_validation':regression,'score':rank}
            model.save(root/'checkpoints'/f'ppo_plus_{added}.zip')
            if best is None or tuple(rank)>best:
                best=tuple(rank);selected=copy.deepcopy(event);model.save(root/'selected_model.zip')
            dump(report/'validation'/f'steps_{added}.json',result);dump(report/'old_validation'/f'steps_{added}.json',old)
            history.append(event);dump(report/'history.json',history)
            status('validation',ppo_steps=added,new_validation=metrics,old_validation=regression)
        class Validation(BaseCallback):
            def _on_training_start(self): evaluate(0)
            def _on_rollout_start(self):
                added=self.model.num_timesteps-start
                if added and added%spec['evaluate_every']==0:evaluate(added)
                status('training',ppo_steps=added,completed_episodes=sum(len(e.records) for e in env.envs))
            def _on_step(self):return True
            def _on_training_end(self):evaluate(self.model.num_timesteps-start)
        status('ppo_start',ppo_steps=0,steps_budget=spec['steps'],hard_train_layouts=len(hard))
        model.learn(spec['steps'],reset_num_timesteps=False,callback=Validation())
        assert model.num_timesteps-start==spec['steps']
        model.save(root/'final_model.zip'); chosen=PPO.load(root/'selected_model.zip',device='cpu')
        fixed=all(torch.equal(v,chosen.policy.state_dict()[k]) for k,v in original.items() if not k.startswith(('multi_adapter.',*CRITIC_PREFIXES)))
        assert fixed
        dump(report/'selection_before_test.json',{**selected,'sha256':sha(root/'selected_model.zip')})
        status('fresh_paired_test',selected_additional_steps=selected['additional_ppo_steps'])
        tests=read(data/'test_cases.json')
        before=evaluate_controller(source,config,settings,tests,mode='ppo')
        after=evaluate_controller(chosen,config,settings,tests,mode='ppo')
        before['independent_test']=after['independent_test']=True
        dump(report/'test_source.json',before);dump(report/'test_selected.json',after)
        regression=evaluate_controller(chosen,config,settings,read(data/'old_test_regression_cases.json'),mode='ppo')
        regression['independent_test']=False;dump(report/'old64_regression.json',regression)
        status('sensor_noise_comparison',clean=after['metrics']['stage_target'])
        noisy_config=copy.deepcopy(config);noisy_config['noise']=copy.deepcopy(spec['noise_comparison'])
        noisy=evaluate_controller(chosen,noisy_config,settings,tests,mode='ppo');dump(report/'test_sensor_noise.json',noisy)
        status('action_noise_comparison',sensor_noise=noisy['metrics']['stage_target'])
        stochastic=stochastic_evaluation(chosen,config,settings,tests,spec['stochastic_seed_offset'])
        dump(report/'test_action_noise.json',stochastic)
        records=[r for e in env.envs for r in e.records]
        for path,expected in inputs.items():assert sha(path)==expected
        summary={'ppo_steps_run':spec['steps'],'selected':selected,'baseline_new_validation':baseline,
            'test_source':before['metrics']['stage_target'],'test_selected':after['metrics']['stage_target'],
            'sensor_noise':noisy['metrics']['stage_target'],'action_noise':stochastic['metrics']['stage_target'],
            'old64_source':read(source_report/'summary.json')['test_selected'],
            'old64_selected':regression['metrics']['stage_target'],
            'training_episodes':dict(Counter(r['reason'] for r in records)),
            'training_completed_transitions':sum(r['steps'] for r in records),'hard_training_layouts':len(hard),
            'continuation':continuation,'base_actor_and_std_unchanged':fixed,'source_unchanged':True,
            'test_used_for_selection':False,'old64_used_for_selection':False,'bc_updates':0,'default_model_preserved':True}
        dump(report/'summary.json',summary)
        shutil.copy2(root/'selected_model.zip',report/'selected_model.zip');shutil.copy2(root/'final_model.zip',report/'final_candidate.zip')
        dump(report/'candidate_manifest.json',{'configuration':config,'evaluation_settings':settings,
            'model_sha256':sha(report/'selected_model.zip'),'selected_additional_ppo_steps':selected['additional_ppo_steps'],
            'four_stage_total_ppo_steps':10000+selected['additional_ppo_steps']})
        for path in code:
            dest=report/'code_archive'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        status('complete',status='completed',ppo_steps=spec['steps'],selected=selected,
               test=summary['test_selected'],sensor_noise=summary['sensor_noise'],action_noise=summary['action_noise'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise
    finally:env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_expanded_ppo.yaml');p.add_argument('--run',action='store_true')
    args=p.parse_args();spec=yaml.safe_load(Path(args.config).read_text())
    if spec['steps']%(spec['workers']*spec['n_steps']) or spec['evaluate_every']%(spec['workers']*spec['n_steps']):
        raise ValueError('Whole rollouts required')
    print({'mode':'run' if args.run else 'check_only','ppo_steps':spec['steps'],'train':192,'fresh_validation':64,'fresh_test':100},flush=True)
    if args.run:run(spec)


if __name__=='__main__':main()
