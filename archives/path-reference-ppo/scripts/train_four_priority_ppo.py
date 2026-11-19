"""Short pure PPO adaptation, four-box success selection, paired held-out test."""
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
from stable_baselines3.common.vec_env import DummyVecEnv
from run_targeted_bc import read, dump, sha
from wall_panel_rl.rl.conditional_recovery import ConditionalRecoveryEnv
from wall_panel_rl.rl.episode_imitation import evaluate_controller
from wall_panel_rl.rl.four_priority_bc import score
from wall_panel_rl.rl.four_priority_ppo import FourPolicyTrainingEnv
from wall_panel_rl.rl.multi_obstacle_adapter import CRITIC_PREFIXES


def run(spec):
    torch.set_num_threads(1)
    root, report = Path(spec['model_root']), Path(spec['report_root'])
    if root.exists() or report.exists(): raise FileExistsError('Refusing PPO overwrite')
    tp = read(Path(spec['teacher_report']) / 'protocol.json')
    data_root = Path(tp['spec']['data_root']); splits = read(Path(spec['teacher_report']) / 'split.json')
    config, settings = tp['configuration'], tp['evaluation_settings']
    train_cases = read(data_root / 'train_cases.json'); cases = read(data_root / 'validation_cases.json')
    source_summary = read(Path(spec['source_report']) / 'summary.json')
    assert sha(spec['source_model']) == source_summary['final_candidate_sha256']
    inputs = {str(p): sha(p) for p in [Path(spec['source_model']), data_root / 'train_cases.json',
              data_root / 'validation_cases.json', data_root / 'test_cases.json']}
    root.mkdir(parents=True); report.mkdir(parents=True)
    state = {'status': 'running', 'started_at': datetime.now().astimezone().isoformat(), 'bc_updates': 0}
    def status(phase, **extra):
        state.update(phase=phase, updated_at=datetime.now().astimezone().isoformat(), **extra)
        dump(report / 'status.json', state); print('FOUR_PPO', json.dumps(state), flush=True)
    def worker(rank):
        return FourPolicyTrainingEnv(ConditionalRecoveryEnv(copy.deepcopy(config), [], settings, rank),
            train_cases, excluded=splits['validation']+splits['test'], seed=spec['seed']+rank*1000,
            reward_scale=spec['reward_scale'], journal=root / f'episodes_worker_{rank}.jsonl')
    env = DummyVecEnv([lambda i=i: worker(i) for i in range(spec['workers'])])
    try:
        model = PPO.load(spec['source_model'], env=env, device='cpu', seed=spec['seed'],
            n_steps=spec['n_steps'], batch_size=spec['batch_size'], n_epochs=spec['n_epochs'],
            learning_rate=spec['learning_rate'], clip_range=spec['clip_range'], target_kl=spec['target_kl'],
            vf_coef=spec['vf_coef'], ent_coef=0.)
        original = copy.deepcopy(model.policy.state_dict())
        model.policy.configure_learning(train_critic=True)
        # Evaluation uses original rewards. Only training has a positive constant
        # scale; reset the value output to its new units, without changing actions.
        with torch.no_grad():
            model.policy.value_net.weight.zero_(); model.policy.value_net.bias.zero_()
        model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
            lr=spec['learning_rate'], **model.policy.optimizer_kwargs)
        start = model.num_timesteps
        code = ['scripts/train_four_priority_ppo.py', 'configs/four_priority_ppo.yaml',
                'src/wall_panel_rl/rl/four_priority_ppo.py', 'src/wall_panel_rl/rl/multi_obstacle_adapter.py',
                'src/wall_panel_rl/rl/four_priority_bc.py', 'src/wall_panel_rl/rl/episode_imitation.py']
        dump(report / 'protocol.json', {'spec': spec, 'configuration': config, 'evaluation_settings': settings,
             'input_hashes': inputs, 'code_hashes': {p: sha(p) for p in code}, 'source_ppo_counter': start,
             'train_seeds': splits['train'], 'validation_seeds': splits['validation'], 'test_seeds': splits['test'],
             'training_controller': 'pure PPO; no RRT episodes, no BC updates', 'two_used_for_selection': False,
             'reward_conditioning': 'training rewards multiplied by0.01; evaluation rewards unchanged; value output reset',
             'log_std_frozen': True, 'default_model_preserved': True})
        history = []; best = None; selected = None; baseline = None
        def evaluate(added):
            nonlocal baseline, best, selected
            rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
            try: result = evaluate_controller(model, config, settings, cases, mode='ppo')
            finally: torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
            metrics = result['metrics']['stage_target']
            if baseline is None:
                baseline = metrics
                assert metrics['success'] == 17 and metrics['collision'] == 15
            rank = score(metrics, baseline['collision']); event = {'additional_ppo_steps': added, 'four': metrics, 'score': rank}
            model.save(root / 'checkpoints' / f'ppo_{added}.zip')
            if best is None or tuple(rank) > best:
                best = tuple(rank); selected = copy.deepcopy(event); model.save(root / 'selected_model.zip')
            dump(report / 'validation' / f'steps_{added}.json', result)
            history.append(event); dump(report / 'history.json', history)
            status('ppo_validation', ppo_steps=added, four=metrics)
        class Validation(BaseCallback):
            def _on_training_start(self): evaluate(0)
            def _on_rollout_start(self):
                added = self.model.num_timesteps-start
                if added and added % spec['evaluate_every'] == 0: evaluate(added)
                status('ppo_training', ppo_steps=added, completed_episodes=sum(len(e.records) for e in env.envs))
            def _on_step(self): return True
            def _on_training_end(self): evaluate(self.model.num_timesteps-start)
        status('ppo_start', steps_budget=spec['steps'], train_layouts=len(train_cases))
        model.learn(spec['steps'], reset_num_timesteps=False, callback=Validation())
        assert model.num_timesteps-start == spec['steps']
        model.save(root / 'final_model.zip')
        chosen = PPO.load(root / 'selected_model.zip', device='cpu')
        fixed = all(torch.equal(v, chosen.policy.state_dict()[k]) for k, v in original.items()
                    if not k.startswith(('multi_adapter.', *CRITIC_PREFIXES)))
        assert fixed
        gained = tuple(selected['score']) > tuple(history[0]['score'])
        dump(report / 'selection_before_test.json', {**selected, 'model_sha256': sha(root / 'selected_model.zip')})
        before = after = None
        if gained:
            status('reserved_test', selected_ppo_steps=selected['additional_ppo_steps'])
            tests = read(data_root / 'test_cases.json'); source = PPO.load(spec['source_model'], device='cpu')
            before = evaluate_controller(source, config, settings, tests, mode='ppo')
            after = evaluate_controller(chosen, config, settings, tests, mode='ppo')
            before['independent_test'] = after['independent_test'] = True
            dump(report / 'test_source.json', before); dump(report / 'test_selected.json', after)
        for path, expected_hash in inputs.items(): assert sha(path) == expected_hash
        records = [r for e in env.envs for r in e.records]
        dump(report / 'summary.json', {'ppo_steps_run': spec['steps'], 'bc_updates': 0, 'source_validation': baseline,
            'selected': selected, 'validation_improved': gained, 'test_evaluated': after is not None,
            'test_source': before['metrics']['stage_target'] if before else None,
            'test_selected': after['metrics']['stage_target'] if after else None, 'test_used_for_selection': False,
            'training_episodes': dict(Counter(r['reason'] for r in records)), 'training_completed_transitions': sum(r['steps'] for r in records),
            'base_actor_and_std_unchanged': fixed, 'source_unchanged': True, 'default_model_preserved': True})
        shutil.copy2(root / 'selected_model.zip', report / 'selected_model.zip'); shutil.copy2(root / 'final_model.zip', report / 'final_candidate.zip')
        dump(report / 'candidate_manifest.json', {'configuration': config, 'evaluation_settings': settings,
             'model_sha256': sha(report / 'selected_model.zip'), 'selected_ppo_steps': selected['additional_ppo_steps']})
        for path in code:
            dest = report / 'code_archive' / path; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        status('complete', status='completed', ppo_steps=spec['steps'], selected=selected, test=after['metrics']['stage_target'] if after else None)
    finally: env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', default='configs/four_priority_ppo.yaml'); p.add_argument('--run', action='store_true')
    args = p.parse_args(); spec = yaml.safe_load(Path(args.config).read_text())
    if spec['steps'] % (spec['workers']*spec['n_steps']): raise ValueError('Whole PPO rollout budget required')
    print({'mode': 'run' if args.run else 'check_only', 'ppo_steps': spec['steps'], 'bc_updates': 0}, flush=True)
    if args.run: run(spec)


if __name__ == '__main__': main()
