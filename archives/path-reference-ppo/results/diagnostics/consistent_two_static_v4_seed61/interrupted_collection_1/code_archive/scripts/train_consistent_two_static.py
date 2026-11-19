"""PPO pilot with coverage guards and a frozen, canonical-only imitation pool."""
import _bootstrap
import copy
import json
from pathlib import Path
import shutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from train_episode_rrt_ppo import inputs
from train_two_static_ppo import RetainedOneValidation, SeededImitation
from run_targeted_bc import read, dump, sha
from run_consistent_two_static import coverage_score
from wall_panel_rl.rl.conditional_recovery import ConditionalTrainingEnv
from wall_panel_rl.rl.episode_imitation import ExpertEpisodeEnv, evaluate_controller
from wall_panel_rl.rl.avoidance_curriculum import gate_result


class CanonicalPoolEpisodes(ExpertEpisodeEnv):
    """Run normal 50/50 RRT/PPO episodes, but do not add arbitrary new RRT labels.

    PPO can explore the existing static phase augmentation. The auxiliary pool
    remains the pre-collected consistent-route pool; fresh RRT routes can have
    a different detour and are logged only as executed expert episodes.
    """
    def _record(self, row):
        if row['controller'] == 'rrt':
            row['accepted_for_imitation'] = False
            row['imitation_exclusion'] = 'auxiliary labels restricted to frozen canonical-route teacher'
        super()._record(row)


class CoverageValidation(RetainedOneValidation):
    def __init__(self, *args, baseline, legacy_cases, **kwargs):
        super().__init__(*args, **kwargs); self.baseline = baseline; self.legacy_cases = legacy_cases

    def score_model(self, candidate):
        result, rows = super().score_model(candidate)
        legacy = evaluate_controller(candidate, self.config, self.s, self.legacy_cases, mode='ppo')
        coverage = {'two': result['metrics']['dynamic_ppo'], 'one': result['one_static_metrics'],
            'legacy_two': legacy['metrics'], 'ppo_ready': result['passed']}
        coverage['eligible'] = result['passed'] and gate_result(coverage['two'], self.s['gate'])['passed'] and gate_result(coverage['legacy_two'], self.s['gate'])['passed']
        coverage['score'] = coverage_score(coverage, self.baseline)
        result['coverage'] = coverage
        dump(self.output/'legacy_two'/f'episodes_{candidate.num_timesteps-self.start}.json', legacy)
        return result, rows

    def candidate_score(self, result): return tuple(result['coverage']['score'])


def train(study, one_settings, one_cases, legacy_cases, baseline, teacher_path, heldout, status):
    training, evaluation, config, scenes, cases = inputs(study)
    root, report = Path(study['output_root']), Path(study['report_root'])
    if root.exists() or report.exists(): raise FileExistsError('Refusing PPO overwrite')
    ppo = root/'ppo'; ppo.mkdir(parents=True); report.mkdir(parents=True)
    dump(report/'manifest.json', {'study': study, 'configuration': config, 'training_settings': training,
        'evaluation_settings': evaluation, 'source_sha256': sha(study['initial_model']),
        'imitation_rule': 'frozen canonical-route train labels only; fresh online RRT labels excluded',
        'selection_rule': 'new validation and legacy validation non-regression, release bucket then collision rate',
        'heldout_excluded': sorted(heldout), 'train_scene_seeds': [s['seed'] for s in scenes]})
    def worker(rank):
        return CanonicalPoolEpisodes(ConditionalTrainingEnv(copy.deepcopy(config), scenes, training, rank),
            probability=study['expert_probability'], seed=study['seed']+rank*1000, journal=root/f'episodes_worker_{rank}.jsonl')
    env = DummyVecEnv([lambda i=i: worker(i) for i in range(training['workers'])])
    try:
        model = PPO.load(study['initial_model'], env=env, device='cpu', seed=study['seed'],
            n_steps=training['n_steps'], batch_size=training['batch_size'], n_epochs=training['n_epochs'],
            learning_rate=study['ppo_learning_rate'], clip_range=training['clip_range'], target_kl=training['target_kl'])
        model.policy.configure_learning(train_critic=True)
        model.policy.optimizer = model.policy.optimizer_class(model.policy.parameters(),
            lr=study['ppo_learning_rate'], **model.policy.optimizer_kwargs)
        start = model.num_timesteps
        val = CoverageValidation(ppo, None, config, evaluation, cases, status,
            one_settings=one_settings, one_cases=one_cases, baseline=baseline, legacy_cases=legacy_cases,
            probability=study['expert_probability'], selection_seed=study['seed'])
        aux = SeededImitation(root/'auxiliary_bc', study['auxiliary_bc'], study['seed'], teacher_path=teacher_path, heldout=heldout)
        model.learn(study['ppo_steps'], reset_num_timesteps=False, callback=[val, aux])
        if model.num_timesteps-start != study['ppo_steps']: raise AssertionError('Unexpected PPO budget')
        selection = read(ppo/'selection.json')
        shutil.copy2(ppo/'best_model.zip', report/'selected_model.zip')
        shutil.copy2(ppo/'final_model.zip', report/'final_candidate.zip')
        records = [r for e in env.envs for r in e.records]
        if any(r.get('accepted_for_imitation') for r in records if r['controller'] == 'rrt'):
            raise AssertionError('Noncanonical RRT labels entered the pool')
        dump(report/'summary.json', {'ppo_transitions': model.num_timesteps-start,
            'rrt_transitions': sum(r['steps'] for r in records if r['controller'] == 'rrt'),
            'completed_episodes': {name: {reason: sum(r['controller']==name and r['reason']==reason for r in records)
                for reason in ('success', 'collision', 'timeout', 'joint_limit', 'planning_failure')} for name in ('rrt', 'ppo')},
            'auxiliary_updates': aux.accepted, 'auxiliary_rejected': aux.rejected,
            'selection': selection, 'online_rrt_labels_excluded': True})
        status('ppo_complete', selected_ppo_steps=selection['additional_steps'])
        return report/'selected_model.zip'
    finally: env.close()
