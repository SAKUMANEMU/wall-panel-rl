"""Keep PPO optimizer/value state and prioritize hard TRAIN layouts safely."""
import hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from wall_panel_rl.rl.episode_imitation import evaluate_controller
from wall_panel_rl.rl.rrt_validation import counts


def state_digest(value):
    h = hashlib.sha256()
    def visit(item):
        h.update(type(item).__name__.encode())
        if torch.is_tensor(item):
            a = item.detach().cpu().contiguous().numpy()
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str): visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for x in item: visit(x)
        else: h.update(repr(item).encode())
    visit(value); return h.hexdigest()


def load_continuation(path, env, spec):
    reference = PPO.load(path, device='cpu')
    expected_optimizer = state_digest(reference.policy.optimizer.state_dict())
    model = PPO.load(path, env=env, device='cpu', seed=spec['seed'],
        n_steps=spec['n_steps'], batch_size=spec['batch_size'], n_epochs=spec['n_epochs'],
        learning_rate=spec['learning_rate'], clip_range=spec['clip_range'], target_kl=spec['target_kl'], vf_coef=spec['vf_coef'])
    model.policy.configure_learning(train_critic=True)
    actual_optimizer = state_digest(model.policy.optimizer.state_dict())
    if actual_optimizer != expected_optimizer: raise AssertionError('PPO continuation lost optimizer state')
    if not all(torch.equal(v, model.policy.state_dict()[k]) for k,v in reference.policy.state_dict().items()):
        raise AssertionError('PPO continuation changed source weights')
    return model, {'optimizer_state_preserved':True, 'optimizer_sha256':actual_optimizer,
                   'value_network_preserved':True, 'source_counter':model.num_timesteps}


def hard_probabilities(cases, hard_seeds, fraction=.25):
    seeds = [int(c['scene']['seed']) for c in cases]; hard = set(map(int, hard_seeds))
    if not cases or not 0 <= fraction <= 1 or not hard.issubset(set(seeds)): raise ValueError('Invalid hard TRAIN layout pool')
    if not hard: return np.full(len(cases), 1/len(cases))
    return np.full(len(cases), (1-fraction)/len(cases)) + np.asarray([fraction/len(hard) if s in hard else 0. for s in seeds])


def stochastic_evaluation(model, config, settings, cases, seed_offset):
    """Final model only; reset action-noise RNG separately for every scene."""
    class Stochastic:
        def predict(self, observation, deterministic=True): return model.predict(observation, deterministic=False)
    saved = torch.get_rng_state(); rows=[]
    try:
        for case in cases:
            torch.manual_seed(int(case['scene']['seed'])+seed_offset)
            rows.extend(evaluate_controller(Stochastic(), config, settings, [case], mode='ppo')['rows'])
    finally: torch.set_rng_state(saved)
    return {'mode':'stochastic_policy','rows':rows,'metrics':{'stage_target':counts(rows)},
            'per_scene_seed_offset':seed_offset,'training_performed':False}
