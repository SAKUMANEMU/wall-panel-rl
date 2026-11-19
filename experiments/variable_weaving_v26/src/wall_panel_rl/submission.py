"""Portable entry points for the Learning Agent C&R submission.

These adapters reuse the evaluated environment and controller unchanged. Only
dataset selection, output paths, and the short training interface are new.
"""
import copy
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.slow_dynamic_four import planning_snapshot, assert_planning_paused
from wall_panel_rl.rl.variable_weaving import VariableDynamicEnv
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController
from wall_panel_rl.rl.episode_imitation import make_rrt
from wall_panel_rl.visualization.episode_replay import capture_frame, recording_payload
from wall_panel_rl.visualization.passage_metrics import passage_measurements


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_runtime(path=None):
    return load_json(path or ROOT / 'configs/runtime.json')


def load_agent(path=None, **kwargs):
    """Local module names are retained so SB3 can restore the custom policy."""
    return PPO.load(path or ROOT / 'models/current_ppo.zip', device='cpu', **kwargs)


class CaseBankEnv(gym.Env):
    """Gymnasium training adapter over an explicitly chosen, frozen case bank.

    One reset chooses one case; actions always come from the learning agent.
    Expert/recovery actions are never injected into PPO's on-policy buffer.
    Reward scaling is applied only after the original environment reward.
    """
    metadata = {'render_modes': [], 'render_fps': 20}

    def __init__(self, bank, runtime=None, reward_scale=1.0):
        super().__init__()
        self.runtime = copy.deepcopy(runtime or load_runtime())
        self.cases = load_json(bank) if isinstance(bank, (str, Path)) else copy.deepcopy(bank)
        if not self.cases or len({c['scene']['seed'] for c in self.cases}) != len(self.cases):
            raise ValueError('The case bank must be nonempty and have unique seeds')
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError('Reward scale must be positive and finite')
        self.reward_scale = float(reward_scale)
        self.core = VariableDynamicEnv(self.runtime['configuration'], [], self.runtime['settings'])
        self.action_space = self.core.action_space
        self.observation_space = self.core.observation_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'case_index' in options:
            index = int(options['case_index'])
            if not 0 <= index < len(self.cases):
                raise ValueError('Case index is outside the bank')
        else:
            index = int(self.np_random.integers(len(self.cases)))
        case = self.cases[index]
        observation, info = self.core.reset(options={
            'scenario': case['scene'], 'task': case['task'], 'strength': case['strength']})
        return observation, dict(info, scene_seed=int(case['scene']['seed']))

    def step(self, action):
        observation, reward, terminated, truncated, info = self.core.step(action)
        return observation, float(reward * self.reward_scale), terminated, truncated, dict(info, raw_reward=reward)

    def close(self):
        self.core.close()


def run_episode(model, case, mode='enhanced', runtime=None):
    """Execute one real episode; prediction trials are rolled back by the controller.

    'ppo' is the actor alone with the shared environment wrapper. 'enhanced'
    adds the evaluated predictive execution/recovery layer. 'rrt' is a separate
    time-aware planner baseline, never a fallback for either PPO mode.
    """
    if mode not in ('ppo', 'enhanced', 'rrt'):
        raise ValueError('Expected ppo, enhanced, or rrt')
    runtime = copy.deepcopy(runtime or load_runtime())
    env = VariableDynamicEnv(runtime['configuration'], [], runtime['settings'])
    started = time.perf_counter()
    try:
        observation, info = env.reset(options={
            'scenario': case['scene'], 'task': case['task'], 'strength': case['strength']})
        reference = env.cursor.path.copy()
        frames = [capture_frame(env.env)]
        before = planning_snapshot(env)
        expert = make_rrt(env, int(case['scene']['seed']) + 100000) if mode == 'rrt' else None
        controller = None
        if mode == 'enhanced':
            spec = runtime['controller']
            controller_class = ProgressRecoveryController
            if spec.get('recovery_strategy') == 'diverse_v23':
                from wall_panel_rl.rl.expanded_recovery import DiverseRecoveryController
                controller_class = DiverseRecoveryController
            elif spec.get('recovery_strategy') == 'timeout_v24':
                from wall_panel_rl.rl.timeout_recovery import TimeoutRecoveryController
                controller_class = TimeoutRecoveryController
            controller = controller_class(model, env, spec['selected']['horizon'],
                                                      spec['gains'], spec['recovery'])
        assert_planning_paused(env, before)
        reason = 'planning_failure' if expert is not None and not expert.planning_success else None
        total_reward, zero_commands, stationary = 0.0, 0, 0
        while reason is None:
            before = planning_snapshot(env)
            previous_q = env.env.joint_state()[0].copy()
            if expert is not None:
                action = expert.action(observation)
            elif controller is not None:
                action = controller.action(observation)
            else:
                action = model.predict(observation, deterministic=True)[0]
            assert_planning_paused(env, before)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            frames.append(capture_frame(env.env))
            zero_commands += int(np.max(np.abs(action)) < 1e-8)
            stationary += int(np.max(np.abs(env.env.joint_state()[0] - previous_q)) < 1e-4)
            if terminated or truncated:
                reason = info['termination_reason']
        assert np.array_equal(reference, env.cursor.path)
        result = {
            'seed': int(case['scene']['seed']), 'controller': mode, 'reason': reason,
            'steps': int(env.env.steps), 'position_m': float(info['position_error']),
            'orientation_deg': float(np.rad2deg(info['orientation_error'])),
            'reward': float(total_reward), 'zero_command_steps': zero_commands,
            'stationary_steps': stationary, 'wall_seconds': time.perf_counter() - started,
            'physics_elapsed_s': env.env.sim_time, 'weights_changed': False,
            'uses_global_rrt': mode == 'rrt', 'reference_preserved': True,
            'local_rescues': controller.rescues if controller else 0,
            'recovery_actions': controller.recovery_actions if controller else 0,
            'recovery_events': copy.deepcopy(controller.recovery_events) if controller else [],
            'no_safe_action_calls': controller.no_safe_action if controller else 0,
            'prediction_steps': controller.prediction_steps if controller else 0,
        }
        payload = recording_payload(runtime['configuration'], case, result, frames)
        measures = passage_measurements(payload)
        result['passage_count'] = measures['passage_count']
        result['success_with_weaving'] = reason == 'success' and measures['passage_count'] >= 3
        payload['result'] = copy.deepcopy(result)
        payload['passage_measurements'] = measures
        return result, payload
    finally:
        env.close()
