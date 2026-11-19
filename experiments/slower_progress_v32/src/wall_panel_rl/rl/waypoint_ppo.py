"""Static RRT-guided PPO: raw sampled actions, one shared actuator projection."""
import copy
import json
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.baselines.classical import ClassicalController
from wall_panel_rl.rl.waypoint_tracking import (
    WaypointTrackingPolicy, RouteCursor, reference_observation, bounded_action)
from wall_panel_rl.utils.config import load_config


class RawWaypointPolicy(WaypointTrackingPolicy):
    """The environment projects actions; forward/predict both use the raw distribution.

    SB3 stores the ORIGINAL Gaussian action/log probability in its rollout buffer.
    Projection is part of environment dynamics, not a replacement probability law.
    The mask is reapplied on construction, including checkpoint reloads.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name in ('tracking_gain_raw', 'tracking_bias')
                or name.startswith('mlp_extractor.value_net.') or name.startswith('value_net.'))

    def _predict(self, observation, deterministic=False):
        return self.get_distribution(observation).get_actions(deterministic=deterministic)


def load_for_finetuning(path, settings):
    model = PPO.load(path, device='cpu', custom_objects={'policy_class': RawWaypointPolicy},
        seed=settings['seed'], n_envs=settings['workers'], n_steps=settings['n_steps'],
        batch_size=settings['batch_size'], learning_rate=settings['learning_rate'],
        n_epochs=settings['n_epochs'], clip_range=settings['clip_range'],
        target_kl=settings['target_kl'], ent_coef=0.)
    with torch.no_grad():
        model.policy.log_std.fill_(np.log(settings['action_std']))
    return model


class WaypointPPOEnv(gym.Env):
    """77 observations; planner failure ends on the next step without robot motion."""
    def __init__(self, config, seed=42, margin=.01, progress_weight=2.,
                 train_seed_range=(300000,400000)):
        if config['obstacles'].get('dynamic', {}).get('enabled', False):
            raise ValueError('This wrapper supports static obstacles only')
        self.env = WallPanelEnv(config)
        if self.env.observation_space.shape != (70,):
            self.env.close()
            raise ValueError('Expected underlying 70-observation environment')
        self.action_space = self.env.action_space
        self.observation_space = spaces.Box(-np.inf, np.inf, (77,), np.float32)
        self.rng = np.random.default_rng(seed)
        self.margin, self.progress_weight = margin, progress_weight
        self.train_seed_range = train_seed_range
        self.finished = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}
        scene_seed = int(options.get('scene_seed', self.rng.integers(*self.train_seed_range)))
        underlying = {'noise_seed': scene_seed+1000000}
        if 'scenario' in options:
            underlying['scenario'] = options['scenario']
        obs, info = self.env.reset(seed=scene_seed, options=underlying)
        planner = ClassicalController(self.env, scene_seed)
        self.planning_success = bool(planner.planning_success)
        self.planning_seconds = float(planner.planning_time)
        self.cursor = RouteCursor(planner.path, self.env.config['rrt']['waypoint_tolerance']) if self.planning_success else None
        q, _ = self.env.joint_state()
        self.ref = self.cursor.reference(q) if self.cursor else q.copy()
        self.finished = False
        info.update(planning_success=self.planning_success, scene_seed=scene_seed)
        return reference_observation(obs, self.ref, self.env.lower, self.env.upper), info

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset required after episode termination')
        if not self.planning_success:
            self.finished = True
            info = self.env.info()
            info.update(success=False, is_success=False, planning_failure=True,
                        termination_reason='planning_failure', reward_waypoint_progress=0.,
                        executed_action=np.zeros(7).tolist(), physics_steps=0)
            obs = reference_observation(self.env.observation(), self.ref, self.env.lower, self.env.upper)
            return obs, 0., True, False, info
        q, _ = self.env.joint_state()
        before = np.linalg.norm(self.ref-q)
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Action must be seven finite values')
        executed = bounded_action(q, np.clip(action,-1,1), self.env.lower, self.env.upper,
                                  self.env.max_increment, self.margin)
        obs, base_reward, done, truncated, info = self.env.step(executed)
        q, _ = self.env.joint_state()
        # SAME waypoint on both sides; advance cursor only after computing progress.
        progress = self.progress_weight*(before-np.linalg.norm(self.ref-q))/self.env.max_increment
        info.update(reward_base=float(base_reward), reward_waypoint_progress=float(progress),
                    executed_action=executed.tolist(), planning_failure=False,
                    termination_reason=('success' if info['success'] else 'collision' if info['collision']
                        else 'joint_limit' if info['joint_limit_violation'] else 'timeout' if truncated else 'running'))
        self.ref = self.cursor.reference(q)
        self.finished = done or truncated
        return reference_observation(obs,self.ref,self.env.lower,self.env.upper), float(base_reward+progress), done, truncated, info

    def close(self):
        self.env.close()


def compare_policies(candidate, baseline, manifest, release, progress_weight=2.):
    """One plan per scene, identical replay for ordinary/BC/PPO; no test seed bank."""
    import pybullet as p
    rows=[]
    for case in manifest:
        config=load_config(release/case['config'])
        wrapper=WaypointPPOEnv(config, progress_weight=progress_weight)
        try:
            scene=json.loads((release/case['scene']).read_text())
            wrapper.reset(options={'scenario':scene,'scene_seed':case['seed']})
            env=wrapper.env
            state=p.saveState(physicsClientId=env.client)
            attrs={k:copy.deepcopy(getattr(env,k)) for k in ('steps','sim_time','previous_error',
                'joint_path_length','cartesian_path_length','observed_target')}
            noise=copy.deepcopy(env.noise_rng.bit_generator.state)
            cursor=copy.deepcopy(wrapper.cursor); ref=wrapper.ref.copy()
            for method,model in [('ordinary',None),('bc',baseline),('ppo',candidate)]:
                p.restoreState(stateId=state,physicsClientId=env.client)
                for key,value in attrs.items():setattr(env,key,copy.deepcopy(value))
                env.noise_rng.bit_generator.state=copy.deepcopy(noise)
                wrapper.cursor=copy.deepcopy(cursor);wrapper.ref=ref.copy();wrapper.finished=False
                obs=reference_observation(env.observation(),ref,env.lower,env.upper)
                total=0.
                while True:
                    q,_=env.joint_state()
                    action=np.clip((wrapper.ref-q)/env.max_increment,-1,1) if model is None else model.predict(obs,deterministic=True)[0]
                    obs,reward,done,truncated,info=wrapper.step(action)
                    total+=reward
                    if done or truncated:break
                rows.append({'method':method,'suite':case['suite'],'seed':case['seed'],
                    'reason':info['termination_reason'],'success':bool(info['success']),
                    'collision':bool(info['collision']),'joint_limit':bool(info['joint_limit_violation']),
                    'steps':env.steps,'position_error_m':float(info['position_error']),
                    'orientation_error_deg':float(np.rad2deg(info['orientation_error'])),
                    'reward':total,'planning_seconds':wrapper.planning_seconds})
            p.removeState(stateUniqueId=state,physicsClientId=env.client)
        finally:wrapper.close()
    return rows


def summarize(rows):
    result={}
    for method in ('ordinary','bc','ppo'):
        result[method]={}
        for suite in ('static','alignment_far','alignment_near'):
            group=[r for r in rows if r['method']==method and
                   (not r['suite'].startswith('alignment') if suite=='static' else r['suite']==suite)]
            if not group:continue
            timeout=[r for r in group if r['reason']=='timeout']
            successes=[r for r in group if r['success']]
            result[method][suite]={'episodes':len(group),
                **{key:sum(r['reason']==reason for r in group)/len(group) for key,reason in
                    [('success_rate','success'),('collision_rate','collision'),('timeout_rate','timeout'),
                     ('joint_limit_rate','joint_limit'),('planning_failure_rate','planning_failure')]},
                'timeout_mean_position_m':float(np.mean([r['position_error_m'] for r in timeout])) if timeout else None,
                'timeout_mean_orientation_deg':float(np.mean([r['orientation_error_deg'] for r in timeout])) if timeout else None,
                'successful_mean_steps':float(np.mean([r['steps'] for r in successes])) if successes else None,
                'mean_reward':float(np.mean([r['reward'] for r in group]))}
    return result


class TrackerValidation(BaseCallback):
    def __init__(self,directory,baseline,manifest,release,source_steps,interval,progress_weight):
        super().__init__()
        self.directory=Path(directory); self.baseline=baseline;self.manifest=manifest
        self.release=Path(release);self.source_steps=source_steps;self.interval=interval
        self.progress_weight=progress_weight;self.last=None;self.best=None

    def _on_step(self):return True
    def _on_training_start(self):self.evaluate('baseline')
    def _on_rollout_start(self):
        added=self.model.num_timesteps-self.source_steps
        if added>0 and added%self.interval==0 and added!=self.last:self.evaluate('post_update')
    def _on_training_end(self):
        path=self.directory/'final_model.zip';self.model.save(path)
        self.evaluate('final_saved_reload',PPO.load(path,device='cpu'))

    def evaluate(self,kind,model=None):
        model=self.model if model is None else model
        added=model.num_timesteps-self.source_steps
        rows=compare_policies(model,self.baseline,self.manifest,self.release,self.progress_weight)
        metrics=summarize(rows)
        record={'kind':kind,'additional_steps':added,'total_steps':model.num_timesteps,'metrics':metrics}
        with (self.directory/'validation.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        with (self.directory/'validation_episodes.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps({'kind':kind,'additional_steps':added,**row})+'\n')
        path=self.directory/'checkpoints'/f'checkpoint_{added}.zip';path.parent.mkdir(exist_ok=True)
        model.save(path)
        m=metrics['ppo']['static']
        # Safety and success first, then preserve alignment, then improve control speed.
        score=(m['success_rate'],-m['collision_rate'],
            min(metrics['ppo'][s]['success_rate'] for s in ('alignment_far','alignment_near')),
            -(m['successful_mean_steps'] or float('inf')))
        if self.best is None or score>self.best:
            self.best=score;model.save(self.directory/'best_model.zip')
            (self.directory/'selection.json').write_text(json.dumps(record,indent=2))
        (self.directory/'progress.json').write_text(json.dumps(record,indent=2))
        self.last=added
        print(json.dumps(record),flush=True)
