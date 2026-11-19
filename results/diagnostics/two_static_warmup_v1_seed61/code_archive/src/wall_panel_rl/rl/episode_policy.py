"""Choose a complete controller once per episode, with a private RNG.

This is an execution router. It does not replace actions inside an SB3 PPO
environment or insert BC transitions into an on-policy rollout buffer.
"""
import numpy as np
from gymnasium import spaces


class EpisodeChoice:
    """An independent coin, drawn once per episode, without a fixed quota."""
    def __init__(self,probability=.5,seed=None,expert_name='rrt'):
        if not np.isfinite(probability) or not 0<=probability<=1:
            raise ValueError('Expert probability must be between zero and one')
        self.expert_probability=float(probability)
        self.expert_name=expert_name
        self.rng=np.random.default_rng(seed)
        self.current_policy=None
        self.episodes_started=0

    def begin_episode(self):
        if self.current_policy is not None:
            raise RuntimeError('End the current episode before selecting another policy')
        self.current_policy=self.expert_name if self.rng.random()<self.expert_probability else 'ppo'
        self.episodes_started+=1
        return self.current_policy

    def end_episode(self):self.current_policy=None


class EpisodePolicy(EpisodeChoice):
    """Optional standalone BC/PPO router with the same episode ownership rule."""

    def __init__(self,bc_model,ppo_model,bc_probability=.5,seed=None):
        super().__init__(bc_probability,seed,expert_name='bc')
        self.models={'bc':bc_model,'ppo':ppo_model}
        for model in self.models.values():
            if not isinstance(model.observation_space,spaces.Box) or model.observation_space.shape!=(77,):
                raise ValueError('Expected the 77-feature reference-preserving observation')
            action=model.action_space
            if (not isinstance(action,spaces.Box) or action.shape!=(7,) or
                not np.array_equal(action.low,-np.ones(7)) or not np.array_equal(action.high,np.ones(7))):
                raise ValueError('Expected seven normalized joint-increment actions')

    def predict(self,observation,*,deterministic=True):
        if self.current_policy is None:
            raise RuntimeError('Select a policy with begin_episode before taking actions')
        return self.models[self.current_policy].predict(observation,deterministic=deterministic)

def safety_first_score(result,thresholds):
    """Retain minimum task ability, then prefer fewer observed collisions.

    Unlike the historical non-regression rule, a timeout increase does not veto
    a candidate that reduces collisions while preserving the success floor.
    This is a finite-sample ranking, not a guarantee of real-world safety.
    """
    from wall_panel_rl.rl.avoidance_curriculum import gate_result
    metrics=result['metrics']['dynamic_ppo']
    gate=gate_result(metrics,thresholds)
    retained=all(result['checks'].get(k,False) for k in
        ('alignment_collision_free','goal_convergence'))
    target=metrics['stage_target'];n=target['episodes']
    if n<=0:return (0,-1.,0.,-1.,-1.)
    return (int(gate['passed'] and retained),-target['collision']/n,
            target['success']/n,-target['timeout']/n,-target['joint_limit']/n)
