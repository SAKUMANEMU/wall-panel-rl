"""Trainable avoidance gate/residual for an inference-only neural controller.

No simulator query, collision filter or planner is called by this policy.
All distribution paths retain the inherited deterministic/stochastic mean.
"""
from wall_panel_rl.rl.multi_obstacle_adapter import MultiObstaclePolicy, CRITIC_PREFIXES

ACTOR_PREFIXES = ('multi_adapter.', 'prior_gate.', 'residual.')

class StandalonePolicy(MultiObstaclePolicy):
    def configure_learning(self, *, train_critic=False):
        for name,parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(ACTOR_PREFIXES) or
                (train_critic and name.startswith(CRITIC_PREFIXES)))

    def imitation_mean(self, observations):
        # Supervise before the final clamp to give oversaturated adapter outputs
        # a gradient back toward feasible labels. PPO retains exact clamp gradients.
        base=self.base_mean(observations)
        return base+self.adapter_correction(observations,base)
