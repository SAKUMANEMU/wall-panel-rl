"""An independently trainable correction, active only for multiple obstacles.

The original actor is frozen. A zero output layer preserves the corrected
initial policy exactly; the presence mask preserves every one/zero-box action
even after BC/PPO updates. No collision queries or online planner run here.
"""
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from wall_panel_rl.rl.avoidance_curriculum import ReferencePreservingPolicy
from wall_panel_rl.rl.autonomous_dynamic_ppo import SpacesOnly


CRITIC_PREFIXES = ('mlp_extractor.value_net.', 'value_net.')


class MultiObstaclePolicy(ReferencePreservingPolicy):
    imitation_prefixes = ('multi_adapter.',)

    def __init__(self, *args, adapter_hidden=128, adapter_scale=2., **kwargs):
        if adapter_hidden < 8 or not 0 < adapter_scale <= 2:
            raise ValueError('Invalid multi-obstacle adapter settings')
        self.adapter_hidden = int(adapter_hidden)
        self.adapter_scale = float(adapter_scale)
        super().__init__(*args, **kwargs)
        # Include local waypoint error and frozen base action explicitly.
        self.multi_adapter = nn.Sequential(
            nn.Linear(91, self.adapter_hidden), nn.Tanh(),
            nn.Linear(self.adapter_hidden, self.adapter_hidden), nn.Tanh(),
            nn.Linear(self.adapter_hidden, 7))
        nn.init.zeros_(self.multi_adapter[-1].weight)
        nn.init.zeros_(self.multi_adapter[-1].bias)
        self.configure_learning(train_critic=True)
        schedule = kwargs.get('lr_schedule', args[2] if len(args) > 2 else None)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=schedule(1), **self.optimizer_kwargs)

    def configure_learning(self, *, train_critic=False):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(self.imitation_prefixes) or
                (train_critic and name.startswith(CRITIC_PREFIXES)))

    def base_mean(self, observations):
        return super().tracking_mean(observations)

    def adapter_correction(self, observations, base=None):
        base = self.base_mean(observations) if base is None else base
        slots = observations[:, 20:70].reshape(-1, 5, 10)
        multi = ((slots[:, :, 9] > .5).sum(1, keepdim=True) >= 2).to(observations.dtype)
        strength, _, _ = self.correction_components(observations)
        span = observations.new_tensor(self.joint_upper) - observations.new_tensor(self.joint_lower)
        error = ((observations[:, 70:77] - observations[:, :7]) * span * .5 /
                 self.joint_increment).clamp(-4, 4) / 4
        features = torch.cat((observations, error, base.detach()), dim=1)
        delta = self.adapter_scale * torch.tanh(self.multi_adapter(features))
        return multi * strength * delta

    def tracking_mean(self, observations):
        base = self.base_mean(observations)
        return (base + self.adapter_correction(observations, base)).clamp(-1, 1)

    def _get_constructor_parameters(self):
        return dict(super()._get_constructor_parameters(),
                    adapter_hidden=self.adapter_hidden, adapter_scale=self.adapter_scale)


def initialize_adapter(source_path, settings, seed=61):
    """Transfer existing weights, zero unused inputs, create fresh optimizers."""
    old = PPO.load(source_path, device='cpu')
    kwargs = dict(old.policy_kwargs)
    kwargs.update(adapter_hidden=settings.get('adapter_hidden', 128),
                  adapter_scale=settings.get('adapter_scale', 2.))
    model = PPO(MultiObstaclePolicy, DummyVecEnv([SpacesOnly]), device='cpu', seed=seed,
        n_steps=1000, batch_size=100, n_epochs=3, learning_rate=5e-5,
        clip_range=.05, target_kl=.005, ent_coef=0., policy_kwargs=kwargs)
    missing, unexpected = model.policy.load_state_dict(old.policy.state_dict(), strict=False)
    if unexpected or any(not name.startswith('multi_adapter.') for name in missing):
        raise ValueError('Source actor architecture changed')
    with torch.no_grad():
        for name in ('residual.0.weight', 'prior_gate.0.weight'):
            dict(model.policy.named_parameters())[name][:, 30:70].zero_()
    model.num_timesteps = old.num_timesteps
    model.policy.configure_learning(train_critic=False)
    return model


def frozen_base_preserved(before, after):
    a, b = before.state_dict(), after.state_dict()
    return all(name in b and torch.equal(value, b[name]) for name, value in a.items()
               if not name.startswith(('multi_adapter.', *CRITIC_PREFIXES)))
