"""Resume a matched PPO/BC boundary without discarding expert or Adam state."""
import copy
import hashlib
from pathlib import Path
import numpy as np
import torch
from wall_panel_rl.rl.episode_imitation import AuxiliaryImitation
from wall_panel_rl.rl.four_ppo_continuation import state_digest


class ContinuedImitation(AuxiliaryImitation):
    def __init__(self,output,spec,seed=61,*,resume_path=None):
        super().__init__(output,spec,seed)
        self.resume_path=None if resume_path is None else Path(resume_path)
        self.initial_accepted=0;self.initial_rejected=0;self.restoration={'restored':False}

    def _on_training_start(self):
        super()._on_training_start()
        if self.resume_path is None:return
        state=torch.load(self.resume_path,map_location='cpu',weights_only=False)
        if state['ppo_num_timesteps']!=self.model.num_timesteps:
            raise ValueError('BC state does not match the PPO checkpoint counter')
        if len(state['pool'])>self.spec['capacity']:raise ValueError('Resume would truncate the expert pool')
        if any(np.asarray(x).shape!=(77,) or np.asarray(y).shape!=(7,) or
               not np.isfinite(x).all() or not np.isfinite(y).all() for x,y in state['pool']):
            raise ValueError('Invalid resumed expert observations/actions')
        if any(group['lr']!=self.spec['learning_rate'] for group in state['optimizer']['param_groups']):
            raise ValueError('Auxiliary optimizer learning rate changed')
        actor_before=[p.detach().clone() for p in self.params]
        saved_ids=[i for group in state['optimizer']['param_groups'] for i in group['params']]
        if len(saved_ids)!=len(self.params):raise ValueError('Auxiliary actor parameter count changed')
        for i,param in zip(saved_ids,self.params):
            values=state['optimizer']['state'].get(i,{})
            if any(values[k].shape!=param.shape for k in ('exp_avg','exp_avg_sq') if k in values):
                raise ValueError('Auxiliary Adam tensor shape changed')
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
        self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.pool.extend(copy.deepcopy(state['pool']))
        self.accepted=self.initial_accepted=int(state['accepted_updates'])
        self.rejected=self.initial_rejected=int(state['rejected_updates'])
        if any(not torch.equal(a,b) for a,b in zip(actor_before,self.params)):
            raise AssertionError('Restoring optimizer altered actor parameters')
        if state_digest(self.optimizer.state_dict())!=state_digest(state['optimizer']):
            raise AssertionError('BC optimizer restoration failed')
        self.restoration={'restored':True,'source_state':str(self.resume_path.resolve()),
            'source_state_sha256':hashlib.sha256(self.resume_path.read_bytes()).hexdigest(),
            'ppo_counter':self.model.num_timesteps,'pool_size':len(self.pool),
            'accepted_updates':self.accepted,'rejected_updates':self.rejected,
            'optimizer_sha256':state_digest(self.optimizer.state_dict()),
            'rng_sha256':state_digest(self.rng.bit_generator.state),'weights_unchanged_by_restoration':True}
