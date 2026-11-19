"""Matched checkpoint restoration and paired fixed-speed experiment constraints."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.imitation_continuation import ContinuedImitation
from wall_panel_rl.rl.four_ppo_continuation import state_digest


SOURCE=ROOT/'results/diagnostics/four_dynamic_ppo_v8_seed61/selected_model.zip'
AUX=ROOT/'models/four_dynamic_ppo_v8_seed61/auxiliary_bc/aux_2000.pt'


def model():
    torch.set_num_threads(1);m=PPO.load(SOURCE,device='cpu');m.policy.configure_learning(train_critic=True)
    return m


def test_restore_bc_adam_pool_rng_and_counters_without_changing_ppo_or_weights(tmp_path):
    m=model();before=copy.deepcopy(m.policy.state_dict());ppo_digest=state_digest(m.policy.optimizer.state_dict())
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_speed_slow01.yaml').read_text())['auxiliary_bc']
    source=torch.load(AUX,map_location='cpu',weights_only=False)
    cb=ContinuedImitation(tmp_path,spec,71,resume_path=AUX);cb.init_callback(m);cb._on_training_start()
    assert cb.restoration['restored'] and cb.initial_accepted==cb.accepted==2 and cb.rejected==0
    assert len(cb.pool)==1467 and m.num_timesteps==62000
    assert state_digest(cb.optimizer.state_dict())==state_digest(source['optimizer'])
    assert state_digest(m.policy.optimizer.state_dict())==ppo_digest
    assert all(torch.equal(v,m.policy.state_dict()[k]) for k,v in before.items())
    for a,b in zip(cb.pool,source['pool']):
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
    rng=np.random.default_rng();rng.bit_generator.state=source['rng']
    np.testing.assert_array_equal(cb.rng.integers(0,len(cb.pool),64),rng.integers(0,len(cb.pool),64))


def test_restore_rejects_mismatched_checkpoint_and_truncated_pool(tmp_path):
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_speed_slow01.yaml').read_text())['auxiliary_bc']
    m=model();m.num_timesteps+=1000
    cb=ContinuedImitation(tmp_path/'counter',spec,resume_path=AUX);cb.init_callback(m)
    with pytest.raises(ValueError,match='counter'):cb._on_training_start()
    m=model();cb=ContinuedImitation(tmp_path/'capacity',dict(spec,capacity=100),resume_path=AUX);cb.init_callback(m)
    with pytest.raises(ValueError,match='truncate'):cb._on_training_start()


def test_restore_rejects_wrong_adam_tensor_shapes(tmp_path):
    state=torch.load(AUX,map_location='cpu',weights_only=False);key=next(iter(state['optimizer']['state']))
    state['optimizer']['state'][key]['exp_avg']=torch.zeros(1)
    corrupted=tmp_path/'bad.pt';torch.save(state,corrupted)
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_speed_slow01.yaml').read_text())['auxiliary_bc']
    cb=ContinuedImitation(tmp_path/'out',spec,resume_path=corrupted);cb.init_callback(model())
    with pytest.raises(ValueError,match='shape'):cb._on_training_start()


def test_pair_differs_only_in_speed_and_output_metadata_and_default_never_trains(monkeypatch,capsys):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import run_dynamic_speed_comparison as script
    comparison=yaml.safe_load((ROOT/'configs/four_dynamic_speed_comparison.yaml').read_text())
    _,specs=script.profiles(comparison)
    assert specs[0]['static_probability']==specs[1]['static_probability']==.1
    assert specs[0]['seed']==specs[1]['seed']==71 and specs[0]['validation_target_speed']==.02
    for s in specs:assert s['resume_auxiliary_state'].endswith('aux_2000.pt') and not s['run_independent_test']
    monkeypatch.setattr(script,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Unexpected training')))
    monkeypatch.setattr('sys.argv',['run_dynamic_speed_comparison.py']);script.main();assert 'check_only' in capsys.readouterr().out


def test_common_trainer_accepts_fixed_low_speed_but_keeps_target_validation(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'));import train_four_dynamic_ppo as script
    for name,speed in (('slow01',.01),('fast02',.02)):
        spec=yaml.safe_load((ROOT/f'configs/four_dynamic_speed_{name}.yaml').read_text())
        _,train,validation,legacy,held,_=script.inputs(spec)
        assert len(train)==192 and len(validation)==len(legacy)==64
        assert not {c['scene']['seed'] for c in train}&held
        assert spec['curriculum_speeds']==[speed] and spec['validation_target_speed']==.02


def test_ninety_percent_dynamic_mode_stream_and_fresh_phases_are_reproducible(monkeypatch):
    from wall_panel_rl.rl.dynamic_four_training import DynamicFourBankEnv,HybridFourEnv
    import json
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_speed_slow01.yaml').read_text())
    p=json.loads((ROOT/spec['preparation_report']/'protocol.json').read_text())
    cases=json.loads((ROOT/spec['data_root']/'train_cases.json').read_text())[:2]
    captured=[]
    def capture(self,**kwargs):captured.append(kwargs['options']);return np.zeros(77,np.float32),{}
    monkeypatch.setattr(HybridFourEnv,'reset',capture)
    env=DynamicFourBankEnv(p['configuration'],p['evaluation_settings'],cases,spec,seed=71)
    try:
        for _ in range(100):env.reset()
        expected=np.random.default_rng(71+4000000).random(100)<.1
        assert [x['task']=='static' for x in captured]==expected.tolist()
        assert all(x['strength'][1]==(0. if x['task']=='static' else .01) for x in captured)
        assert captured[0]['scenario']['obstacles']!=captured[1]['scenario']['obstacles']
    finally:env.close()


def test_resumed_bc_updates_before_each_rollout_keep_real_observation_log_probabilities_on_policy(tmp_path):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    from wall_panel_rl.rl.four_ppo_continuation import load_continuation
    state=torch.load(AUX,map_location='cpu',weights_only=False);observations=[x[0] for x in state['pool'][:8]]
    class ReplayObservations(gym.Env):
        observation_space=spaces.Box(-np.inf,np.inf,(77,),np.float32)
        action_space=spaces.Box(-1,1,(7,),np.float32)
        def reset(self,*,seed=None,options=None):self.index=0;return observations[0].copy(),{}
        def drain_demonstrations(self):return []
        def step(self,action):
            self.index+=1
            return observations[self.index%len(observations)].copy(),1.,self.index==3,False,{}
    env=DummyVecEnv([ReplayObservations]);audits=[]
    class Audit(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_end(self):
            b=self.model.rollout_buffer
            with torch.no_grad():
                _,logp,_=self.model.policy.evaluate_actions(torch.as_tensor(b.observations.reshape(-1,77)),torch.as_tensor(b.actions.reshape(-1,7)))
            np.testing.assert_allclose(logp.numpy(),b.log_probs.ravel(),atol=1e-5,rtol=1e-5);audits.append(True)
    try:
        spec=yaml.safe_load((ROOT/'configs/four_dynamic_speed_slow01.yaml').read_text())
        m,_=load_continuation(SOURCE,env,dict(spec,n_steps=4,batch_size=4,n_epochs=1))
        cb=ContinuedImitation(tmp_path,spec['auxiliary_bc'],71,resume_path=AUX)
        m.learn(8,reset_num_timesteps=False,callback=[Audit(),cb])
        assert m.num_timesteps==62008 and len(audits)==2
        assert cb.initial_accepted==2 and cb.accepted==4 and cb.rejected==0
    finally:env.close()
