"""Progress recovery must preserve state, successful continuations and budgets."""
import copy,json
from collections import deque
from types import SimpleNamespace
import numpy as np,pytest,torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController

@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1);p=json.loads((ROOT/'results/diagnostics/four_staggered_dynamic_preparation_v11_seed91/protocol.json').read_text());spec=yaml.safe_load((ROOT/'configs/four_progress_recovery.yaml').read_text());cases=json.loads((ROOT/'data/four_staggered_dynamic_v11_seed91/train_cases.json').read_text())
    return p,spec,{c['scene']['seed']:c for c in cases},PPO.load(ROOT/spec['source_model'],device='cpu')

def test_budget_risk_triggers_before_zero_progress():
    controller=object.__new__(ProgressRecoveryController);controller.recovery={'minimum_step':100,'window':24,'minimum_progress':.012,'cooldown':24,'maximum_rescues':3,'commit_steps':8}
    controller.env=SimpleNamespace(env=SimpleNamespace(steps=100,config={'simulation':{'max_steps':240}}));controller.rescues=0;controller.last_rescue=-1000
    controller.history=deque([(76+i,.65-.001*i) for i in range(25)],maxlen=25)
    assert controller.history[0][1]-controller.history[-1][1]>.012 and controller._stalled()
    controller.env.env.steps=70;assert not controller._stalled()

def test_existing_late_success_is_preserved_without_recovery_actions(setup):
    p,spec,cases,model=setup;case=cases[1110767];env=HybridFourEnv(copy.deepcopy(p['configuration']),[],copy.deepcopy(p['evaluation_settings']))
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']});controller=ProgressRecoveryController(model,env,spec['horizon'],spec['gains'],spec['recovery']);ref=env.cursor.path.copy()
        # Force one conservative trigger to exercise successful-continuation protection.
        controller._stalled=lambda:env.env.steps==100
        while True:
            before=(env.env.steps,env.env.sim_time,env.used);a=controller.action(obs);assert before==(env.env.steps,env.env.sim_time,env.used)
            obs,_,done,truncated,info=env.step(a)
            if done or truncated:break
        assert info['termination_reason']=='success' and env.env.steps==143 and controller.recovery_actions==0
        assert controller.protected and controller.forecast_calls>=1 and np.array_equal(ref,env.cursor.path)
    finally:env.close()

def test_actual_training_timeout_recovers_without_changing_limits_or_reference(setup):
    p,spec,cases,model=setup;case=cases[1110837];env=HybridFourEnv(copy.deepcopy(p['configuration']),[],copy.deepcopy(p['evaluation_settings']))
    try:
        obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']});controller=ProgressRecoveryController(model,env,spec['horizon'],spec['gains'],spec['recovery']);ref=env.cursor.path.copy()
        while True:
            before=(env.env.steps,env.env.sim_time,env.used);a=controller.action(obs);assert before==(env.env.steps,env.env.sim_time,env.used)
            obs,_,done,truncated,info=env.step(a)
            if done or truncated:break
        assert info['termination_reason']=='success' and env.env.steps<240 and controller.rescues>0 and not info['collision'] and not info['joint_limit_violation']
        assert env.env.config['simulation']['max_steps']==240 and np.array_equal(ref,env.cursor.path)
    finally:env.close()
