"""Physics lookahead must leave the true PPO episode completely unchanged."""
import copy,json
import numpy as np,pytest,torch
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import HybridFourEnv
from wall_panel_rl.rl.predictive_safety import EpisodeSnapshot,PredictiveSafetyController

@pytest.fixture(scope='module')
def setup():
    torch.set_num_threads(1);p=json.loads((ROOT/'results/diagnostics/four_staggered_dynamic_preparation_v11_seed91/protocol.json').read_text())
    cases=json.loads((ROOT/'data/four_staggered_dynamic_v11_seed91/train_cases.json').read_text());diagnosis=json.loads((ROOT/'results/diagnostics/four_staggered_dynamic_bc_v11_seed91/training_diagnosis.json').read_text())
    seed=next(r['seed'] for r in diagnosis if r['reason']=='success');case=next(c for c in cases if c['scene']['seed']==seed)
    return p,case,PPO.load(ROOT/'results/diagnostics/four_dynamic_speed_v9_fast02_seed71/selected_model.zip',device='cpu')

def environment(p,case):
    env=HybridFourEnv(copy.deepcopy(p['configuration']),[],copy.deepcopy(p['evaluation_settings']))
    obs,_=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']});return env,obs

def test_repeated_rollout_restore_matches_entire_original_success_episode(setup):
    p,case,model=setup;a,x=environment(p,case);b,y=environment(p,case)
    try:
        while True:
            command=model.predict(x,deterministic=True)[0]
            with EpisodeSnapshot(a) as state:
                for gain in [0.,-.4,.6]:
                    state.restore()
                    for _ in range(8):
                        _,_,done,truncated,_=a.step(command*gain)
                        if done or truncated:break
            x,r,done,truncated,info=a.step(command);y,s,d,t,j=b.step(command)
            np.testing.assert_allclose(x,y,atol=1e-7,rtol=0);np.testing.assert_allclose(np.array(a.env.joint_state()),np.array(b.env.joint_state()),atol=1e-7,rtol=0)
            assert a.env.steps==b.env.steps and a.env.sim_time==b.env.sim_time and a.cursor.index==b.cursor.index and a.used==b.used
            assert a.recovery_gate.attempts==b.recovery_gate.attempts and done==d and truncated==t and abs(r-s)<1e-7
            if done or truncated:break
        assert info['termination_reason']==j['termination_reason']=='success'
    finally:a.close();b.close()

def test_prediction_restores_clock_reference_and_history_even_after_exception(setup):
    p,case,_=setup;env,obs=environment(p,case);q=env.env.joint_state()[0].copy();ref=env.cursor.path.copy()
    try:
        with pytest.raises(RuntimeError,match='simulated'):
            with EpisodeSnapshot(env):env.step(np.zeros(7,np.float32));raise RuntimeError('simulated prediction failure')
        assert env.env.steps==0 and env.env.sim_time==0 and env.used==0 and not env.finished
        np.testing.assert_array_equal(q,env.env.joint_state()[0]);np.testing.assert_array_equal(ref,env.cursor.path)
        assert len(env.recovery_gate.history)==1
    finally:env.close()

def test_predictive_controller_reports_interventions_without_committing_any_prediction(setup):
    p,case,model=setup;env,obs=environment(p,case)
    try:
        controller=PredictiveSafetyController(model,env,horizon=8);before=copy.deepcopy(env.recovery_gate.history)
        action=controller.action(obs)
        assert env.env.steps==0 and env.env.sim_time==0 and env.used==0 and len(env.recovery_gate.history)==len(before)
        assert action.shape==(7,) and np.isfinite(action).all() and np.max(np.abs(action))<=1
        assert controller.prediction_steps>0 and controller.calls==1
    finally:env.close()
