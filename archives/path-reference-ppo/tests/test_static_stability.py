"""The stability run preserves settings and never silently extends scope."""
import copy
import sys
import pytest
import yaml
from wall_panel_rl.utils.config import ROOT


def test_stability_config_preserves_existing_training_and_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    import run_static_stability as r
    study=yaml.safe_load((ROOT/'configs/static_stability_study.yaml').read_text())
    training,evaluation,config=r.inputs(study)
    assert training['wall_reward']=={'distance':.02,'weight':1.}
    assert config['simulation']['max_steps']==240 and training['active_stage']['mode']=='static'
    assert 'wall_reward' not in evaluation and training['learning_rate']==1e-4
    changed=copy.deepcopy(study);changed['ppo_steps']=20000
    with pytest.raises(ValueError,match='exactly 10k'):r.inputs(changed)
    changed=copy.deepcopy(study);changed['ppo_learning_rate']=3e-4
    with pytest.raises(ValueError,match='settings changed'):r.inputs(changed)


def test_stability_defaults_to_check_only(monkeypatch,capsys):
    import run_static_stability as r
    monkeypatch.setattr(r,'run',lambda *a:(_ for _ in ()).throw(AssertionError('training in check mode')))
    monkeypatch.setattr(sys,'argv',['run_static_stability.py','--config',str(ROOT/'configs/static_stability_study.yaml')])
    r.main();assert 'check_only' in capsys.readouterr().out
