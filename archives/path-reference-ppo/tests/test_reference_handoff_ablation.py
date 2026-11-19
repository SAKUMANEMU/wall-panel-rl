"""Inference scalar changes keep learned tensors and the saved model unchanged."""
import copy
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.avoidance_curriculum import RecoveringReferenceCursor


def load_study():
    sys.path.insert(0,str(ROOT/'scripts'))
    import evaluate_reference_handoff as r
    study=yaml.safe_load((ROOT/'configs/reference_handoff_ablation.yaml').read_text())
    return r,study


def test_each_variant_changes_one_scalar_not_weights_or_original_settings():
    torch.set_num_threads(1);r,study=load_study();r.check(study)
    settings=r.read(Path(study['parent_report'])/'manifest.json')['evaluation_settings']
    before=copy.deepcopy(settings);file_hash=r.sha(study['model'])
    model=PPO.load(study['model'],device='cpu');fingerprint=r.tensor_fingerprint(model.policy)
    s=r.apply_variant(model,settings,study['variants']['recovery_wider'])
    assert s['reference_recovery']['radius']==.45 and model.policy.terminal_radius==.2
    assert settings==before and r.tensor_fingerprint(model.policy)==fingerprint
    s=r.apply_variant(model,settings,study['variants']['terminal_earlier'])
    assert s['reference_recovery']['radius']==.3 and model.policy.terminal_radius==.4
    assert settings==before and r.tensor_fingerprint(model.policy)==fingerprint
    assert r.sha(study['model'])==file_hash
    with torch.no_grad():
        # End-point convergence remains exact under the expanded blend.
        obs=torch.zeros((1,77));lo=obs.new_tensor(model.policy.joint_lower);hi=obs.new_tensor(model.policy.joint_upper)
        obs[:,:7]=2*(obs.new_tensor(model.policy.goal_angles)-lo)/(hi-lo)-1
        obs[:,70:77]=obs[:,:7]+.5;obs[:,29]=1
        assert float(model.policy.tracking_mean(obs).abs().max())<1e-5


def test_expanded_cursor_radius_recovers_only_forward_without_rewriting_route():
    path=np.zeros((5,7));path[:,:2]=[[0,0],[0,.5],[.5,.5],[.5,0],[1,0]]
    q=np.zeros(7);q[:2]=[.75,-.4];original=path.copy()
    narrow=RecoveringReferenceCursor(path,radius=.3)
    wide=RecoveringReferenceCursor(path,radius=.45)
    narrow.reference(q);wide.reference(q)
    assert narrow.recoveries==0 and wide.recoveries==1
    assert wide.index>narrow.index and np.array_equal(wide.path[[0,-1]],original[[0,-1]])
    index=wide.index;wide.reference(np.zeros(7));assert wide.index>=index
    assert np.array_equal(path,original)


def test_ablation_cli_does_not_run_without_explicit_flag(monkeypatch,capsys):
    r,study=load_study()
    monkeypatch.setattr(r,'run',lambda *a:(_ for _ in ()).throw(AssertionError('Evaluation in check mode')))
    monkeypatch.setattr(sys,'argv',['evaluate_reference_handoff.py','--config',str(ROOT/'configs/reference_handoff_ablation.yaml')])
    r.main();assert 'check_only' in capsys.readouterr().out
