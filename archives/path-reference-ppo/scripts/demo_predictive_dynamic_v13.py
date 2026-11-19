"""Explicit RRT / PPO+prediction mixture, separate from the pure-PPO demo."""
import _bootstrap
from demo_autonomous_ppo import main
from wall_panel_rl.utils.config import ROOT

if __name__=='__main__':
    main(default_controller='mixture_predictive',manifest_name='candidate_manifest.json',
        motion_profile=ROOT/'configs/four_dynamic_faster.yaml',
        predictive_selection=ROOT/'configs/four_predictive_controller.json')
