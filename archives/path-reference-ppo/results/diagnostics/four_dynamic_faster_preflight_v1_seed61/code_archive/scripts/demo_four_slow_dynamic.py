"""One fresh slow-dynamic episode; 50% RRT / 50% unchanged v5 PPO by default."""
import _bootstrap
from demo_autonomous_ppo import main
from wall_panel_rl.utils.config import ROOT


if __name__=='__main__':
    main(default_controller='mixture',manifest_name='candidate_manifest.json',
         motion_profile=ROOT/'configs/four_slow_dynamic.yaml')
