"""Four staggered obstacles with 18cm / 12cm-per-second known 3D motion."""
import _bootstrap
from demo_autonomous_ppo import main
from wall_panel_rl.utils.config import ROOT


if __name__=='__main__':
    main(default_controller='mixture',manifest_name='candidate_manifest.json',
         motion_profile=ROOT/'configs/four_dynamic_faster.yaml')
