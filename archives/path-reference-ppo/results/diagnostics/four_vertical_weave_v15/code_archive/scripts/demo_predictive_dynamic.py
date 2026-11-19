"""Current predictive demo with progress recovery; v13 baseline kept separately."""
import _bootstrap
from demo_autonomous_ppo import main
from wall_panel_rl.utils.config import ROOT

if __name__=='__main__':
    main(default_controller='mixture_progress',manifest_name='candidate_manifest.json',
         motion_profile=ROOT/'configs/four_vertical_weave.yaml',
         predictive_selection=ROOT/'configs/four_progress_controller.json')
