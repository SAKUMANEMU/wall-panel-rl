"""Selected four-obstacle PPO policy; fresh random static scene,0.6x playback."""
import _bootstrap
from demo_autonomous_ppo import main
from wall_panel_rl.utils.config import ROOT


if __name__ == '__main__':
    main(default_controller='ppo',
         default_report=ROOT/'results/diagnostics/four_priority_ppo_v4_seed61',
         manifest_name='candidate_manifest.json')
