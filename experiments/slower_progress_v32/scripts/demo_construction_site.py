"""Execute an untrained-for-site selected policy with physical site constraints.

The saved recording includes the construction configuration. Playback never
recomputes the controller. This one-episode demonstration is not a success rate.
"""
import _bootstrap
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from wall_panel_rl.submission import load_agent,load_runtime,load_json,save_json,run_episode
from wall_panel_rl.utils.config import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,default=ROOT/'checks_site/count4.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a fresh output directory')
    args.output.mkdir(parents=True)
    save_json(args.output/'status.json',{'status':'running','phase':'physical_demo','training_performed':False})
    torch.set_num_threads(1)
    runtime=load_runtime();case=load_json(args.case)
    result,recording=run_episode(load_agent(),case,'enhanced',runtime)
    positions=np.asarray([f['panel'] for f in recording['trajectory']])
    rotations=Rotation.from_quat([f['panel_orientation'] for f in recording['trajectory']]).as_matrix()
    half_z=np.abs(rotations[:,2,:])@(np.asarray(runtime['configuration']['panel']['dimensions'])/2)
    site=runtime['configuration']['construction_site']
    diagnostics={'floor_z':site['floor_z'],'ceiling_z':site['ceiling_z'],
        'minimum_recorded_panel_lower_z':float(np.min(positions[:,2]-half_z)),
        'maximum_recorded_panel_upper_z':float(np.max(positions[:,2]+half_z)),
        'physics_collision_checks_include_all_substeps':True,'new_training_performed':False}
    recording['construction_site_diagnostics']=diagnostics
    save_json(args.output/'recording.json',recording);save_json(args.output/'result.json',result)
    save_json(args.output/'site_diagnostics.json',diagnostics)
    save_json(args.output/'status.json',{'status':'completed','phase':'complete','result':result,'site_diagnostics':diagnostics})
    print(result,flush=True);print(diagnostics,flush=True)


if __name__=='__main__':main()
