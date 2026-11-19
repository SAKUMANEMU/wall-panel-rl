"""A disclosed phase-variant demo search to reduce the selected example's wait."""
import _bootstrap
import copy
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
from probe_vertical_corridor import REPORT, WORK, initialize, run_case, passage_measurements, dump
from wall_panel_rl.rl.slow_dynamic_four import validate_placement


def trial(shift):
    case=json.loads((REPORT/'selected_case.json').read_text())[0]
    for box,motion in zip(case['scene']['obstacles'],case['scene']['obstacle_motion']):
        motion['phase']=(np.asarray(motion['phase'])+np.asarray(motion['omega'])*shift).tolist()
        box['center']=(np.asarray(motion['anchor'])+np.asarray(motion['extent'])*np.sin(motion['phase'])).tolist()
    case['scene']['demo_phase_shift_s']=shift
    validate_placement(WORK['config'],case)
    payload=run_case(case)
    measurements=passage_measurements(payload)
    q=np.array([f['joints'] for f in payload['trajectory']])
    movement=np.max(np.abs(np.diff(q,axis=0)),axis=1)
    moving=np.where(movement>1e-4)[0]
    first_movement=int(moving[0]+1) if len(moving) else None
    return {'shift_s':shift,'recording':payload,'measurements':measurements,
            'initial_stationary_steps':first_movement-1 if first_movement is not None else len(q)-1}


def main():
    target=REPORT/'phase_variants.json'
    if target.exists():raise FileExistsError('Refusing phase search overwrite')
    rows=[]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(trial,shift) for shift in [6.,7.5,9.]]
        for future in as_completed(futures):
            item=future.result();rows.append(item);dump(target,rows)
            print(item['shift_s'],item['recording']['result'],item['measurements']['passage_count'],item['initial_stationary_steps'],flush=True)


if __name__=='__main__':main()
