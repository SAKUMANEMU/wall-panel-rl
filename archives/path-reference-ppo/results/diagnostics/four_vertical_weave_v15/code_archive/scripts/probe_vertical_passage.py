"""Selected development examples anchored on measured controller execution.

This is deliberately a demonstration search, never an unbiased performance test.
The original observation reference, action API and source model remain unchanged.
"""
import _bootstrap
import copy
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from check_vertical_weave import initialize, execute, WORK, REPORT, dump
from wall_panel_rl.rl.motion_aware_placement import fit_motion_anchors


def probe(original):
    seed = original['scene']['seed']
    try:
        measured = execute(original)
        candidate = copy.deepcopy(original)
        trace = measured['trajectory']
        candidate['scene']['nominal_motion'] = {
            'panel_positions': [t['panel'] for t in trace],
            'joint_positions': [t['joints'] for t in trace],
            'steps': measured['steps'], 'termination_reason': measured['reason'],
            'route_kind': 'measured_previous_controller_execution_demonstration_only'}
        candidate = fit_motion_anchors(WORK['config'], WORK['settings'], WORK['profile'], candidate)
        row = execute(candidate)
        # At the actual XY-nearest passage of each anchor, compare panel height
        # with the contemporaneous moving box. Require proximity, not an endpoint
        # comparison that could call a remote position a passage.
        xyz = np.array([t['panel'] for t in row['trajectory']])
        passages = []
        order = candidate['scene']['motion_anchor_placement']['route_slot_order']
        for i in order:
            centers = np.array([t['obstacle_centers'][i] for t in row['trajectory']])
            distance = np.linalg.norm((xyz-centers)[:, :2], axis=1)
            index = int(np.argmin(distance))
            passages.append({'slot': i, 'step': index, 'xy_distance_m': float(distance[index]),
                             'relative_center_height_m': float(xyz[index,2]-centers[index,2])})
        row['passages'] = passages
        return {'case': candidate, 'row': row, 'seed': seed}
    except (RuntimeError, ValueError) as exc:
        return {'seed': seed, 'rejection': str(exc)}


def main():
    target = REPORT/'passage_probe.json'
    if target.exists():
        raise FileExistsError('Refusing to overwrite demonstration probes')
    originals = json.loads((REPORT/'cases.json').read_text())[:4]
    rows = []
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        for result in pool.map(probe, originals):
            rows.append(result)
            dump(target, rows)
            print(json.dumps({k:v for k,v in result.items() if k not in ('case','row')}),
                  result.get('row',{}).get('reason'), result.get('row',{}).get('passages'), flush=True)
    successful = [x for x in rows if x.get('row', {}).get('reason') == 'success']
    if not successful:
        raise RuntimeError('No successful passage demonstration; keep initial example')
    selected = max(successful, key=lambda x: min(x['row']['reference_departure_up_m'],
                                                x['row']['reference_departure_down_m']))
    archive = REPORT/'initial_upper_detour'
    archive.mkdir(exist_ok=True)
    for name in ['selected_example.json', 'selected_case.json', 'selected_trajectory.json', 'trajectory.png']:
        if (REPORT/name).exists() and not (archive/name).exists():
            shutil.copy2(REPORT/name, archive/name)
    row = selected['row']
    summary = {k:v for k,v in row.items() if k not in ('trajectory', 'nominal_panel_positions')}
    summary.update(scope='Selected development demonstration, not an independent success-rate estimate',
                   selection='Development search: four measured execution routes, three feasible reanchored layouts; select successful example with largest two-sided reference-height departure',
                   model_weights_changed=False, optimizer_updates=0, real_step_limit=240,
                   profile='configs/four_vertical_weave.yaml',
                   placement_route_kind=selected['case']['scene']['nominal_motion']['route_kind'],
                   interpretation='Repeated changes in panel height in a collision-free dynamic scene. Nearest-XY passage measurements do not establish alternating passage above/below every whole box.')
    dump(REPORT/'selected_example.json', summary)
    dump(REPORT/'selected_case.json', [selected['case']])
    dump(REPORT/'selected_trajectory.json', row)


if __name__ == '__main__':
    main()
