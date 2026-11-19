"""Real-execution telemetry and descriptive timeout classification only."""
import numpy as np
from wall_panel_rl.visualization.episode_replay import capture_frame


def control_frame(env,info,action=None):
    frame=capture_frame(env.env);q=env.env.joint_state()[0]
    frame.update(reference_index=int(env.cursor.index),reference_count=len(env.cursor.path),
        reference_distance_rad=float(np.linalg.norm(env.ref-q)),
        position_error_m=float(info['position_error']),
        orientation_error_deg=float(np.rad2deg(info['orientation_error'])),
        obstacle_clearance_capped_m=info.get('obstacle_clearance_capped_m'))
    if action is not None:frame['command_action']=np.asarray(action).tolist()
    return frame


def diagnose(recording):
    result=recording['result'];frames=recording['trajectory']
    tail=frames[-min(41,len(frames)):];joints=np.asarray([f['joints'] for f in tail])
    moved=np.linalg.norm(np.diff(joints,axis=0),axis=1)
    commands=[f['command_action'] for f in tail if 'command_action' in f]
    stationary=float(np.mean(moved<1e-4)) if len(moved) else 0.
    waiting=float(np.mean(np.max(np.abs(commands),axis=1)<1e-8)) if commands else None
    has_reference=all('reference_index' in f for f in tail)
    plateau=has_reference and tail[-1]['reference_index']==tail[0]['reference_index']
    stale=bool(plateau and tail[-1]['reference_distance_rad']>.15)
    precision=result['position_m']<=.03
    flags={'waiting':stationary>=.5 or (waiting is not None and waiting>=.5),
           'stale_reference':stale,'terminal_precision':precision}
    if result['reason']!='timeout':primary=result['reason']
    elif precision:primary='terminal_precision'
    elif flags['waiting']:primary='waiting'
    elif stale:primary='stale_reference'
    else:primary='insufficient_progress'
    return dict(primary=primary,flags=flags,window_steps=max(0,len(tail)-1),
        stationary_fraction=stationary,zero_command_fraction=waiting,
        reference_telemetry_available=has_reference,
        reference_index_start=tail[0].get('reference_index'),reference_index_end=tail[-1].get('reference_index'),
        final_reference_distance_rad=tail[-1].get('reference_distance_rad'),
        final_position_m=result['position_m'],final_orientation_deg=result['orientation_deg'],
        reference_state_inferred=False)
