"""Read-only statistics and trajectory evidence for frozen-model verification."""
import math
import numpy as np
from wall_panel_rl.rl.avoidance_curriculum import RecoveringReferenceCursor

REASONS=('success','collision','timeout','joint_limit','planning_failure')


def wilson(success,total):
    if not total:return None
    z=1.959963984540054;p=success/total;den=1+z*z/total
    middle=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,middle-half),min(1.,middle+half)]


def metrics(rows):
    def group(items):
        counts={r:sum(x['reason']==r for x in items) for r in REASONS}
        if sum(counts.values())!=len(items):raise ValueError('Unclassified outcome')
        return dict(episodes=len(items),**counts,success_rate=counts['success']/len(items) if items else None,
                    success_wilson95=wilson(counts['success'],len(items)))
    return {'all':group(rows),'by_count':{str(n):group([r for r in rows if r['obstacle_count']==n])
            for n in (2,3,4,5)},'by_max_obstacle_speed':{
            'low_0.04_to_0.10':group([r for r in rows if r.get('max_obstacle_speed_m_s',1.)<=.10]),
            'medium_0.10_to_0.16':group([r for r in rows if .10<r.get('max_obstacle_speed_m_s',1.)<=.16]),
            'high_0.16_to_0.22':group([r for r in rows if .16<r.get('max_obstacle_speed_m_s',1.)<=.22])}}


def paired(rows):
    by_seed={}
    for row in rows:
        pair=by_seed.setdefault(row['seed'],{})
        if row['arm'] in pair:raise ValueError('Duplicate paired evaluation')
        pair[row['arm']]=row['reason']=='success'
    counts=dict(both_success=0,v31_only_success=0,v29_only_success=0,both_failed=0)
    for pair in by_seed.values():
        if set(pair)!=set(['v31_selected','v29_baseline']):continue
        a,b=pair['v31_selected'],pair['v29_baseline']
        key='both_success' if a and b else 'v31_only_success' if a else 'v29_only_success' if b else 'both_failed'
        counts[key]+=1
    total=sum(counts.values())
    return dict(matched_completed=total,**counts,
                paired_success_rate_difference=(counts['v31_only_success']-counts['v29_only_success'])/total if total else None)


def true_runs(flags,steps):
    runs=[];start=None
    for index,flag in enumerate([*flags,False]):
        if flag and start is None:start=index
        elif not flag and start is not None:
            runs.append({'first_step':int(steps[start]),'last_step':int(steps[index-1]),'steps':index-start})
            start=None
    return sorted(runs,key=lambda r:r['steps'],reverse=True)


def trace_evidence(recording):
    frames=recording['trajectory'];result=recording['result'];scene=recording['case']['scene']
    if not all('reference_index' in f for f in frames):raise ValueError('Actual reference telemetry is required')
    q=np.asarray([f['joints'] for f in frames]);idx=np.asarray([f['reference_index'] for f in frames])
    steps=[f['step'] for f in frames]
    action=np.asarray([f['command_action'] for f in frames[1:]])
    zero=np.max(np.abs(action),axis=1)<1e-8
    stationary=np.linalg.norm(np.diff(q,axis=0),axis=1)<1e-4
    cursor=RecoveringReferenceCursor(scene['reference_joint_path'])
    if len(cursor.path)!=frames[-1]['reference_count']:raise ValueError('Reference cursor geometry differs')
    final_index=int(idx[-1]);future=np.linalg.norm(cursor.path[final_index:]-q[-1],axis=1)
    nearest=final_index+int(np.argmin(future))
    positions=np.asarray([f['position_error_m'] for f in frames]);angles=np.asarray([f['orientation_error_deg'] for f in frames])
    costs=positions+.12*np.deg2rad(angles)
    table=[{k:f[k] for k in ('step','time_s','reference_index','reference_distance_rad','position_error_m','orientation_error_deg')}
           for f in frames if f['step']%40==0 or f is frames[-1]]
    clearances=[f['obstacle_clearance_capped_m'] for f in frames if f.get('obstacle_clearance_capped_m') is not None]
    windows=[]
    for event in result['recovery_events']:
        if not event.get('accepted'):continue
        start=int(event['step']);end=min(start+int(event['plan_steps']),len(frames)-1)
        windows.append({'start_step':start,'end_step':end,'mode':event['mode'],
            'reference_advance':int(idx[end]-idx[start]),'observed_task_cost_change':float(costs[start]-costs[end]),
            'interpretation':'Observed time window, not proof that every proposed action was executed'})
    return {'seed':scene['seed'],'termination':result['reason'],'control_steps':result['steps'],
        'simulation_seconds':frames[-1]['time_s'],'final_position_m':result['position_m'],
        'final_orientation_deg':result['orientation_deg'],
        'zero_command_steps':int(sum(zero)),'stationary_steps':int(sum(stationary)),
        'longest_zero_command_runs':true_runs(zero,steps[1:])[:5],
        'longest_unchanged_reference_runs':true_runs(np.diff(idx)==0,steps[1:])[:5],
        'first_reference_advance_step':next((int(steps[i]) for i in range(1,len(idx)) if idx[i]>idx[0]),None),
        'reference_final_index':final_index,'reference_count':len(cursor.path),
        'final_current_reference_distance_rad':frames[-1]['reference_distance_rad'],
        'nearest_forward_reference_index':nearest,'nearest_forward_reference_distance_rad':float(min(future)),
        'best_task_cost_step':int(steps[int(np.argmin(costs))]),'minimum_task_cost':float(min(costs)),
        'minimum_recorded_obstacle_clearance_capped_m':min(clearances) if clearances else None,
        'obstacle_speeds_m_s':[m['speed_m_s'] for m in scene['obstacle_motion']],
        'accepted_recovery_windows':windows,'installation_events':result.get('installation_events',[]),
        'reference_escape_rejections':sum(e['mode']=='reject_reference_escape' for e in result.get('installation_events',[])),
        'trace_every_40_steps':table,'original_diagnosis':recording.get('timeout_diagnosis')}


def diagnostic_markdown(evidence,rows):
    lines=['# Dual-obstacle timeout diagnosis','',
        'Retrospective analysis of two original v31 test failures. Model weights, controller settings,',
        'motion, collision constraints and installation tolerances are frozen. These cases are not',
        'new independent validation samples and are never used for training in this study.','',
        '| Seed | Final position error | Orientation error | Zero commands | Reference index | Reference distance |',
        '|---|---:|---:|---:|---:|---:|']
    for e in evidence:
        lines.append(f"| {e['seed']} | {e['final_position_m']:.3f} m | {e['final_orientation_deg']:.1f} deg | {e['zero_command_steps']} | {e['reference_final_index']}/{e['reference_count']-1} | {e['final_current_reference_distance_rad']:.3f} rad |")
    lines+=['','## Interpretation','',
        '- Seed 4720000: the reference still advances near the end, but the installation pose remains far away. Waiting and recovery consumed time; this is insufficient completion progress, not a millimetre-level terminal-precision failure.',
        '- Seed 4720100: the recorded reference is stale while the robot continues moving. Recovery rejection limits some detours, but later accepted recovery does not restore the tracking interface. This is route-tracking/rejoin failure rather than simply standing still.',
        '- Both terminate at the original 240-step budget without collision. Initial obstacle speeds are below 0.08 m/s, so excessive obstacle speed alone does not explain these failures.',
        '- Valid start/goal placements do not prove that these controllers can execute a feasible dynamic path.',
        '- Recovery windows in the JSON are observed trajectory intervals, not a claim that every predicted action was executed.','',
        '## Frozen 2 x 2 diagnostic counterfactuals','',
        '| Seed | Model / controller | Result | Steps | Position error | Orientation error |',
        '|---|---|---|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['seed'],r['arm'])):
        lines.append(f"| {r['seed']} | {r['arm']} | {r['reason']} | {r['steps']} | {r['position_m']:.3f} m | {r['orientation_deg']:.1f} deg |")
    if not rows:lines+=['','Counterfactual execution is pending; causal attribution is provisional.']
    lines+=['','The expanded random-scene study is reported separately. It does not tune or select models.','']
    return '\n'.join(lines)
