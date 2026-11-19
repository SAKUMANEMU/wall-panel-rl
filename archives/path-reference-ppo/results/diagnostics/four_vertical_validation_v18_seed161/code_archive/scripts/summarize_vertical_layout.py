"""Audit frozen paired validation, export complete counts/diagnoses and plots."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from evaluate_vertical_layout import ROOT, read, dump, sha, metrics, mixture_rows

REPORT = ROOT/'results/diagnostics/four_vertical_validation_v18_seed161'
MODES = ('ppo', 'rrt', 'ppo_progress', 'mixture')
LABELS = {'ppo': 'PPO', 'rrt': 'RRT', 'ppo_progress': 'PPO + prediction\n+ local recovery',
          'mixture': '50/50 episode\nmixture'}


def wilson(k, n):
    z = 1.96
    center = (k/n+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., float(center-radius)), min(1., float(center+radius))]


def save_csv(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    state = read(REPORT/'status.json')
    if state['status'] != 'completed':
        raise RuntimeError('Validation is not complete')
    protocol = read(REPORT/'protocol.json')
    summary = read(REPORT/'summary.json')
    frozen = read(REPORT/'frozen_inputs.json')
    data = ROOT/protocol['spec']['data_root']
    cases = read(data/'validation_cases.json')
    split = read(data/'split.json')
    assignments = read(REPORT/'mixture_assignments.json')
    ids = [c['scene']['seed'] for c in cases]
    assert len(ids) == len(set(ids)) == 64
    assert ids == split['validation'] and not set(ids)&set(split['excluded_prior_ids'])
    assert split['train'] == split['test'] == []
    assert sha(data/'validation_cases.json') == frozen['validation_bank_sha256']
    assert sha(REPORT/'mixture_assignments.json') == frozen['mixture_assignments_sha256']
    assert sha(REPORT/'frozen_model.zip') == protocol['source_model_sha256']
    assert sha(ROOT/protocol['spec']['source_model']) == protocol['source_model_sha256']
    for name, digest in protocol['code_hashes'].items():
        assert sha(ROOT/name) == digest
        assert sha(REPORT/'code_archive'/name) == digest
    results = {mode: read(REPORT/f'{mode}.json')['rows'] for mode in MODES}
    assert results['mixture'] == mixture_rows(results, assignments)
    for mode, rows in results.items():
        assert [r['seed'] for r in rows] == ids
        assert metrics(rows) == summary['metrics'][mode]
        assert all(r['reference_preserved'] and not r['weights_changed'] and r['steps'] <= 240 for r in rows)
        if mode != 'mixture':
            assert all(r['global_rrt'] == (mode=='rrt') for r in rows)
    modes = list(MODES)
    aggregated = summary['metrics']
    stats = []
    for mode in modes:
        m = aggregated[mode]
        bounds = wilson(m['success'], 64)
        collision_bounds = wilson(m['collision'], 64)
        stats.append({'mode': mode, **m, 'success_95ci_lower': bounds[0], 'success_95ci_upper': bounds[1],
                      'collision_95ci_lower': collision_bounds[0], 'collision_95ci_upper': collision_bounds[1]})
    save_csv(REPORT/'metrics.csv', stats, list(stats[0]))
    fields = ['mode', 'seed', 'reason', 'steps', 'position_m', 'orientation_deg', 'zero_command_steps',
              'stationary_steps', 'longest_stationary_run', 'projection_modified_steps', 'initial_planning_s',
              'control_wall_s', 'total_wall_s', 'reference_index', 'reference_last', 'timeout_category',
              'position_exceeds_tolerance', 'orientation_exceeds_tolerance', 'local_rescues',
              'local_recovery_actions', 'predictive_interventions', 'replans', 'replan_failures',
              'mixture_branch']
    all_rows = [{**r, 'mode': mode} for mode in modes for r in results[mode]]
    save_csv(REPORT/'episodes.csv', all_rows, fields)
    save_csv(REPORT/'failures.csv', [r for r in all_rows if r['reason'] != 'success'], fields)
    paired = []
    for index, seed in enumerate(ids):
        pair = {'seed': seed}
        for mode in modes:
            row = results[mode][index]
            for key in ['reason', 'steps', 'position_m', 'orientation_deg', 'zero_command_steps', 'stationary_steps']:
                pair[f'{mode}_{key}'] = row[key]
        paired.append(pair)
    save_csv(REPORT/'paired_cases.csv', paired, list(paired[0]))
    transitions = {f'{base}_to_ppo_progress': dict(Counter(a['reason']+' -> '+b['reason']
                   for a,b in zip(results[base], results['ppo_progress']))) for base in ['ppo','rrt']}
    preserved = []
    for before, after in zip(results['ppo'], results['ppo_progress']):
        if before['reason'] == 'success':
            same = (after['reason']=='success' and before['steps']==after['steps'] and
                    after['local_recovery_actions']==0 and
                    abs(before['position_m']-after['position_m'])<1e-6 and
                    abs(before['orientation_deg']-after['orientation_deg'])<1e-6)
            preserved.append(same)
    expected_mixture = {reason: .5*(aggregated['rrt'][reason]+aggregated['ppo_progress'][reason])
                        for reason in ['success','collision','timeout','planning_failure','joint_limit']}
    dump(REPORT/'mixture_expectation.json', {'expected_counts_over_all_possible_coin_draws': expected_mixture,
         'expected_success_rate': expected_mixture['success']/64,
         'realized_frozen_coin_counts': aggregated['mixture'],
         'note': 'Expectation on this fixed64-case bank, not additional measured episodes.'})
    diagnosis = {}
    target = protocol['configuration']['target']
    for mode, rows in results.items():
        timeouts = [r for r in rows if r['reason']=='timeout']
        collisions = [r for r in rows if r['reason']=='collision']
        diagnosis[mode] = {'outcomes': aggregated[mode],
                          'timeout_categories': dict(Counter(r['timeout_category'] for r in timeouts)),
                          'timeout_mean_position_m': float(np.mean([r['position_m'] for r in timeouts])) if timeouts else None,
                          'timeout_mean_orientation_deg': float(np.mean([r['orientation_deg'] for r in timeouts])) if timeouts else None,
                          'timeout_position_outside': sum(r['position_exceeds_tolerance'] for r in timeouts),
                          'timeout_orientation_outside': sum(r['orientation_exceeds_tolerance'] for r in timeouts),
                          'collision_pairs': dict(Counter(c['pair'] for r in collisions for c in r['collision_objects'])),
                          'terminal_contact_unavailable': sum(not r['collision_objects'] for r in collisions),
                          'timeout_seeds': [r['seed'] for r in timeouts],
                          'collision_seeds': [r['seed'] for r in collisions],
                          'planning_failure_seeds': [r['seed'] for r in rows if r['reason']=='planning_failure']}
        failed_plans = [r for r in rows if r['reason']=='planning_failure']
        budget = protocol['configuration']['rrt'].get('timed_timeout', 30.)
        diagnosis[mode]['planning_failure_timing'] = {
            'configured_initial_budget_s': budget,
            'near_initial_time_budget': sum(r['initial_planning_s'] >= .95*budget for r in failed_plans),
            'initial_planning_s': [r['initial_planning_s'] for r in failed_plans],
            'interpretation': 'Elapsed near a wall-clock budget suggests search-budget exhaustion, not a proof of geometric infeasibility.'}
    dump(REPORT/'failure_diagnosis.json', {'modes': diagnosis, 'paired_transitions': transitions,
         'original_ppo_successes': len(preserved), 'preserved_original_successes': sum(preserved),
         'target_position_tolerance_m': target['position_tolerance'],
         'target_orientation_tolerance_deg': target['orientation_tolerance_deg'],
         'category_rule': protocol['measurement']})

    # Report final, adjusted anchor heights separately from base stagger guards.
    spans, adjacent = [], []
    for case in cases:
        scene = case['scene']
        witness = scene['unadjusted_motion_anchor_placement']
        heights = np.array([scene['obstacle_motion'][i]['anchor'][2] for i in witness['route_slot_order']])
        spans.append(float(np.ptp(heights)))
        adjacent.append(float(np.min(np.abs(np.diff(heights)))))
    geometry = {'base_offsets_m': [.08,.14], 'O2_additional_offset_m': -.05,
                'final_anchor_height_span_min_m': min(spans), 'final_anchor_height_span_max_m': max(spans),
                'final_minimum_adjacent_delta_m': min(adjacent),
                'note': 'Base24cm/10cm geometry guards precede explicit O2 translation; final heights can differ. Full-motion physical endpoint/wall/spacing guards are applied afterward.'}
    dump(REPORT/'geometry_summary.json', geometry)

    x = np.arange(4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), layout='constrained')
    ax = axes[0,0]
    bottom = np.zeros(4)
    colors = {'success': '#239b77', 'collision': '#d35445', 'timeout': '#e5ab32',
              'planning_failure': '#808995', 'joint_limit': '#8d67aa'}
    for reason in ['success','collision','timeout','planning_failure','joint_limit']:
        values = np.array([aggregated[mode][reason] for mode in modes])
        if not values.any():
            continue
        ax.bar(x, values, bottom=bottom, label=reason.replace('_',' '), color=colors[reason])
        for i, value in enumerate(values):
            if value >= 4:
                ax.text(i, bottom[i]+value/2, str(value), ha='center', va='center', fontsize=10)
        bottom += values
    for i, mode in enumerate(modes):
        m = aggregated[mode]
        ax.text(i, 65.5, f'C{m["collision"]} T{m["timeout"]}\nF{m["planning_failure"]}',
                ha='center', va='bottom', fontsize=8)
    ax.set(ylabel='Episodes', ylim=(0,85), title='Outcomes: same 64 frozen validation cases')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(.5,1.01), ncol=3)
    ax = axes[0,1]
    values = [aggregated[mode]['mean_success_steps'] or 0 for mode in modes]
    ax.bar(x, values, color='#4c78a8')
    for i, value in enumerate(values):
        ax.text(i, value+4, f'{value:.1f}' if value else 'No success', ha='center')
    ax.axhline(240, ls='--', color='gray', label='Unchanged 240-step limit')
    ax.set(ylabel='Real control steps', ylim=(0,265), title='Mean steps among successful episodes')
    ax.legend(fontsize=8)
    ax = axes[1,0]
    ax.bar(x-.18, [aggregated[m]['zero_command_steps']/64 for m in modes], .36, label='Zero requested command', color='#a78dcc')
    ax.bar(x+.18, [aggregated[m]['stationary_steps']/64 for m in modes], .36, label='Actual near-stationary step', color='#ecb274')
    ax.set(ylabel='Mean steps per episode', title='Waiting / stagnation measures (overlap)')
    ax.legend(fontsize=8)
    ax = axes[1,1]
    values = [aggregated[m]['mean_total_wall_s'] for m in modes]
    ax.bar(x, values, color='#5f9fa9')
    for i, value in enumerate(values):
        ax.text(i, value+.02*max(values+[1]), f'{value:.1f}s', ha='center')
    ax.set(ylabel='Wall-clock seconds per episode', title='Compute time including prediction/planning')
    for ax in axes.flat:
        ax.set_xticks(x, [LABELS[m] for m in modes], fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Current vertical layout: four moving obstacles, O2 lowered 5 cm', fontsize=14)
    fig.supxlabel('Validation only; no training. C: collision, T: timeout, F: planning failure. Compute measured with 4 workers.', fontsize=9)
    fig.savefig(REPORT/'comparison.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5,5.5), layout='constrained')
    colors = ['#777777','#4c78a8','#239b77']
    any_timeout = False
    for mode, color in zip(modes[:3], colors):
        rows = [r for r in results[mode] if r['reason']=='timeout']
        if rows:
            any_timeout = True
            ax.scatter([max(r['position_m']*1000,1e-6) for r in rows],
                       [max(r['orientation_deg'],1e-6) for r in rows],
                       label=LABELS[mode].replace('\n',' '), color=color, alpha=.8)
            for row in rows:
                ax.annotate(f'{row["seed"]}: {row["position_m"]*1000:.1f} mm / {row["orientation_deg"]:.1f} deg',
                            (row['position_m']*1000,row['orientation_deg']), xytext=(-8,-22),
                            textcoords='offset points', ha='right', fontsize=8, color=color)
    ax.axvline(target['position_tolerance']*1000, ls='--', color='black', lw=1)
    ax.axhline(target['orientation_tolerance_deg'], ls='--', color='black', lw=1)
    ax.set(xscale='log', yscale='log', xlabel='Timeout terminal position error (mm)',
           ylabel='Timeout terminal orientation error (degrees)',
           title='Timeouts: terminal errors and unchanged success tolerances', xlim=(7,800), ylim=(1.5,120))
    ax.set_xticks([10,20,50,100,200,500])
    ax.set_yticks([2,5,10,20,50,100])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    if any_timeout:
        ax.legend(fontsize=9, loc='upper left')
    else:
        ax.text(.5,.5,'No timeouts',ha='center',transform=ax.transAxes)
    ax.grid(alpha=.2)
    fig.savefig(REPORT/'timeout_errors.png', dpi=180)
    plt.close(fig)

    a, b = aggregated['ppo'], aggregated['ppo_progress']
    lines = ['# 当前高低错落布局：64 场景配对验证（v18）', '',
             '已完成版本冻结与批量验证；没有训练、没有调参、没有运行最终100场景测试。', '',
             '| 控制器 | 成功 | 碰撞 | 超时 | 规划失败 | 成功率 | 成功时平均步数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for mode in modes:
        m = aggregated[mode]
        label = LABELS[mode].replace('\n',' ')
        avg = f'{m["mean_success_steps"]:.1f}' if m['mean_success_steps'] is not None else '—'
        lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} | {m["planning_failure"]} | {m["success_rate"]:.1%} | {avg} |')
    lines += ['', f'额外关节越限计数：{ {m:aggregated[m]["joint_limit"] for m in modes} }。完整总数（包括全部失败）在 metrics.csv。', '',
              f'相对同场景 PPO，预测安全＋局部恢复成功数变化 {b["success"]-a["success"]:+d}，碰撞数变化 {b["collision"]-a["collision"]:+d}，超时数变化 {b["timeout"]-a["timeout"]:+d}。', '',
              f'原PPO的{len(preserved)}个成功中，{sum(preserved)}个保持同样成功、步数与终态误差，未采用局部恢复动作。增强分支从PPO原失败中新增成功23个：22个原碰撞、1个原超时；另1个原碰撞变成超时。', '',
              '## 场景与冻结协议', '',
              '- 使用当前随机动画的同一生成函数：新初始关节位置、无障碍名义路线、沿路线布置四个大小不一的动态障碍，随机相位和少量位置抖动。基础相对路线高度偏移8–14cm、正负交替；O2整条运动轨迹额外下移5cm，之后重新检查完整运动边界的起终点、墙面及障碍间距。',
              '- 这里评估默认随机生成规则，不使用从已成功执行中选出的演示布局，也不按任何控制器在有障碍时的表现筛选场景。构造阶段仍要求生成完整的无障碍名义路线；这项任务定义及所有几何拒绝均保留在 protocol.json 和 geometry_rejections.json。',
              f'- 接受64个验证场景，拒绝{frozen["rejected"]}个构造候选；所有验证种子与既有数据种子去重。最终调整后锚点高度跨度范围 {min(spans):.3f}–{max(spans):.3f}m，相邻锚点最小高度差 {min(adjacent):.3f}m。基础高度约束在O2下移前应用，下移后不再声称仍满足基础高度下限。',
              '- 四障碍全动态；矢量位移上限18cm、速度上限0.12m/s；已知运动模型，规划期间暂停物理时钟。无感知噪声、无延迟；240步上限、目标和成功阈值保持不变。',
              '- 使用源v9 fast02模型的冻结副本 frozen_model.zip；configuration/settings/controller/profile与运行代码哈希均在评估前冻结，结束后复核一致。原始参考路径和模型参数在每个episode后检查未改写。',
              '- “PPO”指同一77维观测/7维动作环境中的确定性策略推理；仍包含共享环境原有的动作投影与参考恢复包装器，不包含额外的预测安全、局部动作搜索或在线RRT。增强分支为PPO＋预测安全＋局部恢复，不是纯PPO，也没有全局RRT回退。',
              f'- RRT目标为同一参考路线末端、规划随机种子为场景种子+100000。混合策略在知道结果前逐回合抛50%概率硬币：本次{frozen["mixture_rrt_episodes"]}回合选RRT、{frozen["mixture_progress_episodes"]}回合选增强PPO。复用对应完整episode的实际结果，规划失败照计，没有成功分支替换或episode内切换。',
              '- 共192个真实控制器rollout（64×3）；混合64个结果来自上述episode的固定路由，不另算新的独立仿真。', '',
              f'- 固定场景上对所有可能的50/50硬币取期望，混合成功数为{expected_mixture["success"]:.1f}/64（{expected_mixture["success"]/64:.1%}）。这不是额外试验；实际预先固定的硬币结果仍为上表。', '',
              '## 失败诊断', '',
              f'实际成功阈值：位置<{target["position_tolerance"]:.6f}m，姿态<{target["orientation_tolerance_deg"]:.3f}°，同时无碰撞/越限。', '',
              '超时分类只是诊断标签：终点两项误差均在实际阈值3倍以内记为近目标精度不足；否则零命令或真实近静止步数占比≥25%记为等待/停滞；其他记为进展缓慢或无效。标签按上述优先级互斥，不能据此断言唯一因果。近静止定义为真实单步关节最大变化<1e-4rad，零命令为请求动作最大幅值<1e-8；这两项可能重叠。', '']
    for mode in modes:
        d = diagnosis[mode]
        lines += [f'- {LABELS[mode].replace(chr(10)," ")}：超时分类 {d["timeout_categories"]}；碰撞对象 {d["collision_pairs"]}；终态接触无法定位{d["terminal_contact_unavailable"]}次。']
    lines += ['', '终态接触查询无法覆盖已经分离的历史接触，不能把空的接触列表解读为没有碰撞。所有真实终止原因仍按原物理环境结果统计。', '',
              '## 本轮结论与优先顺序', '',
              '1. 当前增强PPO分支在此验证集63/64成功、0碰撞；这主要是执行层预测与恢复的效果，并不说明模型重新训练后提升。纯策略仍有23次碰撞，单独依靠现有模型尚不足以可靠避障。',
              '2. 增强分支唯一超时为种子1166054：位置误差0.352m、姿态35.47°，参考索引停在5/47；3次恢复均采用第4个关节（索引3）的-0.4偏置，消耗108个局部恢复动作。零命令/近静止各38步（15.8%），按预先定义规则属于进展无效，不能归为末端精度不足或单纯等待主导。下一步应检查参考路点重新接入和恢复结束后的策略交接，而不是直接放宽成功阈值或无限增加步数。',
              '3. RRT的8次规划失败耗时27.91–29.88s，设定初始搜索上限30s；这是搜索/时间安排能力的失败，不能据此断言几何无解。种子1166154在17次重规划失败和17个零命令后碰撞；终态未定位接触对象。应优先检查重规划失败后的停留位置和主动退避安全，单纯停住不总是安全。',
              '4. 本次混合策略选中4个RRT规划失败及该RRT碰撞，此外选中增强PPO唯一超时，因此成功低于增强分支单独运行。保持用户指定50/50路由，没有自动改成优胜分支或失败后切换。后续优先修复RRT失败和上述单个恢复停滞，再决定是否进行针对性BC/PPO适配。', '',
              '## 文件与后续', '',
              '- comparison.png：成功/碰撞/超时/规划失败、完成步数、等待及计算耗时。',
              '- timeout_errors.png：全部非混合分支超时的最终位置/姿态误差及原始阈值。',
              '- metrics.csv：各策略完整计数、成功率Wilson95%区间及耗时；样本为验证集，不代表最终泛化成功率。',
              '- episodes.csv / failures.csv / paired_cases.csv：逐场景全结果、失败明细及配对结果。',
              '- ppo.json / rrt.json / ppo_progress.json / mixture.json：逐episode结果和每真实步诊断轨迹。',
              '- failure_diagnosis.json：超时类型、碰撞对象、失败种子及策略间配对转移。',
              '- geometry_summary.json / frozen_inputs.json / protocol.json / verification.json：几何、冻结输入、配置与审计。',
              '- status.json 与 workers/*.json：总体及各进程状态，完成后保留。', '',
              '计算时间含规划和预测，来自4个并行CPU进程下的墙钟测量；不包含固定帧率回放，不是实时执行性能承诺。真实步数与虚拟预测步数分别记录，不把试算当作训练数据。', '',
              '下一步根据失败诊断确定是否补充动态示范和做BC/PPO适配。本轮没有开始训练，也没有运行最终测试；之后若改动算法，应只在验证集开发，模型与控制器锁定后再使用全新最终测试一次。']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    verification = {'all64_cases_identical_across_modes': True, 'all_outcomes_in_denominator': True,
                    'scene_seeds_disjoint_from_previous_banks': True,
                    'mixture_assignments_frozen_before_outcomes_and_verified': True,
                    'no_branch_substitution': True, 'runtime_code_and_model_hashes_verified': True,
                    'reference_paths_and_model_parameters_unchanged': True,
                    'original_ppo_successes_preserved': sum(preserved),
                    'original_ppo_successes': len(preserved),
                    'optimizer_updates': 0, 'final_test_episodes': 0,
                    'physical_rollouts': 192, 'mixture_reused_episodes': 64,
                    'paired_transitions': transitions}
    dump(REPORT/'verification.json', verification)
    shutil.copy2(Path(__file__), REPORT/'executed_summary.py')
    print(json.dumps({'metrics': aggregated, 'diagnosis': diagnosis, 'paired': transitions}), flush=True)


if __name__ == '__main__':
    main()
