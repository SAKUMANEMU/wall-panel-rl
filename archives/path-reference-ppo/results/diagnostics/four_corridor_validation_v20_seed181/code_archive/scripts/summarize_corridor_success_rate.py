"""Audit installation and actual weaving rates on the frozen v20 validation bank."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, ScalarFormatter
from evaluate_corridor_success_rate import ROOT, read, dump, sha, summary_metrics

REPORT=ROOT/'results/diagnostics/four_corridor_validation_v20_seed181'
MODES=['ppo_progress','ppo']
LABELS={'ppo_progress':'PPO + prediction\n+ local recovery','ppo':'PPO policy\n(shared environment wrapper)'}


def wilson(k,n):
    z=1.96;denom=1+z*z/n
    center=(k/n+z*z/(2*n))/denom
    width=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/denom
    return [max(0.,center-width),min(1.,center+width)]


def csv_file(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def main():
    if read(REPORT/'status.json')['status']!='completed':raise RuntimeError('Evaluation is still running')
    protocol=read(REPORT/'protocol.json');spec=protocol['spec'];data=ROOT/spec['data_root']
    cases=read(data/'validation_cases.json');split=read(data/'split.json');frozen=read(REPORT/'frozen_inputs.json')
    ids=[c['scene']['seed'] for c in cases];by_seed={c['scene']['seed']:c for c in cases}
    assert len(ids)==len(set(ids))==64 and ids==split['validation']
    assert not set(ids)&set(split['excluded_prior_ids'])
    assert sha(data/'validation_cases.json')==frozen['validation_bank_sha256']
    assert sha(REPORT/'frozen_model.zip')==sha(ROOT/spec['source_model'])==protocol['source_model_sha256']
    for name,digest in protocol['code_hashes'].items():
        assert sha(ROOT/name)==sha(REPORT/'code_archive'/name)==digest
    summary=read(REPORT/'summary.json')
    results={mode:read(REPORT/f'{mode}.json')['rows'] for mode in MODES}
    for mode,rows in results.items():
        assert [r['seed'] for r in rows]==ids
        assert summary_metrics(rows)==summary['metrics'][mode]
        for row in rows:
            assert not row['weights_changed'] and row['reference_preserved'] and row['steps']<=240
            payload=read(REPORT/'recordings'/f'{mode}_{row["seed"]}.json')
            assert payload['case']==by_seed[row['seed']]
            assert len(payload['trajectory'])==row['steps']+1
            assert payload['trajectory'][0]['step']==0 and payload['trajectory'][-1]['step']==row['steps']
            assert payload['result']['reason']==row['reason']
    for seed in ids:
        a=read(REPORT/'recordings'/f'ppo_{seed}.json')
        b=read(REPORT/'recordings'/f'ppo_progress_{seed}.json')
        assert a['configuration']==b['configuration'] and a['case']==b['case']
        assert a['trajectory'][0]==b['trajectory'][0]
    metrics=summary['metrics'];stats=[];all_rows=[];diagnosis={};weaving=[]
    target=protocol['configuration']['target']
    for mode in MODES:
        m=metrics[mode];ci=wilson(m['success'],64);wci=wilson(m['successful_with_verified_weaving'],64)
        stats.append({'mode':mode,**m,'success_95ci_lower':ci[0],'success_95ci_upper':ci[1],
                      'success_and_weaving_95ci_lower':wci[0],'success_and_weaving_95ci_upper':wci[1]})
        rows=results[mode];timeouts=[r for r in rows if r['reason']=='timeout'];collisions=[r for r in rows if r['reason']=='collision']
        diagnosis[mode]={'timeout_categories':dict(Counter(r['timeout_category'] for r in timeouts)),
            'timeout_mean_position_m':float(np.mean([r['position_m'] for r in timeouts])) if timeouts else None,
            'timeout_mean_orientation_deg':float(np.mean([r['orientation_deg'] for r in timeouts])) if timeouts else None,
            'timeout_position_outside_success_tolerance':sum(r['position_m']>=target['position_tolerance'] for r in timeouts),
            'timeout_orientation_outside_success_tolerance':sum(r['orientation_deg']>=target['orientation_tolerance_deg'] for r in timeouts),
            'collision_pairs':dict(Counter(c['pair'] for r in collisions for c in r['collision_objects'])),
            'collision_contact_unavailable':sum(not r['collision_objects'] for r in collisions),
            'timeout_seeds':[r['seed'] for r in timeouts],'collision_seeds':[r['seed'] for r in collisions]}
        all_rows+=rows
        weaving += [dict(mode=mode,seed=r['seed'],steps=r['steps'],passages=r['ordered_passages'],
                         recording=f'recordings/{mode}_{r["seed"]}.json') for r in rows if r['successful_with_verified_weaving']]
    csv_file(REPORT/'metrics.csv',stats,list(stats[0]))
    fields=['mode','seed','reason','steps','position_m','orientation_deg','passage_count',
            'successful_with_verified_weaving','zero_command_steps','stationary_steps','longest_stationary_run',
            'reference_index','reference_last','timeout_category','local_rescues','local_recovery_actions',
            'predictive_interventions','discarded_prediction_steps','control_wall_s','total_wall_s']
    csv_file(REPORT/'episodes.csv',all_rows,fields)
    csv_file(REPORT/'failures.csv',[r for r in all_rows if r['reason']!='success'],fields)
    paired=[]
    for index,seed in enumerate(ids):
        row={'seed':seed}
        for mode in MODES:
            for key in ['reason','steps','passage_count','successful_with_verified_weaving']:
                row[f'{mode}_{key}']=results[mode][index][key]
        paired.append(row)
    csv_file(REPORT/'paired_cases.csv',paired,list(paired[0]))
    transitions=dict(Counter(a['reason']+' -> '+b['reason'] for a,b in zip(results['ppo'],results['ppo_progress'])))
    dump(REPORT/'failure_diagnosis.json',{'modes':diagnosis,'ppo_to_enhanced_transitions':transitions})
    dump(REPORT/'weaving_examples.json',weaving)
    attempts=read(REPORT/'construction_attempts.json')
    csv_file(REPORT/'construction_attempts.csv',attempts,['seed','accepted','geometry_valid','reason','preliminary_reason','generation_wall_s'])
    geometry={'attempts':len(attempts),'accepted':64,'rejected':sum(not a['geometry_valid'] for a in attempts),
        'unused_valid_candidates':sum(a['geometry_valid'] and not a['accepted'] for a in attempts),
        'rejection_reasons':dict(Counter(a.get('reason') for a in attempts if not a['geometry_valid'])),
        'accepted_preliminary_outcomes':dict(Counter(c['scene']['preliminary_episode']['reason'] for c in cases)),
        'minimum_final_height_span_m':min(c['scene']['final_height_stagger']['anchor_height_span_m'] for c in cases),
        'minimum_final_adjacent_height_delta_m':min(c['scene']['final_height_stagger']['minimum_adjacent_delta_m'] for c in cases)}
    dump(REPORT/'geometry_summary.json',geometry)
    plot_rates(metrics,stats)
    plot_timeouts(results,target)
    write_report(protocol,metrics,stats,geometry,diagnosis,transitions)
    audit={'64_new_valid_scenes':True,'prior_seeds_and_development_examples_excluded':True,
           'identical_bank_for_both_modes':True,'128_final_scene_rollouts':True,
           'model_configuration_controller_and_code_unchanged':True,
           'all_final_failures_counted_without_replacement':True,'full_recordings_match_frozen_cases_and_counts':True,
           'all64_paired_initial_states_exactly_identical':True,
           'optimizer_updates':0,'final_test_episodes':0,'rate_denominator':64,
           'weaving_is_secondary_diagnostic_not_scene_filter':True,
           'installation_success_95ci':{r['mode']:[r['success_95ci_lower'],r['success_95ci_upper']] for r in stats}}
    dump(REPORT/'verification.json',audit)
    shutil.copy2(data/'validation_cases.json',REPORT/'validation_cases.json')
    shutil.copy2(data/'split.json',REPORT/'split.json')
    shutil.copy2(Path(__file__),REPORT/'executed_summary.py')
    print(json.dumps({'metrics':metrics,'construction':geometry,'diagnosis':diagnosis,'paired':transitions}),flush=True)


def plot_rates(metrics,stats):
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.5),layout='constrained')
    x=np.arange(2);bottom=np.zeros(2)
    for reason,color in [('success','#269e79'),('collision','#d55448'),('timeout','#e6ae37'),('joint_limit','#9370b1')]:
        values=np.array([metrics[m][reason] for m in MODES])
        if not values.any():continue
        axes[0].bar(x,values,bottom=bottom,color=color,label=reason.replace('_',' '))
        for i,v in enumerate(values):
            if v>=3:axes[0].text(i,bottom[i]+v/2,str(v),ha='center',va='center')
        bottom+=values
    axes[0].set(ylabel='Episodes',ylim=(0,80),title='Same 64 frozen valid scenes')
    for i,mode in enumerate(MODES):
        m=metrics[mode]
        axes[0].text(i,66,f'S{m["success"]} C{m["collision"]} T{m["timeout"]}',ha='center',fontsize=9)
    axes[0].legend(loc='upper center',ncol=3,fontsize=8)
    rates=np.array([metrics[m]['success_rate'] for m in MODES])
    lower=np.array([r['success_95ci_lower'] for r in stats]);upper=np.array([r['success_95ci_upper'] for r in stats])
    axes[1].bar(x-.18,rates,.36,color='#4c78a8',label='Installation success',
                yerr=np.stack([rates-lower,upper-rates]),capsize=4)
    weave=np.array([metrics[m]['success_and_weaving_rate'] for m in MODES])
    axes[1].bar(x+.18,weave,.36,color='#75b5a3',label='Success + measured weaving')
    for i in range(2):
        axes[1].text(i-.18,min(1.12,upper[i]+.025),f'{rates[i]:.1%}',ha='center',fontsize=9)
        axes[1].text(i+.18,weave[i]+.025,f'{weave[i]:.1%}',ha='center',fontsize=9)
    axes[1].set(ylabel='Fraction of all 64 scenes',ylim=(0,1.04),title='Success and actual vertical passage diagnostic')
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.))
    axes[1].legend(fontsize=8,loc='upper center')
    for ax in axes:
        ax.set_xticks(x,[LABELS[m] for m in MODES],fontsize=9)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Current larger-stagger layout: 14–22 cm offsets, O2 lowered 5 cm',fontsize=13)
    fig.supxlabel('Validation only; unchanged weights, controller and 240-step limit. Error bars: Wilson 95% intervals.',fontsize=9)
    fig.savefig(REPORT/'success_rate.png',dpi=180);plt.close(fig)


def plot_timeouts(results,target):
    fig,ax=plt.subplots(figsize=(8,5.5),layout='constrained')
    any_points=False
    for mode,color in zip(MODES,['#269e79','#777777']):
        rows=[r for r in results[mode] if r['reason']=='timeout']
        if rows:
            any_points=True
            ax.scatter([max(1e-6,r['position_m']*1000) for r in rows],
                       [max(1e-6,r['orientation_deg']) for r in rows],color=color,alpha=.7,
                       label=LABELS[mode].replace('\n',' '))
    ax.axvline(target['position_tolerance']*1000,ls='--',color='black',lw=1)
    ax.axhline(target['orientation_tolerance_deg'],ls='--',color='black',lw=1)
    ax.set(xscale='log',yscale='log',xlabel='Timeout terminal position error (mm)',
           ylabel='Timeout terminal orientation error (degrees)',title='Actual timeout errors; dashed lines are unchanged success thresholds')
    position_max=max([r['position_m']*1000 for rows in results.values() for r in rows if r['reason']=='timeout']+[1])
    angle_max=max([r['orientation_deg'] for rows in results.values() for r in rows if r['reason']=='timeout']+[1])
    ax.set_xticks([10,20,50,100,200,500,1000,2000]);ax.set_yticks([2,5,10,20,50,100,200])
    ax.set_xlim(7,max(1000,position_max*1.2));ax.set_ylim(1.5,max(100,angle_max*1.2))
    ax.xaxis.set_major_formatter(ScalarFormatter());ax.yaxis.set_major_formatter(ScalarFormatter())
    if any_points:ax.legend(fontsize=8)
    else:ax.text(.5,.5,'No timeouts',transform=ax.transAxes,ha='center')
    ax.grid(alpha=.2)
    fig.savefig(REPORT/'timeout_errors.png',dpi=180);plt.close(fig)


def write_report(protocol,metrics,stats,geometry,diagnosis,transitions):
    lines=['# 当前更大高低错落场景的成功率（v20，64个全新验证场景）','','本轮没有训练或调参，没有使用最终测试集。当前动画使用的是PPO＋预测安全＋局部恢复；纯PPO单独列出。','',
           '| 执行方式 | 安装成功 | 碰撞 | 超时 | 越限 | 安装成功率 | 成功且有上下通过 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for mode in MODES:
        m=metrics[mode]
        lines.append(f'| {LABELS[mode].replace(chr(10)," ")} | {m["success"]} | {m["collision"]} | {m["timeout"]} | {m["joint_limit"]} | {m["success_rate"]:.1%} | {m["successful_with_verified_weaving"]}/64（{m["success_and_weaving_rate"]:.1%}） |')
    lines += ['','## 统计口径','',
        '- 主成功率是原任务的安装成功：仍使用原目标、误差阈值和240步预算，同时无碰撞、无关节越限。模型77维观测/7维动作、共享环境的动作投影/参考恢复包装器保持。增强分支没有全局RRT回退；纯PPO不使用额外预测或局部动作搜索。',
        '- 次指标“成功且上下通过”要求完成安装，并且实测轨迹对至少3个相邻障碍依次有完整墙板上/下侧通过（净空至少1cm、XY距箱体中心不超过24cm），实际高度变化至少7cm且方向交替。仅安装成功并不意味着一定走了这种路线。这项诊断不参与场景筛选或改变任务奖励/成功标准。',
        '- 当前配置：基础高度偏移14–22cm、O2额外下移5cm；下移之后总高度跨度至少35cm、相邻高度差至少14cm；四个大小不一动态障碍，矢量位移上限18cm、速度上限0.12m/s。已知运动、无噪声无延迟、预测时暂停物理时间。',
        '- 场景使用当前随机动画的同一生成器：先产生旧布局的预备回合实测路线，再沿该路线构造更大错落的最终布局。预备回合成功/碰撞/超时不作为筛选条件；其结果仅记录在场景元数据。对最终布局只做几何/完整运动边界检查；64个布局全部冻结后才正式评估两种控制器。没有按最终成功或上下绕行效果挑选。',
        f'- 本轮生成尝试{geometry["attempts"]}次，几何或名义路线构造拒绝{geometry["rejected"]}次，最终统计64个合法场景；另有{geometry["unused_valid_candidates"]}个末批合法候选未用于填充配额，详见construction_attempts.json。构造失败不伪装成模型碰撞/超时，也不混入64个任务回合分母。',
        '- 种子与既有数据和演示去重。使用源模型冻结副本；配置、控制参数、源代码和数据哈希在结束后复核。两个分支在完全相同的64个布局各执行一次，共128个正式物理回合；几何构造中的预备仿真另计，不是训练数据。',
        '- 几何合法并不等于已证明每个场景在240步内都有可行路径。本轮没有运行RRT可行性基准，不能把全部失败唯一归因于策略学习不足。',
        '- 这里统计更大错落、实测路线布置的当前场景族；不是同一个固定成功录像反复播放，也不能沿用旧布局的98.4%。这是验证集估计，不是最终独立测试结论。','',
        '## 不确定性与失败类型','']
    for row in stats:
        mode=row['mode'];d=diagnosis[mode]
        lines += [f'- {LABELS[mode].replace(chr(10)," ")}：成功率Wilson95%区间{row["success_95ci_lower"]:.1%}–{row["success_95ci_upper"]:.1%}；超时类型{d["timeout_categories"]}；终态可定位碰撞对象{d["collision_pairs"]}，另有{d["collision_contact_unavailable"]}次碰撞未能在终态定位接触。']
    lines += ['', '超时分类沿用预先定义口径：两项误差均在成功阈值3倍以内为近目标精度不足；否则零命令或近静止实际步数占比≥25%为等待/停滞；其他为进展缓慢或无效。标签是诊断线索，不证明唯一因果。终态接触不存在不等于没有发生物理碰撞，原终止判定仍照计。', '',
        f'纯PPO到增强分支的配对转移：{transitions}。', '',
        '## 文件','',
        '- success_rate.png：安装成功/碰撞/超时计数及成功、成功且上下绕行两个口径。',
        '- metrics.csv：完整计数、成功率区间、等待/停滞和计算时间。',
        '- episodes.csv / paired_cases.csv / failures.csv：逐场景结果、配对和失败明细。',
        '- ppo_progress.json / ppo.json：原始汇总及逐真实步诊断轨迹。',
        '- recordings/*.json：128个正式回合的真实状态、原结果、配置与场景，可直接用scripts/replay_episode.py回放。',
        '- weaving_examples.json：成功且通过上述诊断的录像及种子，不回过头改变成功率分母。',
        '- construction_attempts.json / geometry_summary.json：全部构造尝试和拒绝原因，含预备回合结果分布。',
        '- protocol.json / frozen_model.zip / code_archive/ / validation_cases.json / verification.json：冻结模型、规则、代码、64场景及核验。',
        '- status.json / workers/*.json：运行状态。','',
        '执行耗时在4个并行CPU进程下测得，含预测与局部恢复；固定帧率回放不参与仿真结果。虚拟预测步数单列，没有当作训练步数或真实回合。']
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
