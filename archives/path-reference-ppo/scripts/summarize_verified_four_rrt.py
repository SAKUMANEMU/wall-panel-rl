"""Compare immutable original and pre-execution-verified RRT regressions."""
import _bootstrap
import json
from pathlib import Path
import shutil
import csv
import numpy as np
from run_targeted_bc import read,dump,sha


def main():
    root=Path('results/diagnostics/four_on_path_rrt_verified_v2_seed61')
    previous=root.parent/'four_on_path_rrt_validation_v1_seed61'
    if read(root/'status.json')['status']!='completed':raise RuntimeError('Verified regression incomplete')
    protocol=read(root/'protocol.json');s=read(root/'summary.json');old=read(previous/'summary.json')
    for path,expected in protocol['code_hashes'].items():
        if sha(path)!=expected:raise AssertionError('Executed source changed')
    if sha(protocol['study']['cases'])!=protocol['scene_bank_sha256']:raise AssertionError('Regression bank changed')
    rows=[json.loads(line) for line in (root/'rrt_primary_episodes.jsonl').read_text().splitlines()]
    old_rows=[json.loads(line) for line in (previous/'rrt_primary_episodes.jsonl').read_text().splitlines()]
    times={name:{'mean_s':float(np.mean([r['planning_seconds'] for r in values])),
                 'median_s':float(np.median([r['planning_seconds'] for r in values])),
                 'maximum_s':max(r['planning_seconds'] for r in values)}
           for name,values in [('original',old_rows),('verified',rows)]}
    dump(root/'planning_time.json',times)
    if not all(r['static_path_verified'] and r['planned_route_recheck']['passed'] for r in rows if r['planning_success']):
        raise AssertionError('A reported valid path failed its verification')
    metrics=[{'version':name,**value} for name,value in [('old_rrt',old['rrt_primary']),('verified_rrt',s['rrt_primary'])]]
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['version','episodes','success','collision','timeout','joint_limit','planning_failure']);w.writeheader();w.writerows(metrics)
    dump(root/'verification.json',{'all_found_primary_paths_preverified':True,'all_found_primary_paths_postcheck_passed':True,
        'regression_case_bank_unchanged':True,'original_initialization_reproduced':s['initialization']==old['initialization'],
        'training_steps':0,'finite_sampling_resolution_rad':.005,'continuous_collision_certificate':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.7),layout='constrained');bottom=np.zeros(2)
    for key,label,color in [('success','Success','#16834b'),('collision','Collision','#bc3b36'),('planning_failure','Plan failure','#808080'),('timeout','Timeout','#c8962f')]:
        values=[r[key] for r in metrics];ax.bar(['Original RRT','Preverified RRT'],values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v>=5:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white')
            elif v:ax.annotate(str(v),(i+.3,bottom[i]+v/2),xytext=(12,0),textcoords='offset points',arrowprops={'arrowstyle':'-'})
        bottom+=values
    ax.set(ylim=(0,113),ylabel='Episodes (same99 scenes)',title='Static RRT: original vs execution-waypoint validation')
    ax.legend(loc='upper center',ncol=2,fontsize=8,frameon=False);fig.savefig(root/'overview.png',dpi=170);plt.close(fig)
    a,b=metrics
    lines=['# 四障碍：RRT执行前安全复检修复', '',
        '**原5毫米规划间距、15秒搜索预算、240步实际执行上限保持不变。初始化策略和物理碰撞阈值未改。**', '',
        '| 99场景主评估 | 成功 | 碰撞 | 超时 | 规划失败 |', '|---|---:|---:|---:|---:|',
        f'| 原RRT | {a["success"]}/99 | {a["collision"]} | {a["timeout"]} | {a["planning_failure"]} |',
        f'| 执行前复检RRT | {b["success"]}/99 | {b["collision"]} | {b["timeout"]} | {b["planning_failure"]} |', '',
        '- 增加原采样网格之外的0.005弧度细采样，并覆盖控制器加密后的实际路点。',
        '- 直线路径、捷径和完整候选路线均经过验证；不合格候选继续搜索，预算耗尽返回规划失败。控制器另有最终拒绝防线。',
        '- 全部找到的主评估路径均在执行前验证通过，事后原网格复检也通过。有限采样不构成连续碰撞安全证明。',
        f'- 主评估规划成功{s["rrt_primary_plans_found"]}/99；额外诊断重试结果：`{s["rrt_retries"]}`，不替换主成绩。',
        f'- 包含诊断重试后，已有实际成功解的场景数{s["feasibility"]["scenes_with_any_successful_execution"]}/99。',
        '- 更密检查增加规划墙钟时间；规划期间仿真时钟仍暂停，没有模拟延迟或改变执行器。', '',
        f'- 平均规划墙钟耗时由{times["original"]["mean_s"]:.2f}秒增至{times["verified"]["mean_s"]:.2f}秒，最大{times["verified"]["maximum_s"]:.2f}秒。15秒是搜索预算，完整耗时还包含最终路线复检，可超过15秒。', '',
        '## 下一阶段的数据', '',
        '新的示范数据使用额外10毫米规划余量，独立于本回归配置；只保存路径复检通过并实际执行成功的训练路线。',
        '原99例仍仅作回归。新增训练/验证/测试种子分别分开，未启动BC或PPO。', '',
        '- comparison.csv / overview.png：原RRT与修复版主成绩。',
        '- rrt_primary_episodes.jsonl / traces/：逐场景、规划与实际执行路径；原报告未改写。',
        '- protocol.json / code_archive/ / verification.json：代码、配置和验证来源。',
        '- 详细设计：docs/VERIFIED_FOUR_RRT.md。', '']
    (root/'README.md').write_text('\n'.join(lines))
    dest=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    if dest.resolve()!=root.resolve():
        if dest.exists():
            for name in ('README.md','comparison.csv','overview.png','verification.json','planning_time.json'):shutil.copy2(root/name,dest/name)
        else:shutil.copytree(root,dest)
    print(json.dumps({'old':a,'verified':b,'report':str(root.resolve())},indent=2))


if __name__=='__main__':main()
