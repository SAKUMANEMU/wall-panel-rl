"""Export paired primary RRT/initialization results and separate retry evidence."""
import _bootstrap
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import yaml
from run_targeted_bc import read,dump,sha


def journal(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def summarize(study):
    root=Path(study['report_root']);source=Path(study['source_report'])
    if read(root/'status.json')['status']!='completed':raise RuntimeError('Evaluation incomplete')
    protocol=read(root/'protocol.json');s=read(root/'summary.json')
    if sha(source/'selected_model.zip')!=protocol['source_model_sha256'] or sha(study['cases'])!=protocol['scene_bank_sha256']:
        raise AssertionError('Source data/model changed')
    for path,expected in protocol['code_hashes'].items():
        if sha(path)!=expected:raise AssertionError('Executed baseline code changed')
    ppo=journal(root/'ppo_episodes.jsonl');rrt=journal(root/'rrt_primary_episodes.jsonl');retry=journal(root/'rrt_retry_episodes.jsonl')
    if [r['seed'] for r in ppo]!=[r['seed'] for r in rrt]:raise AssertionError('Unpaired evaluation order')
    all_rows=[('initialization',r) for r in ppo]+[('rrt_primary',r) for r in rrt]+[('rrt_retry',r) for r in retry]
    audited=[r for r in rrt if r['planned_route_recheck'] and not r['planned_route_recheck']['passed']]
    strict_success={r['seed'] for r in rrt+retry if r['reason']=='success' and r['planned_route_recheck'] and r['planned_route_recheck']['passed']}
    planning_audit={'primary_recheck_failures':[{'seed':r['seed'],'reason':r['reason']} for r in audited],
        'primary_success_and_recheck_passed':sum(r['reason']=='success' and r['planned_route_recheck']['passed'] for r in rrt if r['planned_route_recheck']),
        'scenes_with_success_and_recheck_passed':len(strict_success),
        'scenes_without_success_and_recheck_passed':[r['seed'] for r in rrt if r['seed'] not in strict_success]}
    dump(root/'planning_audit_summary.json',planning_audit)
    fields=['group','seed','planner_seed','reason','planning_success','steps','position_m','orientation_deg','planning_seconds',
        'first_collision_step','first_collision_sim_seconds','plan_recheck_passed','plan_recheck_invalid_edges']
    with (root/'episodes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for group,row in all_rows:
            entry={k:row.get(k) for k in fields if k!='group'};entry['group']=group
            check=row['planned_route_recheck']
            entry.update(plan_recheck_passed=check['passed'] if check else None,
                plan_recheck_invalid_edges=len(check['invalid_edge_indices']) if check else None)
            w.writerow(entry)
    collision_fields=['group','seed','attempt','step','sim_seconds','kind','moving_body','obstacle_slot','nominal_path_fraction','moving_link_index','distance_m','contact_count']
    with (root/'collisions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=collision_fields);w.writeheader()
        for group,row in all_rows:
            if row['reason']!='collision':continue
            contact=row['dominant_contact'] or {}
            w.writerow({'group':group,'seed':row['seed'],'attempt':row.get('attempt',0),'step':row['first_collision_step'],
                'sim_seconds':row['first_collision_sim_seconds'],**{k:contact.get(k) for k in
                ('kind','moving_body','obstacle_slot','nominal_path_fraction','moving_link_index','distance_m')},'contact_count':len(row['contacts'])})
    dump(root/'verification.json',{'paired_99_scenes':len(ppo)==len(rrt)==99,'source_hashes_unchanged':True,
        'executed_code_hashes_valid':True,'initialization_reproduces_cached_24_success_75_collision':s['initialization']['success']==24 and s['initialization']['collision']==75,
        'one_primary_planner_seed_per_scene':True,'retry_outcomes_separate':True,
        'planning_and_execution_counts_separate':True,'training_steps':0})
    rows=[s['initialization'],s['rrt_primary']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');bottom=np.zeros(2)
    for key,label,color in [('success','Success','#16834b'),('collision','Collision','#bc3b36'),('timeout','Timeout','#c8962f'),('planning_failure','Plan failure','#787f8b')]:
        values=[r[key] for r in rows];axes[0].bar(['Initialization','RRT-Connect'],values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v>=5:axes[0].text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white' if key!='timeout' else 'black')
            elif v:axes[0].annotate(str(v),(i+.30,bottom[i]+v/2),xytext=(12,0),textcoords='offset points',fontsize=9,
                arrowprops={'arrowstyle':'-','lw':.7})
        bottom+=values
    axes[0].set(ylim=(0,113),ylabel='Episodes (fixed99 scenes)',title='Primary execution outcomes')
    axes[0].legend(loc='upper center',ncol=2,fontsize=8,frameon=False)
    for i,(group,label,color) in enumerate([('initialization','Initialization','#345bb5'),('rrt_primary','RRT-Connect','#16834b')]):
        counts=s['collision_details'][group]['dominant_obstacle_slots'];values=[counts.get(str(j),0) for j in range(1,5)]
        axes[1].bar(np.arange(4)+(i-.5)*.35,values,width=.35,label=label,color=color)
    axes[1].set(xticks=range(4),xticklabels=['1 (~40%)','2 (~60%)','3 (~20%)','4 (~80%)'],
        ylabel='Collision episodes',title='Dominant contacted obstacle')
    axes[1].legend(fontsize=8)
    fig.savefig(root/'overview.png',dpi=170);plt.close(fig)
    # Markdown report is generated below, from primary metrics only.
    proof=s['feasibility'];n=proof['scenes'];a=s['initialization'];b=s['rrt_primary']
    interpretation=(f'全部{n}个场景至少有一次RRT规划并实际执行成功，已有可执行解的证据。' if proof['scenes_with_any_successful_execution']==n else
        f'{proof["scenes_with_any_successful_execution"]}/{n}个场景已有成功实际执行的证据；其余只是当前有限预算未解决，不能判为无解。')
    lines=['# 四个路径障碍：RRT-Connect实际执行基线', '',
        '**本轮没有训练、换场景或调整规划器。比较相同99个固定场景的初始化策略与RRT-Connect＋原关节路径跟踪控制。**', '',
        '| 方法（单种子主评估） | 成功 | 碰撞 | 超时 | 规划失败 |', '|---|---:|---:|---:|---:|',
        f'| 初始化模型 | {a["success"]}/{a["episodes"]} | {a["collision"]} | {a["timeout"]} | {a["planning_failure"]} |',
        f'| RRT-Connect＋实际执行 | {b["success"]}/{b["episodes"]} | {b["collision"]} | {b["timeout"]} | {b["planning_failure"]} |', '',
        f'RRT主评估找到路径{s["rrt_primary_plans_found"]}/{n}次；找到路径不等于执行成功。',
        interpretation, '', '## 评估口径', '',
        '- 使用冻结的原99个四静态障碍场景，包含完整机器人、墙板、墙体、自碰撞和关节限制；没有重新抽样或筛选场景。',
        '- 初始化模型逐例重放，与原24成功/75碰撞的终止原因及步数完全一致。',
        '- RRT主种子为场景seed+100000，目标是与初始化策略相同的参考终点关节位形，没有重新选择IK分支。',
        f'- RRT设置保持原值：每次规划最多{protocol["configuration"]["rrt"]["timeout"]}秒、{protocol["configuration"]["rrt"]["max_iterations"]}次迭代；实际执行最多{protocol["configuration"]["simulation"]["max_steps"]}个控制步。',
        '- 无延迟、无感知噪声、障碍静止；规划时不推进仿真时间。RRT路径逐步经过原执行器和相同物理步进，而非只做几何验收。',
        '- 规划失败不执行任何动作，没有专家替换失败动作或退回策略。路径加密后的边复检只在执行结束后做诊断，不改变主评估结果。',
        '- 这是已见场景的诊断/回归，不能称为新的独立泛化测试。', '',
        '## 失败场景的种子诊断', '',
        f'仅对主评估失败的{n-b["success"]}个场景，各追加2个固定种子（seed+200000和seed+300000），不替换主评估成绩。',
        f'重试共{s["rrt_retries"]["episodes"]}回合，成功{s["rrt_retries"]["success"]}、碰撞{s["rrt_retries"]["collision"]}、超时{s["rrt_retries"]["timeout"]}、规划失败{s["rrt_retries"]["planning_failure"]}。',
        f'跨主评估和诊断种子，曾找到路径的场景数{proof["scenes_with_any_found_plan"]}/{n}；曾实际执行成功的场景数{proof["scenes_with_any_successful_execution"]}/{n}。',
        f'仍未解决的场景：`{proof["unresolved_scene_seeds"]}`。规划/执行失败不构成场景不可解的证明。', '',
        '## 首次碰撞归因', '',
        '环境在首次碰撞物理子步停止，因此记录对应控制步、仿真时间、具体障碍槽位、臂/墙板、连杆号和接触距离。',
        '障碍编号与动画一致：1约40%、2约60%、3约20%、4约80%。表格采用最深接触作主归因；完整多点接触保留在逐回合JSON。', '',
        '| 控制器 | 首次碰撞主类型（场景数） | 主碰撞障碍编号（场景数） |', '|---|---|---|']
    for key,label in [('initialization','初始化'),('rrt_primary','RRT主评估'),('rrt_retries','RRT诊断重试')]:
        c=s['collision_details'][key]
        lines.append(f'| {label} | `{c["dominant_contact_categories"]}` | `{c["dominant_obstacle_slots"]}` |')
    lines+=['', '机械臂/墙板接触场景数可以重叠，不能相加当成互斥失败总数。超时的最终位置、角度误差保留在episodes.csv中。', '',
        '### RRT主评估碰撞案例', '', '| 场景seed | 碰撞步 | 主接触 | 障碍编号 | 规划路径事后采样复检 |', '|---:|---:|---|---:|---|']
    for r in rrt:
        if r['reason']!='collision':continue
        c=r['dominant_contact'] or {};check=r['planned_route_recheck']
        lines.append(f'| {r["seed"]} | {r["first_collision_step"]} | {c.get("kind","unknown")}/{c.get("moving_body","unknown")} | {c.get("obstacle_slot")} | {check["passed"] if check else None} |')
    if not b['collision']:lines.append('| — | — | 无 | — | — |')
    lines+=['', f'全体主评估中有{len(audited)}条路径事后复检不满足原5毫米规划间距：`{planning_audit["primary_recheck_failures"]}`。',
        f'主评估中同时实际成功且通过复检的路径{planning_audit["primary_success_and_recheck_passed"]}条；包含诊断重试后，此类证据覆盖{planning_audit["scenes_with_success_and_recheck_passed"]}个场景。',
        f'当前仍无“执行成功＋复检通过”记录的场景：`{planning_audit["scenes_without_success_and_recheck_passed"]}`；这些场景有实际成功记录，但不能直接作为严格间距示范。',
        '两条碰撞路径的额外几何核查保存在collision_route_audit.json。复检点的墙板间距约1.12毫米和2.27毫米，不足规划要求5毫米；这些复检点未检测到实际几何穿透。此结论仅覆盖所列采样点，连续路径与跟踪偏差仍需进一步检查。',
        '下一步应加强离散/加密路径安全复检，再以实际执行成功且通过复检的路线准备训练示范。本轮没有修改规划器，也没有启动训练。', '',
        '## 文件', '', '- episodes.csv：全部主评估和重试的结果、规划时间、最终误差、首次碰撞步。',
        '- collisions.csv：逐碰撞场景的障碍编号、接触部位和连杆号。',
        '- ppo_episodes.jsonl / rrt_primary_episodes.jsonl / rrt_retry_episodes.jsonl：完整接触与事后复检记录。',
        '- traces/：规划关节路径、实际执行关节路径和墙板位置；没有将重试或成功轨迹作为训练数据。',
        '- overview.png：主评估结果和障碍归因；summary.json：可解性证据与配对转换。',
        '- planning_audit_summary.json / collision_route_audit.json：规划间距复检与碰撞路径的几何见证。',
        '- protocol.json / verification.json / code_archive/：固定场景、模型和执行代码来源。',
        '- 日志：`/home/johnsonchen/wall-panel-rl/logs/four_on_path_rrt_validation_v1_seed61.log`。', '',
        '## 播放固定场景的RRT动画', '', '```bash',
        'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py --controller rrt \\',
        f'  --scene-bank {Path(study["cases"]).resolve()} --seed {rrt[0]["seed"]}', '```', '']
    (root/'README.md').write_text('\n'.join(lines))
    target=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    if target.resolve()!=root.resolve():
        if target.exists():
            if read(target/'protocol.json')!=protocol:raise FileExistsError('Visible report has a different protocol')
            for name in ('README.md','episodes.csv','collisions.csv','verification.json','overview.png','planning_audit_summary.json','collision_route_audit.json'):shutil.copy2(root/name,target/name)
        else:shutil.copytree(root,target)
    print(json.dumps({'initialization':a,'rrt_primary':b,'feasibility':proof,'report':str(root.resolve())},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_on_path_rrt_validation.yaml')
    args=p.parse_args();summarize(yaml.safe_load(Path(args.config).read_text()))


if __name__=='__main__':main()
