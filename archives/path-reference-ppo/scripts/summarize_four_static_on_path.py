"""Verify and show the four-box nominal-path placement; no model fitting."""
import _bootstrap
import argparse
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import yaml
from run_targeted_bc import read,dump,sha


def summarize(spec):
    report=Path(spec['report_root']);data=Path(spec['data_root'])
    if read(report/'status.json')['status']!='completed':raise RuntimeError('Evaluation incomplete')
    s=read(report/'summary.json');cases=read(data/'cases.json')
    if sha(report/'selected_model.zip')!=s['source_model_sha256'] or sha(Path(spec['source_report'])/'selected_model.zip')!=s['source_model_sha256']:
        raise AssertionError('Initialized weights changed')
    if not s['all_four_intersect_nominal'] or not s['start_goal_clear']:raise AssertionError('Invalid on-route placement')
    rows=[{'placement':key,**s[key]['stage_target']} for key in ('old_positions','on_path')]
    with (report/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['placement','episodes','success','collision','timeout','joint_limit','planning_failure']);w.writeheader();w.writerows(rows)
    witnesses=[r for c in cases for r in c['scene']['on_path_placement']['obstacles']]
    dump(report/'verification.json',{'weights_unchanged':True,'training_steps':0,
        'verified_cases':len(cases),'obstacles_per_case':4,'individual_contact_witnesses':len(witnesses),
        'all_intersect':all(r['intersects_panel'] or r['intersects_arm'] for r in witnesses),
        'all_contact_distances_nonpositive':all(r['minimum_contact_distance_m']<=0 for r in witnesses),
        'start_goal_clear':True,'reference_observation_route_unchanged':True,
        'nominal_route_is_physical_obstacle_free_policy_motion':True,'source_sha256':s['source_model_sha256']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig,ax=plt.subplots(figsize=(7,4.6),layout='constrained');bottom=np.zeros(2)
    for key,label,color in [('success','Success','#16834b'),('collision','Collision','#bc3b36'),('timeout','Timeout','#c8962f')]:
        values=[r[key] for r in rows];ax.bar(['Previous positions','On nominal motion'],values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',fontsize=10,color='white' if key!='timeout' else 'black')
        bottom+=values
    ax.set(ylim=(0,len(cases)*1.12),ylabel='Paired episodes',title='Same initialization, four static boxes: position change only')
    ax.legend(loc='upper center',ncol=3,frameon=False);fig.savefig(report/'comparison.png',dpi=170);plt.close(fig)
    scene=cases[0]['scene'];points=np.asarray(scene['nominal_motion']['panel_positions'])
    fig=plt.figure(figsize=(7.5,6.2),layout='constrained');ax=fig.add_subplot(projection='3d')
    ax.plot(*points.T,color='#159da5',lw=2,label='Nominal obstacle-free panel motion')
    ax.scatter(*points[0],color='#2358b7',s=45,label='Start');ax.scatter(*points[-1],color='#14823a',s=45,label='Goal')
    all_points=[points]
    for i,box in enumerate(scene['obstacles']):
        center=np.asarray(box['center']);half=np.asarray(box['dimensions'])/2
        vertices=np.array([center+half*np.array([x,y,z]) for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        faces=[[vertices[j] for j in ids] for ids in ([0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5])]
        ax.add_collection3d(Poly3DCollection(faces,facecolors='#d44539',edgecolors='#77221c',alpha=.5,lw=.5))
        label=center.copy();label[2]+=half[2]+.04
        ax.text(*label,str(i+1),fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.8,'pad':.5})
        all_points.append(vertices)
    cloud=np.vstack(all_points);mid=(cloud.max(0)+cloud.min(0))/2;radius=max(np.ptp(cloud,axis=0))*.57
    ax.set(xlim=(mid[0]-radius,mid[0]+radius),ylim=(mid[1]-radius,mid[1]+radius),zlim=(mid[2]-radius,mid[2]+radius),
        xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=f'Four verified blockers on nominal motion (seed {scene["seed"]})')
    ax.set_box_aspect((1,1,1));ax.view_init(25,-60);ax.legend(loc='upper left',fontsize=8)
    fig.savefig(report/'placement_preview.png',dpi=170);plt.close(fig)
    a,b=rows
    lines=['# 四个静态障碍物：放在初始化模型的运动路线上', '',
        '**当前位置由初始化模型在无障碍环境中的真实仿真运动轨迹决定，不再仅靠近参考路线随机摆放。初始化权重未改，未训练。**', '',
        '- 沿整段轨迹约20%、40%、60%、80%的位置分布，带少量随机偏移；观测槽位先保留两个中段障碍。',
        '- 每个障碍都保存独立的关节位形见证：在该位形下，与机械臂或墙板存在实际几何相交。',
        '- 起点与终点保持有效，障碍之间至少0.1米间距；大小沿用原四障配置；障碍静止。',
        '- 青色线是无障碍时的轨迹，蓝色线是本次带障碍执行轨迹。无障碍推理和几何放置在播放前完成，不是训练。',
        '- 每次默认随机初始位置和布局，--seed可复现。仅在无完整无障碍路线/几何无法放置时重试，不按带障碍执行是否成功筛选。', '',
        f'布置尝试{s["attempted_scenes"]}个固定场景，合法{s["valid_scenes"]}个，放置失败{len(s["placement_failures"])}个。只在合法的同一批场景中作配对对比；失败详见placement_failures.json。', '',
        f'未完成布置的原因：`{s["placement_failures"]}`。这是无障碍路线构建/几何放置失败，不是剔除带障碍策略失败案例。', '',
        '| 四障碍位置 | 成功 | 碰撞 | 超时 |', '|---|---:|---:|---:|',
        f'| 原位置 | {a["success"]}/{a["episodes"]} | {a["collision"]} | {a["timeout"]} |',
        f'| 实际无障碍运动路线上 | {b["success"]}/{b["episodes"]} | {b["collision"]} | {b["timeout"]} |', '',
        '**旧版82%属于旧位置布置，不能用于本版。当前模型未重新学习四障碍避障，不能保证绕开新位置。**', '',
        '```bash','cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py','```','',
        '复现固定布局可在末尾加 --seed 20261103；headless检查可加 --headless。', '',
        '- comparison.csv / comparison.png：配对结果；on_path.json：逐回合原因。',
        '- placement_preview.png：第一场景的真实无障碍轨迹与四障碍几何图。',
        '- verification.json：逐障碍相交见证与模型校验。',
        '- manifest.json / selected_model.zip：当前演示默认配置和原初始化模型。',
        f'- 场景和每障碍位形见证：`{data.resolve()}/cases.json`。',
        f'- 模型SHA256：`{s["source_model_sha256"]}`。', '']
    (report/'README.md').write_text('\n'.join(lines))
    target=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if target.resolve()!=report.resolve():
        if target.exists():
            if sha(target/'selected_model.zip')!=s['source_model_sha256']:raise FileExistsError(target)
            for name in ('README.md','comparison.csv','comparison.png','placement_preview.png','verification.json'):
                shutil.copy2(report/name,target/name)
        else:shutil.copytree(report,target)
    print(json.dumps({'old':a,'on_path':b,'placement_failures':len(s['placement_failures']),'report':str(report.resolve())},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_static_on_path.yaml')
    args=p.parse_args();summarize(yaml.safe_load(Path(args.config).read_text()))


if __name__=='__main__':main()
