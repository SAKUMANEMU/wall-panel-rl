"""Export the count-only initialization comparison; no model fitting."""
import _bootstrap
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import yaml
from run_targeted_bc import read, dump, sha


def summarize(study):
    report=Path(study['report_root']);data=Path(study['data_root'])
    if read(report/'status.json')['status']!='completed': raise RuntimeError('Evaluation has not completed')
    s=read(report/'summary.json'); manifest=read(report/'manifest.json')
    if sha(study['initial_model'])!=s['source_sha256'] or sha(report/'selected_model.zip')!=s['source_sha256']:
        raise AssertionError('Initialized model changed')
    two=read(report/'two_static.json');four=read(report/'four_static.json')
    old={r['seed']:r for r in two['rows'] if r['suite']=='stage_target'}
    new={r['seed']:r for r in four['rows'] if r['suite']=='stage_target'}
    if set(old)!=set(new): raise AssertionError('Unpaired count-change evaluation')
    transition=Counter(f'{old[seed]["reason"]} -> {new[seed]["reason"]}' for seed in old)
    dump(report/'paired_outcomes.json',{'episodes':len(old),'transitions':dict(transition)})
    audit=read(report/'placement_audit.json')
    dump(report/'verification.json',{'initial_model_sha256':s['source_sha256'],'weights_byte_identical':True,
        'four_static_bodies_in_all_cases':all(a['obstacles']==4 and a['static'] for a in audit),
        'start_goal_valid_in_all_cases':all(a['start_valid'] and a['goal_valid'] for a in audit),
        'minimum_box_gap_m':min(a['minimum_box_gap_m'] for a in audit),
        'configuration_changes_only_active_stage':True,'training_steps':0,
        'new_obstacle_inputs_ignored_by_initialization':s['new_obstacle_inputs_ignored_by_initialization'],
        'seen_underlying_bank':True,'independent_test':False})
    rows=[{'obstacles':n,**s[key]['stage_target']} for n,key in [(2,'two_static'),(4,'four_static')]]
    with (report/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['obstacles','episodes','success','collision','timeout','joint_limit','planning_failure'])
        writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.8),layout='constrained');bottom=np.zeros(2)
    for key,label,color in [('success','Success','#16834b'),('collision','Collision','#bc3b36'),('timeout','Timeout','#c8962f')]:
        values=[r[key] for r in rows];ax.bar(['2 static obstacles','4 static obstacles'],values,bottom=bottom,label=label,color=color)
        for i,v in enumerate(values):
            if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white' if key!='timeout' else 'black')
        bottom+=values
    ax.set(ylim=(0,110),ylabel='Episodes (same 100 scenes)',title='Retained initialization: obstacle count only')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.02),ncol=3,frameon=False)
    fig.savefig(report/'overview.png',dpi=170);plt.close(fig)
    a,b=rows;lines=['# 初始化模型：四个静态障碍物', '',
        '**已切换当前默认演示至四个静态障碍，使用保留的初始化权重。没有BC/PPO再训练。**', '',
        '| 障碍数 | 成功 | 碰撞 | 超时 |', '|---:|---:|---:|---:|',
        f'| 2 | {a["success"]}/{a["episodes"]} | {a["collision"]} | {a["timeout"]} |',
        f'| 4 | {b["success"]}/{b["episodes"]} | {b["collision"]} | {b["timeout"]} |', '',
        f'逐场景变化：`{dict(transition)}`。复用已见的v4底层100场景作为障碍数量变化的配对回归，不是新独立测试。', '',
        '## 场景与模型', '',
        '- 四个障碍均为静态，大小不一。保留原双障槽位顺序，再增加前段与后段障碍，原索引顺序[2,3,1,4]。',
        '- 观测仍为77维、动作7维；保留5个障碍信息槽位，最后一个槽位为空，以兼容初始化模型。',
        '- 初始位置、障碍位置默认每次随机生成；--seed固定场景。只因几何放置无效而重试，不按策略能否成功筛选。',
        '- 目标位置、机器人、墙板尺寸、奖励和240步上限沿用源配置；播放速度0.6倍，无感知噪声。',
        '- 新场景的无障碍参考在播放前准备；控制执行期间纯策略没有在线避障规划器。固定场景库则直接复用已有参考。',
        '- 初始化权重对新增障碍槽位的连接及独立修正输出仍为零，因此该模型尚未学会主动利用其余三个障碍的信息。82%仅是当前场景下的表现，不是四障碍训练成果。'.replace('82%',f'{b["success"]}%'), '',
        f'原模型：`{Path(study["initial_model"]).resolve()}`；SHA256：`{s["source_sha256"]}`。复制模型与原模型逐字节一致。', '',
        '## 终端演示', '',
        '```bash', 'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py', '```', '',
        '指定种子复现：', '', '```bash',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_static.py --seed 20261103', '```', '',
        '原demo_autonomous_ppo.py入口也默认使用本配置。需要历史双障演示可运行demo_consistent_two_static.py；需复用固定四障场景可加：', '',
        '```bash', f'--scene-bank {data.resolve()}/four_cases.json --seed 1018000', '```', '',
        '## 文件', '', '- comparison.csv / overview.png：配对计数与图；four_static.json：逐回合结果。',
        '- manifest.json + selected_model.zip：四障环境配置与初始化模型。',
        '- placement_audit.json / verification.json：全部四障实体、初末状态及模型来源核验。', '']
    (report/'README.md').write_text('\n'.join(lines))
    code=['configs/four_static_initial.yaml','scripts/run_four_static_initial.py','scripts/demo_autonomous_ppo.py',
        'scripts/demo_four_static.py','src/wall_panel_rl/rl/static_demo_scene.py','src/wall_panel_rl/rl/avoidance_curriculum.py']
    dump(report/'code_hashes.json',{name:sha(name) for name in code})
    for name in code:
        dest=report/'code_archive'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    dest=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if dest.resolve()!=report.resolve():
        if dest.exists():
            if sha(dest/'selected_model.zip')!=s['source_sha256']:raise FileExistsError(dest)
            for name in ('README.md','comparison.csv','overview.png','verification.json','paired_outcomes.json','code_hashes.json'):
                shutil.copy2(report/name,dest/name)
        else:shutil.copytree(report,dest)
    print(json.dumps({'two':a,'four':b,'model_unchanged':True,'training_steps':0,'report':str(report.resolve())},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_static_initial.yaml')
    args=p.parse_args();summarize(yaml.safe_load(Path(args.config).read_text()))


if __name__=='__main__':main()
