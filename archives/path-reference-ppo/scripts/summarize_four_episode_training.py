"""Finalize completed mixed-training results without replacing the source."""
import _bootstrap
from collections import Counter
import csv
from pathlib import Path
import json
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha


def main():
    torch.set_num_threads(1);root=Path('results/diagnostics/four_episode_mix_v6_seed61')
    status=read(root/'status.json');assert status['status']=='completed'
    s,p=read(root/'summary.json'),read(root/'protocol.json');spec=p['spec'];models=Path(spec['model_root'])
    for name,expected in p['code_hashes'].items():assert sha(root/'code_archive'/name)==expected
    assert sha(spec['source_model'])==p['source_sha256']
    assert sha(Path(spec['data_root'])/'validation_cases.json')==p['validation_case_sha256']
    history=read(root/'history.json');assert [h['ppo_steps'] for h in history]==[0,5000,10000]
    rows=[]
    for h in history:
        row={'ppo_steps':h['ppo_steps'],'selected':h['ppo_steps']==s['selected']['ppo_steps']}
        for mode in ('ppo','mixture'):row.update({mode+'_'+k:v for k,v in h[mode].items()})
        rows.append(row)
    with (root/'checkpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for mode in ('ppo','mixture'):
        initial=read(root/'validation'/f'{mode}_0.json')['rows']
        final=read(root/'validation'/f'{mode}_10000.json')['rows']
        assert [(r['seed'],r['controller']) for r in initial]==[(r['seed'],r['controller']) for r in final]
        dump(root/f'paired_{mode}.json',dict(Counter(a['reason']+' -> '+b['reason'] for a,b in zip(initial,final))))
    training=[json.loads(line) for file in models.glob('episodes_worker_*.jsonl') for line in file.read_text().splitlines()]
    assert all(r['seed'] in p['train_seeds'] and r['seed'] not in p['excluded_seeds'] for r in training)
    assert sum(r['steps'] for r in training if r['controller']=='rrt')==s['rrt_transitions']
    assert all(r['pre_verified'] and r['post_verified'] and r['reference_preserved'] for r in training if r.get('accepted_for_imitation'))
    original=PPO.load(spec['source_model'],device='cpu');selected=PPO.load(root/'selected_model.zip',device='cpu')
    same=all(torch.equal(v,selected.policy.state_dict()[k]) for k,v in original.policy.state_dict().items())
    assert s['selected']['ppo_steps']==0 and same
    status['completed']={name:sum(s['completed_by_controller'][name].values()) for name in ('rrt','ppo')}
    status['episode_starts']=s['episode_starts'];status['process_exited']=True;dump(root/'status.json',status)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    colors={'success':'#26865b','collision':'#bd4246','timeout':'#c18f2b'}
    x=[h['ppo_steps'] for h in history]
    for ax,mode,title in [(axes[0],'ppo','Pure PPO:64 validation episodes'),(axes[1],'mixture','Episode mixture:64 validation episodes')]:
        for key,color in colors.items():ax.plot(x,[h[mode][key] for h in history],'o-',label=key,color=color)
        ax.axvline(0,ls='--',color='#666666',label='Selected:0 added steps')
        ax.set(title=title,xlabel='Additional PPO transitions',ylabel='Episodes',ylim=(-1,66))
        ax.legend(frameon=False,fontsize=8)
    fig.savefig(root/'curves.png',dpi=175);plt.close(fig)
    lines=['# 四障碍RRT/PPO正式混合训练已完成','',
        '**100%完成：PPO10,000步 + RRT10,131步 =20,131次物理转移。后台进程已退出。**','',
        '| 新增PPO步数 | 纯PPO成功/碰撞/超时 | 混合成功/碰撞/超时 |','|---:|---|---|']
    for h in history:
        a,b=h['ppo'],h['mixture'];lines.append(f'| {h["ppo_steps"]} | {a["success"]}/{a["collision"]}/{a["timeout"]} | {b["success"]}/{b["collision"]}/{b["timeout"]} |')
    lines.extend(['','**没有选中本轮更新候选；保留0步原模型。选中权重与原PPOv5全部逐项相同。**',
        '- 选中模型的混合验证为60/64（93.75%），纯PPO为56/64（87.5%）。混合优势在训练前已存在，不是本轮训练带来的提升。',
        '- 最后候选的混合验证为55/64（85.94%）：RRT24/24成功，PPO31/40成功、7碰撞、2超时。完整纯PPO验证50/64成功、11碰撞、3超时。',
        '- 每episode独立50%概率抽签，整回合固定；固定评估种子得到24个RRT和40个PPO，同一分配用于全部检查点。',
        '- 正式训练启动149个RRT和145个PPO回合；完成149个RRT（全成功）、141个PPO（57成功/73碰撞/11超时），另4个PPO在预算结束时未完成。',
        '- 独立BC辅助更新10次均通过KL检查，没有拒绝。专家轨迹只在执行成功且前后复检通过时入池，PPO缓冲区不混入专家动作。',
        '- 这些为64例验证结果，不是全新独立测试；原84/100独立测试成绩没有在本轮重新测量，也不能把93.75%归为纯PPO。',
        '- 源模型及旧演示保持不变。selected_model.zip是原策略的选中副本；final_candidate.zip是退化的末候选，不能与选中模型混用。',
        '- curves.png/checkpoints.csv：完整三检查点曲线和计数。history.json及validation/：原始逐场景评估。',
        '- summary.json：回合启动/完成数、两分支物理步数及辅助更新次数。paired_ppo.json/paired_mixture.json：初末场景变化。',
        f'- 训练日志：`/home/johnsonchen/wall-panel-rl/logs/four_episode_mix_v6_seed61.log`。',
        f'- 模型与优化器：`{models.resolve()}`。',''])
    (root/'README.md').write_text('\n'.join(lines))
    dump(root/'verification.json',{'executed_archive_matches':True,'source_file_unchanged':True,
        'selected_weights_identical_to_source':same,'train_heldout_isolation':True,'rrt_verified_labels_only':True,
        'ppo_transitions':s['ppo_transitions'],'rrt_transitions':s['rrt_transitions'],'independent_test':False})
    dest=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    if dest.resolve()!=root.resolve():shutil.copytree(root,dest,dirs_exist_ok=True)
    print({'selected_ppo_steps':s['selected']['ppo_steps'],'completed':status['completed'],'report':str(dest)})


if __name__=='__main__':main()
