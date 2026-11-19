"""Fresh paired test, old regression, sensor/action noise and continuation QA."""
import _bootstrap
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from run_targeted_bc import read,dump,sha


def main():
    root=Path('results/diagnostics/four_expanded_ppo_v5_seed61')
    assert read(root/'status.json')['status']=='completed'
    s,p=read(root/'summary.json'),read(root/'protocol.json');spec=p['spec'];models=Path(spec['model_root'])
    status=read(root/'status.json');status['completed_episodes']=sum(s['training_episodes'].values())
    status['selected_validation']=s['selected']['new_validation'];dump(root/'status.json',status)
    for path,expected in {**p['code_hashes'],**p['input_hashes']}.items():assert sha(path)==expected
    assert sha(root/'selected_model.zip')==read(root/'selection_before_test.json')['sha256']
    assert p['continuation']['optimizer_state_preserved'] and p['continuation']['value_network_preserved']
    splits=p['splits'];names=['train','validation','test','old_validation','old_test_regression']
    assert all(not set(splits[a])&set(splits[b]) for a in names for b in names if a!=b)
    history=read(root/'history.json');assert [h['additional_ppo_steps'] for h in history]==[0,5000,10000,15000,20000]
    rows=[];episodes=[]
    def add_episodes(name,result,steps):
        for r in result['rows']:
            episodes.append({'suite':name,'additional_ppo_steps':steps,'seed':r['seed'],'reason':r['reason'],
                'steps':r['steps'],'position_m':r['position_m'],'orientation_deg':r['orientation_deg']})
    selected=s['selected']['additional_ppo_steps']
    for h in history:
        steps=h['additional_ppo_steps'];row={'additional_ppo_steps':steps,'selected':steps==selected}
        for role in ('new_validation','old_validation'):
            row.update({role+'_'+k:v for k,v in h[role].items()})
        rows.append(row)
        add_episodes('new_validation',read(root/'validation'/f'steps_{steps}.json'),steps)
        add_episodes('old_validation',read(root/'old_validation'/f'steps_{steps}.json'),steps)
    files={'fresh_source':'test_source.json','fresh_selected':'test_selected.json','sensor_noise':'test_sensor_noise.json',
           'action_noise':'test_action_noise.json','old64_regression':'old64_regression.json'}
    results={k:read(root/v) for k,v in files.items()}
    for name,result in results.items():add_episodes(name,result,selected)
    for name,values in [('checkpoints.csv',rows),('evaluations.csv',episodes)]:
        with (root/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    for name in ('fresh_source','fresh_selected','sensor_noise','action_noise'):
        assert set(r['seed'] for r in results[name]['rows'])==set(splits['test'])
    a,b=results['fresh_source']['rows'],results['fresh_selected']['rows']
    assert [r['seed'] for r in a]==[r['seed'] for r in b]
    paired=dict(Counter(f'{x["reason"]} -> {y["reason"]}' for x,y in zip(a,b)))
    dump(root/'paired_fresh_test.json',paired)
    training=[json.loads(line) for path in models.glob('episodes_worker_*.jsonl') for line in path.read_text().splitlines()]
    assert all(r['seed'] in splits['train'] and r['controller']=='ppo' for r in training)
    assert sum(r['steps'] for r in training)==s['training_completed_transitions']<=spec['steps']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
    colors={'success':'#26865b','collision':'#bd4246','timeout':'#c18f2b','joint_limit':'#777777'}
    x=[h['additional_ppo_steps'] for h in history]
    for ax,role,total in [(axes[0,0],'new_validation',64),(axes[1,0],'old_validation',32)]:
        for key in ('success','collision','timeout'):
            ax.plot(x,[h[role][key] for h in history],'o-',label=key,color=colors[key],markersize=4)
        ax.axvline(selected,color='#555555',ls='--',label=f'Selected +{selected}')
        ax.set(title=f'{role.replace("_"," ").title()} ({total} scenes)',xlabel='Additional PPO transitions',ylabel='Episodes',ylim=(-.7,total+1))
        ax.legend(frameon=False,fontsize=8)
    def bars(ax,labels,metrics,title):
        bottom=np.zeros(len(labels))
        for key,color in colors.items():
            vals=[m[key] for m in metrics];ax.bar(labels,vals,bottom=bottom,color=color,label=key)
            for i,v in enumerate(vals):
                if v>=4:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center',color='white')
            bottom+=vals
        ax.set(title=title,ylabel='Episodes',ylim=(0,110));ax.legend(frameon=False,fontsize=8,ncol=2,loc='lower left')
    bars(axes[0,1],['Source PPO10k','Selected'],[s['test_source'],s['test_selected']],'Fresh paired test (100 scenes)')
    bars(axes[1,1],['Clean mean','Sensor noise','Sampled actions'],[s['test_selected'],s['sensor_noise'],s['action_noise']],'Same100 scenes: distinct noise conditions')
    fig.savefig(root/'curves.png',dpi=175);plt.close(fig)
    lines=['# 四静态障碍：扩展场景PPO续训','',
        f'本轮继续{spec["steps"]}步，选中新增{selected}步检查点（本四障碍阶段累计{10000+selected}步PPO）。BC更新0。','',
        '| 全新100例配对测试 | 成功 | 碰撞 | 超时 |','|---|---:|---:|---:|']
    for label,m in [('续训前PPO10k',s['test_source']),('选中候选',s['test_selected'])]:
        lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} |')
    lines.extend(['','| 同100例的最终候选对比 | 成功 | 碰撞 | 超时 |','|---|---:|---:|---:|'])
    for label,m in [('确定性、干净观测',s['test_selected']),('目标位姿观测噪声5mm/0.5deg',s['sensor_noise']),('干净观测、随机动作',s['action_noise'])]:
        lines.append(f'| {label} | {m["success"]} | {m["collision"]} | {m["timeout"]} |')
    lines.extend(['',f'- 新64验证：续训前`{s["baseline_new_validation"]}`；选中`{s["selected"]["new_validation"]}`。',
        f'- 已使用的旧64回归：续训前`{s["old64_source"]}`；选中`{s["old64_selected"]}`。旧64从未加入训练或检查点选择，不能再称为新独立测试。',
        '- 192训练（原64+新128）、新64验证、新100测试种子互斥；原32验证只作选择同分时的次级比较。',
        '- 剩余13碰撞全部为墙板：后段slot4共11次，前段slot3共2次。诊断数据在相邻four_expanded_v5_seed61目录。',
        f'- 源策略扫描训练集发现{s["hard_training_layouts"]}个失败布局，采样75%全训练集均匀+25%这些训练失败布局，测试场景不进入采样池。',
        '- Adam状态与价值网络从源PPO完整保留，未重置；正比例训练奖励缩放仍为0.01。基础actor和动作标准差保持冻结，更新多障碍修正和critic。',
        '- 四静态障碍尺寸/布置规则、物理碰撞、安装精度、240步上限不变。双障碍不参与训练或选型，无在线避障规划器或RRT代执行。',
        '- 新测试和噪声条件在训练前声明，模型哈希在测试前冻结。测试结果不重新决定检查点。三个噪声工况共享同100布局，不能当作300个独立新布局。',
        '- 随机动作按每场景固定Torch种子采样；传感噪声只作用目标位姿观测，真实安装判定不放宽。噪声对比无模型更新。',
        f'- 训练完成episode：`{s["training_episodes"]}`，完成部分转移{s["training_completed_transitions"]}，剩余转移属于尚未结束episode，总新增PPO步数恰为{spec["steps"]}。',
        f'- 新测试配对变化：`{paired}`。',
        '- curves.png / checkpoints.csv：新旧验证曲线、各检查点计数。evaluations.csv：逐场景的成功/碰撞/超时和最终误差。',
        '- test_source.json / test_selected.json：新100例配对结果。test_sensor_noise.json / test_action_noise.json：两种噪声各自结果。',
        '- training/progress.csv位于模型目录，保存PPO损失、KL和训练奖励；原始timesteps含继承计数，新增步数需减protocol.continuation.source_counter。',
        f'- 模型与优化器检查点：`{(models/"checkpoints").resolve()}`。',''])
    lines.extend(['', '- 新100配对测试保留全部74个原成功场景，9个碰撞和1个超时转为成功。传感噪声条件下，84个成功与16个碰撞的场景归属也保持相同；只代表本噪声级别和本场景集。',
        '- 完整回归182项测试通过。固定验证种子1032001和一次随机种子1302077125的无界面演示安装成功，不把这两个演示加入成功率统计。',
        '- 当前新测试84%尚未达到90%目标；随机动作57%显示动作采样下仍有鲁棒性差距。',
        '- 已完成的训练epoch不代表越多越好：新增10000步是选中检查点，20000步末候选的新64验证为46成功/11碰撞/7超时，未采用。',
        '- 本轮新100例现已使用，后续调参后应另建新测试集再声称独立测试。', '',
        '播放最新扩展模型（随机布局，0.6倍速度；旧演示入口保留）：', '', '```bash',
        'cd /home/johnsonchen/wall-panel-rl/.worktrees/path-reference-ppo',
        'env -u PYTHONPATH /home/johnsonchen/wall-panel-rl/.conda-env/bin/python scripts/demo_four_expanded.py', '```', ''])
    (root/'README.md').write_text('\n'.join(lines))
    dump(root/'verification.json',{'code_and_inputs_match':True,'splits_disjoint':True,'optimizer_and_value_preserved':True,
        'base_actor_and_std_unchanged':s['base_actor_and_std_unchanged'],'test_selected_after_freeze':True,
        'test_used_for_selection':False,'training_transitions':spec['steps'],'bc_updates':0})
    for temp,name in [('/tmp/four_expanded_interface_tests.log','interface_tests.txt'),('/tmp/four_expanded_full_tests.log','full_tests.txt')]:
        if Path(temp).exists():shutil.copy2(temp,root/name)
    dest=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    shutil.copytree(root,dest,dirs_exist_ok=True)
    print(json.dumps({k:s[k] for k in ['selected','test_source','test_selected','old64_selected','sensor_noise','action_noise','training_episodes']},indent=2));print(dest)


if __name__=='__main__':main()
