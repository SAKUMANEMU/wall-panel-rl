"""Summarize the completed autonomous dynamic pilot from recorded outcomes."""
import _bootstrap
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wall_panel_rl.rl.autonomous_dynamic_ppo import AutonomousDynamicEnv
from wall_panel_rl.utils.config import load_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('results/diagnostics/dynamic_autonomous_pilot_seed61'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    root=Path('models/dynamic_autonomous_pilot_seed61')
    training=json.loads((root/'training.json').read_text())
    assert training['additional_steps']==10000
    events=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines()]
    assert [r['additional_steps'] for r in events]==[0,5000,10000]
    assert events[-1]['kind']=='final_saved_reload'
    rows=[]
    for event in events:
        for method,suites in event['metrics'].items():
            for suite,m in suites.items():rows.append({'kind':event['kind'],'additional_steps':event['additional_steps'],
                'method':method,'suite':suite,**m})
    frame=pd.DataFrame(rows);frame.to_csv(out/'validation_summary.csv',index=False)
    monitors=pd.concat([pd.read_csv(p,skiprows=1) for p in root.glob('worker_*.monitor.csv')],ignore_index=True)
    groups=monitors.groupby(['curriculum_task','termination_reason']).size()
    groups.rename('episodes').to_csv(out/'training_episode_counts.csv')
    torch.set_num_threads(1)
    baseline=PPO.load(root/'checkpoints/checkpoint_0.zip',device='cpu')
    final=PPO.load(root/'final_model.zip',device='cpu')
    changes={name:float((value-baseline.policy.state_dict()[name]).abs().max()) for name,value in final.policy.state_dict().items()}
    for name in ('tracking_gain_raw','tracking_bias','tracking_velocity','log_std'):assert changes[name]==0
    assert any(value>0 for name,value in changes.items() if name.startswith('residual.'))
    settings=json.loads((root/'plan.json').read_text())['settings']
    config=load_config(settings['environment'])
    cases=json.loads((Path(settings['data'])/'validation.json').read_text())
    env=AutonomousDynamicEnv(config,[],settings);corrections=[]
    try:
        for case in cases:
            obs,_=env.reset(options={'scenario':case['scene'],'task':case['task'],'strength':case['strength']})
            b=baseline.predict(obs,deterministic=True)[0];f=final.predict(obs,deterministic=True)[0]
            corrections.extend(np.abs(f-b).tolist())
    finally:env.close()
    summary={'training':training,'completed_training_episodes':len(monitors),
        'training_counts':[{ 'task':a,'reason':b,'episodes':int(n)} for (a,b),n in groups.items()],
        'selection':json.loads((root/'selection.json').read_text()),
        'parameter_max_absolute_changes':changes,
        'initial_validation_states_mean_absolute_action_change':float(np.mean(corrections)),
        'initial_validation_states_max_absolute_action_change':float(np.max(corrections)),
        'observed_motion_bounds':{'max_amplitude':float(monitors.amplitude.max()),'max_speed':float(monitors.max_speed.max())},
        'online_avoidance_planner':False,'independent_test_executed':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for suite in ('dynamic_easy','dynamic_full','static'):
        g=frame[(frame.method=='dynamic_ppo')&(frame.suite==suite)]
        axes[0].plot(g.additional_steps,100*g.success/g.episodes,'o-',label=suite)
        axes[1].plot(g.additional_steps,100*g.collision/g.episodes,'o-',label=suite)
    for suite in ('alignment_far','alignment_near'):
        g=frame[(frame.method=='dynamic_ppo')&(frame.suite==suite)]
        axes[2].plot(g.additional_steps,100*g.success/g.episodes,'o-',label=suite)
    for ax in axes:ax.set(xlabel='Additional dynamic PPO transitions',ylim=(-3,105));ax.legend(fontsize=8)
    axes[0].set_ylabel('Avoidance validation success (%)')
    axes[1].set_ylabel('Avoidance validation collisions (%)')
    axes[2].set_ylabel('Alignment validation success (%)')
    fig.tight_layout();fig.savefig(out/'overview.png',dpi=180);plt.close(fig)
    lines=['# 自主动态避障 PPO 短试验','',
        f"追加训练 {training['additional_steps']:,} 步；源模型累计 {training['source_steps']:,} 步；新计数 {training['total_steps']:,} 步。完成训练回合 {len(monitors)} 个。训练及三次验证耗时 {training['elapsed_seconds']:.2f} 秒，不含场景库生成。",'',
        '场景已缓存，执行中不调用 RRT，失败回合通常很短，因此本轮比逐回合重新规划的静态试验快。预算与计数已核对。','',
        '## 最终保存权重的验证结果','',
        '| 条件 | 成功 | 碰撞 | 越界 | 超时 |','|---|---:|---:|---:|---:|']
    for suite,m in events[-1]['metrics']['dynamic_ppo'].items():
        lines.append(f"| {suite} | {m['success']}/{m['episodes']} | {m['collision']} | {m['joint_limit']} | {m['timeout']} |")
    lines+=['','低/完整运动、静态自主避障在初始、5k、10k 检查点均零成功。远/近对准各 10/10，保留了对准能力。',
        '这些结果证明接口运行及动作学习已发生，但没有证明学会了动态避障。','',
        '## 训练回合分布','', '| 课程 | 终止原因 | 回合数 |','|---|---|---:|']
    for (task,reason),n in groups.items():lines.append(f'| {task} | {reason} | {n} |')
    lines+=['','20/30/50 是按回合抽样概率，实际比例和训练步数占比不保证精确相等。未结束的回合不计入完整回合数。',
        '', '## 模型选择和失败解释','',
        f"按预定规则选中 {summary['selection']['additional_steps']} 个新增动态步的检查点，即带零残差网络的源策略初始化；final_model.zip 保存实际训练后的 10k 权重。未覆盖源模型，未追加预算。",
        '冻结的跟踪系数及标准差没有变化；残差网络确实发生更新。原模型仅跟踪未考虑障碍物的参考，参考线上布置了障碍物，因此初始化避障验证也全部碰撞。',
        '此前静态 89/90 使用 RRT 避障路径；本轮 RRT 只生成障碍物放置前的参考。两者任务不同，不能将本轮零成功直接解释为原静态系统退化。',
        f"50 个验证初始状态上，最终与初始化动作的平均绝对差 {np.mean(corrections):.6f}，最大差 {np.max(corrections):.6f}（归一化动作）。",
        '新网络初始输出为零、标准差沿用 0.02（约每关节 0.08° 探索尺度），KL/clip 较保守。这些设置及短预算可能限制大幅绕行动作的发现；这里只是可能原因，尚未做因果对照。',
        '小运动幅度只代表运动强度较低，不保证几何问题更容易：同锚点改变幅度会同时改变给定相位下的实际位置。完整包络保护了起点/终点，不保证中间存在可行路线。',
        '下一步应单独设计探索强度、渐进障碍数量/布局及奖励的对照，先检查低难度成功，再决定长训练；本次未自动修改这些设定。',
        '', '## 数据与复现范围','',
        '128 个训练布局，每回合随机相位；10 个独立于训练的动态验证布局及原有 20 个对准案例，组成 50 个验证案例。每次验证三种方法，三次共 450 回合。无观测噪声、无延迟。',
        '方法为 reference_servo / source_ppo / dynamic_ppo，共享布局、相位和未考虑障碍物的参考；没有在线规划、碰撞动作拒绝或专家接管。全局参考仍由 RRT 提供，因此不是完全无规划输入的端到端策略。',
        '未执行预留独立测试集；单训练种子、每动态条件仅 10 个验证场景，不支持泛化结论。',
        '', '模型目录：`models/dynamic_autonomous_pilot_seed61/`；场景库：`data/dynamic_autonomous_pilot_seed61/`（均在工作副本内）。',
        '', '![成功、碰撞及对准曲线](overview.png)','']
    (out/'README.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('selection','parameter_max_absolute_changes')},indent=2))


if __name__=='__main__':main()
