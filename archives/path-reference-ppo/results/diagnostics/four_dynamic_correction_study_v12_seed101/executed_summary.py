"""Audit corrective BC and report its conditional PPO decision."""
import _bootstrap
from pathlib import Path
import csv,json,shutil
import torch,yaml
from wall_panel_rl.utils.config import ROOT
from run_targeted_bc import read,dump,sha


def main():
    spec=yaml.safe_load((ROOT/'configs/four_dynamic_corrections.yaml').read_text());bc=ROOT/spec['bc_report'];collection=ROOT/spec['report_root'];root=ROOT/'results/diagnostics/four_dynamic_correction_study_v12_seed101';root.mkdir(exist_ok=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
    if visible.resolve()!=root.resolve() and not visible.exists():visible.symlink_to(root,target_is_directory=True)
    s=read(bc/'summary.json');p=read(bc/'protocol.json');history=read(bc/'history.json');c=read(collection/'summary.json')
    for f,h in p['code_hashes'].items():assert sha(bc/'code_archive'/f)==h
    for f,h in p['input_hashes'].items():assert sha(ROOT/f)==h
    dataset=torch.load(ROOT/spec['output_root']/'corrections.pt',map_location='cpu',weights_only=False)
    assert all(e['row']['pre_verified'] and e['row']['post_time_verified'] and e['row']['physical_replay_verified'] and e['case']['task']=='dynamic' for e in dataset['episodes'])
    rows=[{'arm':h['arm'],'learning_rate':h['learning_rate'],'bc_updates':h['bc_updates'],**h['metrics'],'accepted_updates':h['accepted_updates'],'rejected_updates':h['rejected_updates']} for h in history]
    with (root/'checkpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.5))
    for ax,key in zip(axes,['success','collision','timeout']):
        for arm,lr in enumerate(spec['bc_learning_rates']):
            h=[x for x in history if x['arm']==arm];ax.plot([x['bc_updates'] for x in h],[100*x['metrics'][key]/x['metrics']['episodes'] for x in h],marker='o',label=str(lr))
        ax.set(title=key.capitalize()+' rate',xlabel='Attempted BC updates',ylabel='%');ax.grid(alpha=.2)
    axes[0].legend(title='Learning rate',fontsize=8);fig.suptitle('Corrective dynamic BC with successful-policy anchors');fig.tight_layout();fig.savefig(root/'curves.png',dpi=160);plt.close(fig)
    ppo_report=ROOT/spec['ppo_report'];performed=ppo_report.exists() and (ppo_report/'summary.json').exists()
    summary={'source_validation':s['source_validation'],'selected_bc':s['selected'],'bc_improved':s['validation_improved'],'ppo_started':performed,
        'ppo_skipped_reason':None if performed else 'Corrective BC did not improve validation; original source retained',
        'correction_data':c,'independent_test_evaluated_during_bc':False,'source_unchanged':s['source_unchanged'],'archived_code_and_inputs_verified':True,'static_training_episodes':0}
    if performed:summary['ppo']=read(ppo_report/'summary.json')
    dump(root/'summary.json',summary);dump(root/'status.json',{'status':'completed','phase':'complete','summary':summary})
    text='# Dynamic corrective BC study\n\nThe fixed validation target remains four moving staggered obstacles,18cm offset and0.12m/s speed. Source model, scene bank, reward and success criteria are unchanged.\n\n'
    text+=f"Data: {c['verified_corrections']} verified corrective episodes from {c['corrected_layouts']} training layouts; {c['head_training_pairs']} first16-step labels, {c['success_anchor_states']} successful PPO states. Prefixes replay from the original reset, preserving nonzero time, velocity and reference state.\n\n"
    text+='| Arm LR | Updates | Success | Collision | Timeout |\n|---|---:|---:|---:|---:|\n'
    for h in history:text+=f"| {h['learning_rate']} | {h['bc_updates']} | {h['metrics']['success']} | {h['metrics']['collision']} | {h['metrics']['timeout']} |\n"
    text+=f"\nSelected checkpoint: {s['selected']}. PPO started: {performed}. The predeclared gate prevents PPO after unsuccessful BC. No independent test is evaluated during BC; all labels come from training cases. Source-policy retention targets also use only successful training trajectories.\n"
    (root/'README.md').write_text(text);shutil.copy2(__file__,root/'executed_summary.py');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
