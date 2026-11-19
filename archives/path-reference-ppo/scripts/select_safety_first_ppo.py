"""Adopt an existing collision-first PPO candidate without retraining/history edits."""
import _bootstrap
import argparse
import copy
import json
from pathlib import Path
import shutil
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from train_conditional_ppo import ConditionalValidation,score_model
from wall_panel_rl.rl.episode_policy import safety_first_score


class SafetyFirstValidation(ConditionalValidation):
    """Opt-in callback for future training; archived callbacks remain unchanged."""
    def candidate_score(self,result):return safety_first_score(result,self.s['gate'])


def choice(study):
    source=Path(study['source_report']);manifest=read(source/'manifest.json')
    root=Path(manifest['study']['output_root'])/'ppo'
    events=[json.loads(line) for line in (root/'validation.jsonl').read_text().splitlines() if line.strip()]
    if len({e['metrics']['dynamic_ppo']['stage_target']['episodes'] for e in events})!=1:
        raise ValueError('Checkpoint comparisons require equal validation sample counts')
    selected=max(events,key=lambda e:safety_first_score(e['gate'],study['thresholds']))
    if not safety_first_score(selected['gate'],study['thresholds'])[0]:
        raise ValueError('No checkpoint preserves minimum success/alignment/goal criteria')
    steps=selected['additional_steps']
    path=(source/'selected_model.zip' if steps==0 else source/'final_candidate.zip'
          if steps==10000 else root/'checkpoints'/f'ppo_{steps}.zip')
    if not path.is_file():raise FileNotFoundError(path)
    return selected,path,events,manifest


def apply(study):
    selected,path,events,parent=choice(study);report=Path(study['output_report'])
    if report.exists():raise FileExistsError('Refusing to overwrite a selection report')
    cases=[c for c in read(study['validation_cases']) if c['suite']!='five_full_diagnostic']
    model=PPO.load(path,device='cpu')
    gate,rows=score_model(model,None,parent['configuration'],parent['evaluation_settings'],cases)
    if gate['metrics']!=selected['metrics']:
        raise RuntimeError('Candidate no longer reproduces the archived validation results')
    if not safety_first_score(gate,study['thresholds'])[0]:raise RuntimeError('Candidate failed safety-first eligibility')
    report.mkdir(parents=True)
    shutil.copy2(path,report/'selected_model.zip')
    manifest=copy.deepcopy(parent)
    manifest.update(selection_rule='minimum_ability_then_collision_rate_then_success_rate',
        selection_thresholds=study['thresholds'],source_report=str(Path(study['source_report'])),
        selected_additional_steps=selected['additional_steps'],model_sha256=sha(path),
        training_performed=False,independent_test=False)
    dump(report/'manifest.json',manifest)
    controller=read(Path(study['source_report'])/'controller_config.json')
    controller.update(model_sha256=sha(path),selection_rule=manifest['selection_rule'])
    dump(report/'controller_config.json',controller)
    dump(report/'selection.json',{'additional_steps':selected['additional_steps'],
        'score':safety_first_score(gate,study['thresholds']),'metrics':gate['metrics'],
        'previous_selection_preserved':True,'source_checkpoint':str(path),
        'model_sha256':sha(path),'training_performed':False})
    dump(report/'validation.json',{'gate':gate,'rows':rows,'independent_test':False})
    dump(report/'checkpoint_comparison.json',[{'additional_steps':e['additional_steps'],
        'metrics':e['metrics']['dynamic_ppo']['stage_target'],
        'score':safety_first_score(e['gate'],study['thresholds'])} for e in events])
    (report/'README.md').write_text(
        '# 碰撞优先的 PPO 选模（2026-11-02）\n\n'
        '按教授建议，在成功率至少80%、碰撞率至多10%、保留对准和终点收敛能力的候选中，'
        '优先减少碰撞，再比较成功率和超时。\n\n'
        '重新采用既有追加10,000步候选：17/20成功、0碰撞、3超时；对准20/20，关节越限0。'
        '原选择是18/20成功、2碰撞、0超时。本次没有重新训练；原模型与原选模报告完整保留。\n\n'
        '本轮重放的是已见验证场景；0/20碰撞不保证其他场景零碰撞。原保留模型的282/300回归结果'
        '不能归给此模型，此候选尚未进行该300场景评估。\n\n'
        '`selected_model.zip`配合`manifest.json`或`controller_config.json`使用条件恢复环境。\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/safety_first_ppo.yaml')
    parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);study=yaml.safe_load(Path(args.config).read_text())
    selected,path,_,_=choice(study)
    print({'mode':'apply' if args.apply else 'check_only','selected_additional_steps':selected['additional_steps'],
           'model':str(path),'metrics':selected['metrics']['dynamic_ppo']['stage_target']},flush=True)
    if args.apply:print('Exported:',apply(study),flush=True)


if __name__=='__main__':main()
