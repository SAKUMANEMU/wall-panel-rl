"""Package the retained initialization for four static boxes and evaluate it.

--run creates an immutable profile and paired count-only evaluation. It never
calls learn(), runs BC, or changes any source checkpoint. The bank reuses seen
v4 scenes as a count-change regression, not a fresh independent policy test.
"""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read, dump, sha
from run_static_expansion import project_cases, minimum_box_gap
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.episode_imitation import evaluate_controller


def four_settings(settings):
    result=copy.deepcopy(settings)
    if result['active_stage']['mode']!='static': raise ValueError('Expected the retained static profile')
    result['active_stage'].update(name='four_static',count=4,mode='static')
    return result


def run(study):
    data, report = Path(study['data_root']), Path(study['report_root'])
    if data.exists() or report.exists(): raise FileExistsError('Refusing profile overwrite')
    parent=read(Path(study['source_report'])/'candidate_manifest.json')
    if read(Path(study['source_report'])/'assessment.json')['selected_bc_update']!=0:
        raise ValueError('Source is no longer the retained initialization')
    source_hash=sha(study['initial_model']); model=PPO.load(study['initial_model'],device='cpu')
    selected=PPO.load(Path(study['source_report'])/'candidate_model.zip',device='cpu')
    if not all(torch.equal(value,selected.policy.state_dict()[name]) for name,value in model.policy.state_dict().items()):
        raise ValueError('Initialized weights do not match the retained candidate')
    if model.observation_space.shape!=(77,) or model.action_space.shape!=(7,): raise ValueError('Unexpected model interface')
    config=copy.deepcopy(parent['configuration']); two=copy.deepcopy(parent['evaluation_settings']); four=four_settings(two)
    full=read(study['full_scene_bank']); two_cases=project_cases(full,two['active_stage']); four_cases=project_cases(full,four['active_stage'])
    data.mkdir(parents=True); report.mkdir(parents=True)
    dump(data/'four_cases.json',four_cases); dump(data/'two_cases.json',two_cases)
    shutil.copy2(study['initial_model'],report/'selected_model.zip')
    dump(report/'manifest.json',{'configuration':config,'evaluation_settings':four,'model_sha256':source_hash,
        'source_model':str(Path(study['initial_model']).resolve()),'source_report':study['source_report'],
        'training_performed':False,'training_steps':0,'obstacle_original_indices':[2,3,1,4],
        'demo_defaults':{'random_fresh_scene':True,'playback_speed':.6,'controller':'ppo'}})
    state={'status':'running','phase':'placement_audit','training_performed':False,'started_at':datetime.now().astimezone().isoformat()}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('FOUR_INITIAL',json.dumps(state),flush=True)
    try:
        audit=[];env=WallPanelEnv(copy.deepcopy(config))
        try:
            for case in four_cases:
                if case['suite']!='stage_target': continue
                scene=case['scene'];env.reset(seed=scene['seed'],options={'scenario':scene})
                audit.append({'seed':scene['seed'],'obstacles':len(env.obstacle_ids),'static':not env.obstacle_motion,
                    'start_valid':env.collision.is_configuration_valid(scene['initial_q']),
                    'goal_valid':env.collision.is_configuration_valid(scene['reference_joint_path'][-1]),
                    'minimum_box_gap_m':minimum_box_gap(scene)})
        finally:env.close()
        dump(report/'placement_audit.json',audit)
        if any(a['obstacles']!=4 or not a['static'] or not a['start_valid'] or not a['goal_valid'] for a in audit):
            raise RuntimeError('Four-box projection contains an invalid start/goal; see placement audit')
        status('paired_two_static')
        before=evaluate_controller(model,config,two,two_cases,mode='ppo')
        status('paired_four_static')
        after=evaluate_controller(model,config,four,four_cases,mode='ppo')
        for name,result in [('two_static',before),('four_static',after)]:
            result.update(independent_test=False,seen_underlying_scene_bank=True,training_performed=False)
            dump(report/f'{name}.json',result)
        params=dict(model.policy.named_parameters())
        ignored=all(torch.count_nonzero(params[name][:,30:70]).item()==0 for name in ('residual.0.weight','prior_gate.0.weight'))
        ignored=ignored and torch.count_nonzero(model.policy.multi_adapter[-1].weight).item()==0 and torch.count_nonzero(model.policy.multi_adapter[-1].bias).item()==0
        if sha(study['initial_model'])!=source_hash or sha(report/'selected_model.zip')!=source_hash:
            raise AssertionError('Original initialized checkpoint changed')
        dump(report/'summary.json',{'two_static':before['metrics'],'four_static':after['metrics'],
            'source_sha256':source_hash,'model_unchanged':True,'training_steps':0,
            'new_obstacle_inputs_ignored_by_initialization':ignored,'active_count':4,'mode':'static',
            'paired_scene_count':len(full),'original_indices':[2,3,1,4]})
        status('complete',status='completed',four_static=after['metrics']['stage_target'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_static_initial.yaml')
    p.add_argument('--run',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text())
    if study['obstacle_count']!=4 or study['mode']!='static' or study['training_steps']!=0: raise ValueError('Four static, no training')
    print({'mode':'run' if args.run else 'check_only','initial_model':study['initial_model'],'count':4},flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
