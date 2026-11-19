"""Move four boxes onto initialization's nominal motion; no learning updates."""
import _bootstrap
import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil
import json
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from run_static_expansion import project_cases
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.on_path_obstacles import nominal_rollout,place_on_nominal
from wall_panel_rl.rl.episode_imitation import evaluate_controller


def run(spec):
    data,report=Path(spec['data_root']),Path(spec['report_root']);source=Path(spec['source_report'])
    if data.exists() or report.exists():raise FileExistsError('Refusing placement study overwrite')
    parent=read(source/'manifest.json');config=copy.deepcopy(parent['configuration'])
    settings=copy.deepcopy(parent['evaluation_settings']);settings['on_path_obstacles']=copy.deepcopy(spec['on_path_obstacles'])
    if settings['active_stage']['count']!=4 or settings['active_stage']['mode']!='static':raise ValueError('Four static boxes required')
    source_hash=sha(source/'selected_model.zip');model=PPO.load(source/'selected_model.zip',device='cpu')
    data.mkdir(parents=True);report.mkdir(parents=True);shutil.copy2(source/'selected_model.zip',report/'selected_model.zip')
    manifest=copy.deepcopy(parent);manifest.update(evaluation_settings=settings,model_sha256=source_hash,
        source_report=spec['source_report'],training_performed=False,training_steps=0,
        obstacle_placement='nominal_physical_policy_route',nominal_model_sha256=source_hash)
    dump(report/'manifest.json',manifest)
    state={'status':'running','training_steps':0,'started_at':datetime.now().astimezone().isoformat()}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('FOUR_ON_PATH',json.dumps(state),flush=True)
    status('place_on_nominal')
    try:
        full=read(spec['full_scene_bank']);old=project_cases(full,settings['active_stage'],alignments=False)
        cases=[];before=[];failed=[];env=WallPanelEnv(copy.deepcopy(config))
        try:
            for i,case in enumerate(old):
                try:
                    nominal=nominal_rollout(model,config,settings,case['scene'])
                    moved=place_on_nominal(env,case['scene'],nominal,spec['on_path_obstacles'])
                except RuntimeError as exc:
                    failed.append({'seed':case['scene']['seed'],'reason':str(exc)});continue
                changed=copy.deepcopy(case);changed['scene']=moved;cases.append(changed);before.append(case)
                if (i+1)%10==0:status('place_on_nominal',layouts=i+1,accepted=len(cases),placement_failures=len(failed))
        finally:env.close()
        if not cases:raise RuntimeError('No valid four-box on-path placements')
        dump(data/'cases.json',cases);dump(data/'old_positions_paired.json',before);dump(data/'placement_failures.json',failed)
        status('evaluate_old_positions',valid_cases=len(cases))
        old_result=evaluate_controller(model,config,parent['evaluation_settings'],before,mode='ppo')
        status('evaluate_on_path')
        new_result=evaluate_controller(model,config,settings,cases,mode='ppo')
        for name,result in [('old_positions',old_result),('on_path',new_result)]:
            result.update(independent_test=False,training_steps=0,placement_change_only=True)
            dump(report/f'{name}.json',result)
        if sha(source/'selected_model.zip')!=source_hash or sha(report/'selected_model.zip')!=source_hash:
            raise AssertionError('Initialized weights changed')
        dump(report/'summary.json',{'source_model_sha256':source_hash,'weights_unchanged':True,'training_steps':0,
            'attempted_scenes':len(old),'valid_scenes':len(cases),'placement_failures':failed,
            'old_positions':old_result['metrics'],'on_path':new_result['metrics'],
            'all_four_intersect_nominal':all(c['scene']['on_path_placement']['all_obstacles_intersect_nominal'] for c in cases),
            'start_goal_clear':all(c['scene']['on_path_placement']['start_goal_valid'] for c in cases),
            'route_kind':'initialization_obstacle_free_physical_rollout'})
        status('complete',status='completed',on_path=new_result['metrics']['stage_target'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/four_static_on_path.yaml')
    p.add_argument('--run',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    spec=yaml.safe_load(Path(args.config).read_text())
    if spec['training_steps']!=0:raise ValueError('Placement only')
    print({'mode':'run' if args.run else 'check_only','count':4,'route':'physical nominal policy trajectory'},flush=True)
    if args.run:run(spec)


if __name__=='__main__':main()
