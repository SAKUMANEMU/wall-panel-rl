"""Freeze 100 new paired scenes, verify one-static, then a bounded three-static pilot.

No dynamic stage or extra budget is launched. The final one/three test banks are
never passed to training or checkpoint selection. Default validates inputs only.
"""
import _bootstrap
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from train_episode_rrt_ppo import train
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.utils.metrics import capture_scenario
from wall_panel_rl.rl.avoidance_curriculum import stage_scene,gate_result,assess_history
from wall_panel_rl.rl.autonomous_dynamic_ppo import without_obstacles
from wall_panel_rl.rl.episode_imitation import evaluate_controller
from wall_panel_rl.rl.episode_policy import safety_first_score


def collect_seeds(value):
    """Collect original scene IDs without confusing a case index with a seed."""
    result=set();stack=[value]
    while stack:
        item=stack.pop()
        if isinstance(item,dict):
            if isinstance(item.get('seed'),int):result.add(item['seed'])
            stack.extend(item.values())
        elif isinstance(item,list):stack.extend(item)
    return result


def known_scene_seeds(data_root=Path('data')):
    seeds=set()
    for path in data_root.rglob('*.json'):
        if path.name=='train.json' or any(term in path.name for term in ('validation','cases','fresh','test')):
            seeds.update(collect_seeds(read(path)))
    return seeds


def static_stage(settings,count):
    stage=copy.deepcopy(settings['active_stage'])
    if stage['mode']!='static' or count not in (1,3):raise ValueError('One/three static scenes only')
    stage.update(name='one_static' if count==1 else 'three_static',count=count)
    return stage


def project_cases(full,stage,alignments=True):
    cases=[{'suite':'stage_target','task':'static','strength':[stage['amplitude'],stage['speed']],
            'scene':stage_scene(scene,stage)} for scene in full]
    if alignments:
        if len(full)<20:raise ValueError('Need twenty independent alignment cases')
        for i,case in enumerate(cases[:20]):
            near=i>=10
            cases.append({'suite':'alignment_near' if near else 'alignment_far','task':'alignment',
                'strength':case['strength'],'scene':without_obstacles(case['scene'],.1 if near else None)})
    return cases


def minimum_box_gap(scene):
    boxes=scene['obstacles'];distances=[]
    for i,a in enumerate(boxes):
        for b in boxes[i+1:]:
            gap=np.maximum(np.abs(np.asarray(a['center'])-b['center'])-
                (np.asarray(a['dimensions'])+b['dimensions'])/2,0)
            distances.append(float(np.linalg.norm(gap)))
    return min(distances) if distances else float('inf')


def stage_bundle(parent):
    """Change stage metadata, never physical settings or learned weights."""
    bundle=copy.deepcopy(parent)
    for key in ('training_settings','evaluation_settings'):
        bundle[key]['active_stage']=static_stage(bundle[key],3)
    bundle.update(stage_preparation_only=True,stage_change='one_static -> three_static',
        auxiliary_optimizer='fresh per-stage Adam; PPO weights/optimizer are continued',
        test_cases_used_for_training=False)
    return bundle


def score(model,config,settings,cases):
    return evaluate_controller(model,config,settings,cases,mode='ppo')


def ranked(result,settings,goal_ok=True):
    metrics=result['metrics']
    return safety_first_score({'metrics':{'dynamic_ppo':metrics},'checks':{
        'goal_convergence':goal_ok,'alignment_collision_free':all(metrics[k]['collision']==0 and metrics[k]['joint_limit']==0
            for k in ('alignment_far','alignment_near'))}},settings['gate'])


def check(study):
    parent=read(Path(study['source_report'])/'manifest.json')
    if sha(study['initial_model'])!=parent['model_sha256']:raise ValueError('Wrong selected source checkpoint')
    if parent['evaluation_settings']['active_stage']['count']!=1:raise ValueError('Expected one-static source')
    if study['fresh_targets']!=100 or study['ppo_steps']!=10000 or study['ppo_eval_every']!=5000:
        raise ValueError('One bounded 100-case verification plus 10k pilot required')
    if parent['configuration']['noise']!={'position_std':0.,'orientation_std_deg':0.}:raise ValueError('Keep clean sensing')
    return parent


def freeze_fresh(study,config,settings,forbidden,status):
    env=WallPanelEnv(copy.deepcopy(config));audit=WallPanelEnv(copy.deepcopy(config))
    full=[];failures=[];one=static_stage(settings,1);three=static_stage(settings,3)
    try:
        for seed in range(study['fresh_seed_start'],study['fresh_seed_start']+study['maximum_generation_attempts']):
            if seed in forbidden:raise ValueError(f'Fresh scene seed was already used: {seed}')
            try:
                env.reset(seed=seed);scene=capture_scenario(env,seed)
                # Geometric placement validity only, not planner/policy success.
                for stage in (one,three):
                    projected=stage_scene(scene,stage)
                    if minimum_box_gap(projected)<config['obstacles']['min_obstacle_gap']:
                        raise RuntimeError('Static projection violates minimum box spacing')
                    audit.reset(seed=seed,options={'scenario':projected})
                    if not audit.collision.is_configuration_valid(projected['initial_q']):
                        raise RuntimeError('Static projection overlaps the start')
                    if not audit.collision.is_configuration_valid(projected['reference_joint_path'][-1]):
                        raise RuntimeError('Static projection overlaps the goal')
                full.append(scene)
            except RuntimeError as exc:
                failures.append({'seed':seed,'reason':str(exc)});continue
            if len(full)%10==0:status('freeze_fresh_scenes',accepted=len(full),rejected=len(failures))
            if len(full)==study['fresh_targets']:break
    finally:env.close();audit.close()
    if len(full)!=study['fresh_targets']:raise RuntimeError('Bounded scene-generation attempts exhausted')
    return full,failures


def run(study):
    parent=check(study);data=Path(study['data_root']);report=Path(study['report_root'])
    protected=[data,report,Path(study['three_output_root']),Path(study['three_report_root']),Path(study['three_training_config'])]
    if any(p.exists() for p in protected):raise FileExistsError('Refusing repeated experiment/overwrite')
    # Audit existing scene IDs BEFORE writing the new data.
    forbidden=known_scene_seeds();data.mkdir(parents=True);report.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),
        'fresh_targets':100,'three_ppo_budget':10000,'automatic_dynamic_stage':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('STATIC_EXPANSION',json.dumps(state),flush=True)
    config=copy.deepcopy(parent['configuration']);one=copy.deepcopy(parent['evaluation_settings'])
    source_hash=sha(study['initial_model'])
    dump(report/'protocol.json',{'study':study,'source_sha256':source_hash,
        'training_and_selection_do_not_receive_fresh_banks':True,
        'one_fresh_gate':one['gate'],'auxiliary_optimizer_fresh_for_three_stage':True,
        'placement_rejections_reported_without_policy_or_expert_filtering':True})
    try:
        status('prepare')
        full,failures=freeze_fresh(study,config,one,forbidden,status)
        one_cases=project_cases(full,static_stage(one,1));three_cases=project_cases(full,static_stage(one,3))
        dump(data/'fresh_full.json',full);dump(data/'one_fresh.json',one_cases);dump(data/'three_fresh.json',three_cases)
        dump(data/'generation_failures.json',failures)
        validation_full=read(study['raw_validation'])
        validation=project_cases(validation_full,static_stage(one,3),alignments=False)
        validation.extend(copy.deepcopy(c) for c in read(study['alignment_validation']) if c['suite'].startswith('alignment'))
        dump(data/'three_validation.json',validation)
        bundle=stage_bundle(parent);dump(data/'three_stage_source/manifest.json',bundle)
        three=bundle['evaluation_settings']
        # Training uses the original 54 non-held-out full layouts, projected to 3.
        training=yaml.safe_load(Path(study['base_training_config']).read_text())
        training.update(source_report=str(data/'three_stage_source'),initial_model=study['initial_model'],
            validation_cases=str(data/'three_validation.json'),output_root=study['three_output_root'],
            report_root=study['three_report_root'],evaluation_output=str(report/'three_preflight_unused'),
            ppo_steps=study['ppo_steps'],ppo_eval_every=study['ppo_eval_every'])
        Path(study['three_training_config']).write_text(yaml.safe_dump(training,sort_keys=False))
        train_seeds=collect_seeds(read(training['train_scenes']))
        validation_seeds=collect_seeds(validation);fresh_seeds=collect_seeds(full)
        if fresh_seeds&(train_seeds|validation_seeds|forbidden):raise ValueError('Fresh scene leakage')
        dump(report/'frozen_banks.json',{'seeds':sorted(fresh_seeds),'train_validation_disjoint':True,
            'hashes':{str(p):sha(p) for p in (data/'fresh_full.json',data/'one_fresh.json',data/'three_fresh.json',data/'three_validation.json')},
            'configuration_unchanged':config==parent['configuration'],'generation_failures':len(failures)})
        source=PPO.load(study['initial_model'],device='cpu')
        status('one_fresh_verification')
        one_before=score(source,config,one,one_cases);one_before['independent_test']=True
        dump(report/'one_before.json',one_before)
        gate_one=gate_result(one_before['metrics'],one['gate'])
        if not gate_one['passed']:
            dump(report/'summary.json',{'one_fresh':one_before['metrics'],'one_gate':gate_one,
                'three_training_started':False,'reason':'Current one-obstacle generalization gate failed',
                'default_model_preserved':True,'automatic_dynamic_stage':False})
            status('complete',status='completed',three_training_started=False);return
        status('three_initial_verification',one_fresh=one_before['metrics']['stage_target'])
        before_val=score(source,config,three,validation);dump(report/'three_before_validation.json',before_val)
        status('three_training',three_initial=before_val['metrics']['stage_target'])
        train(training)
        three_report=Path(study['three_report_root']);trained=read(three_report/'summary.json')
        model=PPO.load(three_report/'selected_model.zip',device='cpu')
        status('three_holdout_verification')
        before_test=score(source,config,three,three_cases);dump(report/'three_holdout_before.json',before_test)
        after_test=score(model,config,three,three_cases);dump(report/'three_holdout_after.json',after_test)
        status('one_regression_verification')
        one_after=score(model,config,one,one_cases);dump(report/'one_after.json',one_after)
        history=[json.loads(line) for line in (Path(training['output_root'])/'ppo/validation.jsonl').read_text().splitlines() if line]
        stage_gate=assess_history(history,three['gate'])
        selected_gate=gate_result(trained['selection']['metrics']['dynamic_ppo'],three['gate'])
        holdout_gate=gate_result(after_test['metrics'],three['gate'])
        regression_ok=ranked(one_after,one)>=ranked(one_before,one)
        eligible=stage_gate['passed'] and selected_gate['passed'] and holdout_gate['passed'] and regression_ok
        if sha(study['initial_model'])!=source_hash:raise AssertionError('Source checkpoint modified')
        dump(report/'summary.json',{'source_sha256':source_hash,'three_training_started':True,
            'one_fresh':one_before['metrics'],'one_gate':gate_one,
            'three_validation_before':before_val['metrics'],
            'three_validation_selected':trained['selection']['metrics']['dynamic_ppo'],
            'three_selected_additional_steps':trained['selection']['additional_steps'],
            'three_holdout_before':before_test['metrics'],'three_holdout_after':after_test['metrics'],
            'one_regression_after':one_after['metrics'],'one_regression_safety_not_worse':regression_ok,
            'three_stage_gate':stage_gate,'three_selected_gate':selected_gate,'three_holdout_gate':holdout_gate,
            'three_stage_ready':eligible,'default_model_preserved':True,'automatic_dynamic_stage':False,
            'source_file_unchanged':True,'fresh_banks_used_for_selection':False})
        status('complete',status='completed',three_stage_ready=eligible)
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/static_expansion.yaml');parser.add_argument('--run',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check(study)
    print({'mode':'run' if args.run else 'check_only','study':study},flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
