"""Explicit bounded BC budget study, followed by a gated 10k PPO pilot.

First reuse fixed data with the original source/split and 10k BC updates. If BC
release fails, generate 256 layouts / 1,000 successful demonstrations including
200 corrective episodes, and run one more 10k BC budget. No harder stage or
additional budget is launched. Default checks only, --run executes the study.
"""
import _bootstrap
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime
import torch
import yaml
from stable_baselines3 import PPO
import staged_avoidance as stage
from wall_panel_rl.rl.avoidance_curriculum import initialize_gated
from wall_panel_rl.rl.avoidance_imitation import collect_teacher,pretrain_actor,teacher_reuse_contract
from wall_panel_rl.utils.config import load_config


def settings_for(study,name):
    s=stage.resolve(yaml.safe_load(Path(study['base_config']).read_text()),study['stage'])
    if s['stage_index']!=0:raise ValueError('Budget study is restricted to the first stage')
    s['output_root']=study['output_root']
    s['stage_output']=str(Path(study['output_root'])/name)
    s['output']=str(Path(s['stage_output'])/'ppo')
    s['teacher']['updates']=study['bc_updates']
    s['teacher']['closed_loop_updates']=study['closed_loop_updates']
    if name=='same_data_10k':s['data']=study['existing_data']
    elif name=='expanded_10k':
        s['data']=str(Path(study['data_root'])/'expanded')
        s['data_root']=study['data_root'];s['train_scenes']=study['expanded']['train_scenes']
        for key,value in study['expanded'].items():
            if key!='train_scenes':s['teacher'][key]=value
        s['teacher']['minimum_unique_scenes']=study['expanded']['minimum_unique_avoidance_scenes']
        s['teacher']['correction_model']=str(Path(study['output_root'])/'same_data_10k/bc/model.zip')
    else:raise ValueError(name)
    stage.check(s)
    return s


def environment(s):
    c=load_config(s['environment'])
    if c['noise']!={'position_std':0.,'orientation_std_deg':0.}:raise ValueError('Noise-free study required')
    c['obstacles']['freeze_motion']=False
    c['obstacles']['dynamic'].update(enabled=True,placement='rrt_path',motion_mode='random_3d',amplitude=.4,max_speed=.4)
    return c


def check_study(study):
    steps=study['closed_loop_updates']
    if (type(study['bc_updates']) is not int or study['bc_updates']<=0 or
        sorted(set(steps))!=steps or not steps or steps[-1]!=study['bc_updates'] or
        any(type(v) is not int or v<=0 or v%100 for v in steps)):
        raise ValueError('Invalid bounded BC budget/checkpoints')
    expanded=study['expanded']
    if not 0<expanded['correction_episodes']<=expanded['avoidance_episodes']:
        raise ValueError('Corrections must be included in avoidance count')
    if not 1<=expanded['minimum_unique_avoidance_scenes']<=min(expanded['train_scenes'],expanded['avoidance_episodes']):
        raise ValueError('Invalid expanded scene coverage')
    if (not 1<=expanded['correction_prefix_min']<=expanded['correction_prefix_max'] or
        expanded['correction_handoff_clearance']<=0):raise ValueError('Invalid correction handoff')
    if expanded['train_scenes']>1000:raise ValueError('Preparation has a 1000-seed attempt range')
    for name in ('same_data_10k','expanded_10k'):settings_for(study,name)


def check_bank(config,s):
    data=Path(s['data']);meta=json.loads((data/'stage_metadata.json').read_text())
    if (meta['environment_config']!=config or meta['stage']!=s['active_stage'] or
        meta.get('interface_contract')!=s['interface_contract'] or
        meta.get('reference_recovery')!=s['reference_recovery']):
        raise ValueError('Scene bank contract changed')
    if any(hashlib.sha256((data/p).read_bytes()).hexdigest()!=h for p,h in meta['data_hashes'].items()):
        raise ValueError('Scene bank hash changed')


def run(study):
    root=Path(study['output_root']);report=Path(study['report_root'])
    if root.exists() or report.exists():raise FileExistsError('Study outputs already exist; refusing overwrite/restart')
    s=settings_for(study,'same_data_10k');config=environment(s);check_bank(config,s)
    teacher_path=Path(s['data'])/'teacher'
    teacher=json.loads((teacher_path/'metadata.json').read_text())
    binding=teacher_reuse_contract(teacher,s,config)
    if hashlib.sha256((teacher_path/'teacher.npz').read_bytes()).hexdigest()!=binding['teacher_sha256']:
        raise ValueError('Existing teacher arrays changed')
    root.mkdir(parents=True);report.mkdir(parents=True)
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),
        'runner_pid':os.getpid(),
        'completed_phases':[],'automatic_stage_advance':False,'ppo_max_steps':s['additional_timesteps'],
        'bc_budget_per_run':study['bc_updates'],'expanded_successful_episodes':
            study['expanded']['alignment_episodes']+study['expanded']['avoidance_episodes']}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        stage.dump(report/'status.json',state);print('STUDY',json.dumps(state),flush=True)
    manifest={'study':study,'same_data_binding':binding,'configuration':config,
        'source_sha256':binding['source_sha256'],'same_initialization_not_optimizer_resume':True,
        'code_hashes':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            ['scripts/run_bc_budget.py','scripts/staged_avoidance.py',
             'src/wall_panel_rl/rl/avoidance_imitation.py','src/wall_panel_rl/rl/avoidance_curriculum.py',
             'src/wall_panel_rl/rl/autonomous_dynamic_ppo.py','src/wall_panel_rl/baselines/classical.py']}}
    stage.dump(report/'manifest.json',manifest)
    (report/'study.yaml').write_text(yaml.safe_dump(study))
    try:
        chosen=None
        for name in ('same_data_10k','expanded_10k'):
            s=settings_for(study,name);data=Path(s['data']);bc_dir=Path(s['stage_output'])/'bc'
            if name=='expanded_10k':
                status('expanded_prepare');stage.prepare_stage(config,s);check_bank(config,s)
                status('expanded_teacher')
                collect_teacher(config,json.loads((data/'train.json').read_text()),s,data/'teacher')
            source=PPO.load(s['source_model'],device='cpu')
            cases=json.loads((data/'validation_staged.json').read_text())
            status(name+'_bc')
            def evaluator(candidate,update,losses):
                result,rows=stage.score_bc_model(candidate,source,config,s,cases)
                stage.dump(bc_dir/'evaluations'/f'update_{update}.json',result)
                stage.dump(bc_dir/'evaluations'/f'episodes_{update}.json',rows)
                status(name+'_bc',bc_update=update,bc_target_success=result['metrics']['dynamic_ppo']['stage_target']['success'])
                return result
            started=time.time()
            pretrain_actor(initialize_gated(s['source_model'],s),data/'teacher',s,bc_dir,
                           reuse_environment=config,evaluator=evaluator)
            status(name+'_validate')
            gate=stage.validate_bc(config,s,data,bc_dir)
            stage.dump(report/(name+'.json'),{'settings':s,'bc_elapsed_seconds':time.time()-started,
                'bc_gate':gate,'metadata':json.loads((bc_dir/'metadata.json').read_text())})
            state['completed_phases'].append(name)
            if gate['passed']:chosen=copy.deepcopy(s);break
        if chosen is not None:
            status('ppo',bc_variant=Path(chosen['stage_output']).name,bc_gate_passed=True)
            stage.train(config,chosen,Path(chosen['data']),Path(chosen['stage_output'])/'bc',Path(chosen['output']))
            assessment=json.loads((Path(chosen['output'])/'assessment.json').read_text())
            stage.dump(report/'ppo_assessment.json',assessment)
            state['completed_phases'].append('ppo')
            status('complete',status='completed',stage_gate_passed=assessment['passed'],
                   finished_at=datetime.now().astimezone().isoformat())
        else:
            status('complete',status='stopped_bc_gate',bc_gate_passed=False,
                   finished_at=datetime.now().astimezone().isoformat(),
                   reason='Both bounded BC variants failed release; PPO not launched')
    except Exception as exc:
        status(state['phase'],status='failed',error=repr(exc),finished_at=datetime.now().astimezone().isoformat())
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/bc_budget_study.yaml')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    study=yaml.safe_load(Path(args.config).read_text());check_study(study)
    print(json.dumps({'mode':'run' if args.run else 'check_only','study':study},indent=2),flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
