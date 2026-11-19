"""Correct inactive-slot initialization, protect old actor, then BC and gated PPO."""
import _bootstrap
import argparse
import copy
from datetime import datetime
from pathlib import Path
import shutil
import torch
import yaml
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from run_static_expansion import known_scene_seeds,freeze_fresh,project_cases
from run_two_static_warmup import stage,validation
from train_two_static_ppo import train as train_ppo
from wall_panel_rl.rl.added_slot_adapter import configure_adapter,original_actor_preserved
from wall_panel_rl.rl.multi_obstacle_warmup import warmup_actor,teacher_arrays
from wall_panel_rl.rl.episode_imitation import evaluate_controller
from wall_panel_rl.rl.avoidance_curriculum import gate_result


def resolve(path):
    override=yaml.safe_load(Path(path).read_text())
    study=yaml.safe_load(Path(override['base_config']).read_text());study.update(override)
    study['bc']=copy.deepcopy(study['bc'])
    study['bc'].update(updates=override['bc_updates'],checkpoints=override['bc_checkpoints'])
    return study


def run(study):
    data=Path(study['data_root']);models=Path(study['model_root']);report=Path(study['report_root'])
    if any(p.exists() for p in (data,models,report)):raise FileExistsError('Refusing corrective study overwrite')
    parent=read(Path(study['source_report'])/'manifest.json');source_hash=sha(study['initial_model'])
    if source_hash!=parent['model_sha256']:raise ValueError('Wrong original source')
    teacher=Path(study['teacher_root']);meta=read(teacher/'metadata.json')
    if not meta['completed'] or sha(teacher/'teacher.npz')!=meta['teacher_sha256'] or meta['source_sha256']!=source_hash:
        raise ValueError('Reused teacher data/source changed')
    held=set(read(study['heldout_metadata'])['heldout_scene_seeds']);teacher_arrays(teacher/'teacher.npz',held)
    forbidden=known_scene_seeds();data.mkdir(parents=True);models.mkdir(parents=True);report.mkdir(parents=True)
    (data/'resolved_study.yaml').write_text(yaml.safe_dump(study,sort_keys=False))
    state={'status':'running','started_at':datetime.now().astimezone().isoformat(),'protected_added_slots':True,
           'automatic_dynamic_stage':False}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra)
        dump(report/'status.json',state);print('PROTECTED_TWO',state,flush=True)
    code=['configs/protected_two_static.yaml','scripts/run_protected_two_static.py','scripts/train_two_static_ppo.py',
          'src/wall_panel_rl/rl/added_slot_adapter.py','src/wall_panel_rl/rl/multi_obstacle_warmup.py',
          'src/wall_panel_rl/rl/avoidance_curriculum.py','src/wall_panel_rl/rl/episode_imitation.py']
    dump(report/'code_snapshot.json',{'code_hashes':{p:sha(p) for p in code}})
    dump(report/'protocol.json',{'study':study,'source_sha256':source_hash,
        'correction':'zero inactive actor input columns30:70; freeze all original actor connections',
        'trainable_actor_parameters':'only residual.0.weight and prior_gate.0.weight columns30:70',
        'ppo_critic_trainable':True,'teacher_reused_without_test_selection':str(teacher),
        'first_attempt_preserved':'results/diagnostics/two_static_warmup_v1_seed61',
        'new_test_seed_start':study['fresh_seed_start'],'default_model_preserved':True})
    config=copy.deepcopy(parent['configuration']);one=stage(parent['evaluation_settings'],1);two=stage(one,2)
    try:
        status('freeze_new_test')
        fresh,failures=freeze_fresh(study,config,one,forbidden,status)
        dump(data/'fresh_full.json',fresh);dump(data/'generation_failures.json',failures)
        test=project_cases(fresh,two['active_stage']);dump(data/'two_test.json',test)
        for name in ('one_validation.json','two_validation.json'):
            shutil.copy2(Path(study['validation_root'])/name,data/name)
        one_cases=read(data/'one_validation.json');two_cases=read(data/'two_validation.json')
        dump(report/'data_hashes.json',{str(p):sha(p) for p in (data/'fresh_full.json',data/'two_test.json',
            data/'one_validation.json',data/'two_validation.json')})
        shutil.copy2(study['teacher_replay'],report/'teacher_replay.json')
        source=PPO.load(study['initial_model'],device='cpu')
        cold,_,_=validation(source,config,two,one,two_cases,one_cases);dump(report/'baseline.json',cold)
        initialized=PPO.load(study['initial_model'],device='cpu')
        configure_adapter(initialized.policy,initialize=True,train_critic=False)
        if not original_actor_preserved(source.policy,initialized.policy):raise AssertionError('Initialization changed old actor')
        initialized.save(models/'initialized_model.zip')
        start,_,_=validation(initialized,config,two,one,two_cases,one_cases)
        dump(report/'initialization_validation.json',start)
        status('protected_bc',two_initial=start['two']['stage_target'],one_initial=start['one']['stage_target'])
        def evaluator(candidate,update):
            if not original_actor_preserved(source.policy,candidate.policy):raise AssertionError('BC changed original actor')
            result,rows,one_rows=validation(candidate,config,two,one,two_cases,one_cases)
            dump(report/'bc_validation'/f'update_{update}.json',{'summary':result,'two_rows':rows,'one_rows':one_rows})
            status('protected_bc',bc_updates=update,two_validation=result['two']['stage_target'],
                one_validation=result['one']['stage_target'],ppo_ready=result['ppo_ready'])
            return result
        warm,metadata=warmup_actor(initialized,teacher/'teacher.npz',held,dict(study['bc'],seed=study['seed']),models/'bc',evaluator)
        selected=metadata['selected'];chosen_path=warm;ppo_started=False
        if selected['ppo_ready']:
            bundle=copy.deepcopy(parent)
            for key in ('training_settings','evaluation_settings'):bundle[key]=stage(bundle[key],2)
            bundle.update(model_sha256=sha(warm),warmup_metrics=selected,protected_added_slots=True)
            dump(data/'warm_source/manifest.json',bundle)
            ppo_study={'source_report':str(data/'warm_source'),'initial_model':str(warm),
                'train_scenes':study['raw_train'],'heldout_scenes':study['heldout_metadata'],
                'validation_cases':str(data/'two_validation.json'),'output_root':str(models/'ppo_run'),
                'report_root':str(report/'ppo_run'),'ppo_steps':study['ppo_steps'],'ppo_eval_every':study['ppo_eval_every'],
                'ppo_learning_rate':study['ppo_learning_rate'],'expert_probability':study['expert_probability'],
                'seed':study['seed'],'auxiliary_bc':study['auxiliary_bc'],'protected_added_slots':True}
            (data/'ppo_study.yaml').write_text(yaml.safe_dump(ppo_study,sort_keys=False))
            status('protected_ppo',selected_bc_update=selected['update'])
            chosen_path,_=train_ppo(ppo_study,one,one_cases,teacher/'teacher.npz',held,status)
            ppo_started=True
        else:status('bc_gate_failed')
        chosen=PPO.load(chosen_path,device='cpu')
        if not original_actor_preserved(source.policy,chosen.policy):raise AssertionError('PPO changed original actor')
        status('new_holdout_test',ppo_started=ppo_started)
        before=evaluate_controller(source,config,two,test,mode='ppo')
        init_test=evaluate_controller(PPO.load(models/'initialized_model.zip',device='cpu'),config,two,test,mode='ppo')
        after=evaluate_controller(chosen,config,two,test,mode='ppo')
        one_test=read(study['one_regression'])
        old=evaluate_controller(source,config,one,one_test,mode='ppo')
        regression=evaluate_controller(chosen,config,one,one_test,mode='ppo')
        if old['metrics']!=regression['metrics']:raise AssertionError('One-static protected regression changed')
        for name,result in [('two_test_before',before),('two_test_initialized',init_test),('two_test_after',after),('one_before',old),('one_after',regression)]:
            dump(report/f'{name}.json',result)
        final,_,_=validation(chosen,config,two,one,two_cases,one_cases)
        shutil.copy2(chosen_path,report/'candidate_model.zip')
        dump(report/'candidate_manifest.json',{'configuration':config,'evaluation_settings':two,'model_sha256':sha(chosen_path),
            'protected_added_slots':True,'validation_eligible':final['eligible']})
        if sha(study['initial_model'])!=source_hash:raise AssertionError('Source file changed')
        dump(report/'summary.json',{'baseline':cold,'initialization_validation':start,'selected_bc':selected,
            'bc_updates_run':study['bc']['updates'],'ppo_started':ppo_started,
            'ppo_budget':study['ppo_steps'] if ppo_started else 0,
            'two_holdout_before':before['metrics'],'two_initialized_holdout':init_test['metrics'],'two_holdout_after':after['metrics'],
            'one_before':old['metrics'],'one_after':regression['metrics'],'final_validation':final,
            'heldout_gate':gate_result(after['metrics'],two['gate']),'candidate_path':str(chosen_path),'candidate_sha256':sha(chosen_path),
            'original_actor_preserved':True,'source_unchanged':True,'default_model_preserved':True,'fresh_test_used_for_selection':False})
        status('complete',status='completed',two_test=after['metrics']['stage_target'])
    except Exception as exc:
        status('failed',status='failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/protected_two_static.yaml')
    parser.add_argument('--run',action='store_true');args=parser.parse_args();torch.set_num_threads(1)
    study=resolve(args.config)
    print({'mode':'run' if args.run else 'check_only','bc_updates':study['bc']['updates'],'ppo_steps':study['ppo_steps']},flush=True)
    if args.run:run(study)


if __name__=='__main__':main()
