"""Small actor BC steps on corrective prefixes, anchored to successful PPO states."""
import _bootstrap
import argparse,copy,json,shutil
from datetime import datetime
from pathlib import Path
import numpy as np
import torch,yaml
from stable_baselines3 import PPO
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.dynamic_four_training import evaluate_ppo,compose_mixture
from wall_panel_rl.rl.four_ppo_continuation import state_digest
from run_targeted_bc import read,dump,sha


def score(m,b):return [int(m['joint_limit']==0 and m['collision']<=b['collision']),m['success'],-m['collision'],-m['timeout']]


def run(spec,config_file):
    torch.set_num_threads(1);report=ROOT/spec['bc_report'];models=ROOT/spec['bc_models']
    if report.exists() or models.exists():raise FileExistsError('Refusing corrective BC overwrite')
    collection=read(ROOT/spec['report_root']/'protocol.json');hashes=read(ROOT/spec['report_root']/'data_hashes.json')
    for f,h in hashes.items():assert sha(ROOT/f)==h
    data=torch.load(ROOT/spec['output_root']/'corrections.pt',map_location='cpu',weights_only=False)
    excluded=set(collection['excluded_seeds']);train=set(collection['train_seeds'])
    assert data['train_only'] and data['dynamic_only']
    assert all(e['row']['accepted_for_imitation'] and e['case']['task']=='dynamic' and e['row']['seed'] in train and e['row']['seed'] not in excluded for e in data['episodes'])
    x=torch.as_tensor(np.stack([a for a,b in data['training_pairs']]),dtype=torch.float32);y=torch.as_tensor(np.stack([b for a,b in data['training_pairs']]),dtype=torch.float32)
    anchors=torch.as_tensor(data['success_anchor_observations'],dtype=torch.float32)
    assert x.shape[1:]==anchors.shape[1:]==(77,) and y.shape[1:]==(7,) and torch.isfinite(x).all() and torch.isfinite(y).all()
    source=PPO.load(ROOT/spec['source_model'],device='cpu');source_hash=sha(ROOT/spec['source_model']);assert source_hash==collection['source_sha256']
    initial=copy.deepcopy(source.policy.state_dict());counter=source.num_timesteps;optimizer_hash=state_digest(source.policy.optimizer.state_dict())
    report.mkdir(parents=True);models.mkdir(parents=True)
    visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/report.name
    if visible.resolve()!=report.resolve():visible.symlink_to(report,target_is_directory=True)
    state={'status':'running','ppo_steps':0,'static_episodes':0}
    def status(phase,**extra):
        state.update(phase=phase,updated_at=datetime.now().astimezone().isoformat(),**extra);dump(report/'status.json',state);print('CORRECTIVE_BC',json.dumps(state),flush=True)
    files=[str(config_file.relative_to(ROOT)),'scripts/train_dynamic_correction_bc.py','src/wall_panel_rl/rl/dynamic_four_training.py',
        'src/wall_panel_rl/rl/multi_obstacle_adapter.py','src/wall_panel_rl/rl/conditional_recovery.py','src/wall_panel_rl/envs/wall_panel_env.py']
    code={f:sha(ROOT/f) for f in files};inputs=dict(hashes);inputs[spec['source_model']]=source_hash
    validation=read(ROOT/spec['data_root']/'validation_cases.json');inputs[spec['data_root']+'/validation_cases.json']=sha(ROOT/spec['data_root']/'validation_cases.json')
    effective=dict(spec,reward_scale=.01)
    dump(report/'protocol.json',{'spec':effective,'configuration':collection['configuration'],'evaluation_settings':collection['evaluation_settings'],
        'source_sha256':source_hash,'input_hashes':inputs,'code_hashes':code,'rule':'Corrective first16 expert steps plus successful source-policy action retention; validation-only success/collision selection, earlier wins ties; no test evaluation here',
        'fresh_bc_optimizer':True,'ppo_optimizer_preserved':True,'critic_base_std_frozen':True,'static_labels':0})
    for f in files:
        dest=report/'code_archive'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dest)
    rng=np.random.default_rng(spec['seed'])
    guard=torch.cat([x[rng.choice(len(x),min(512,len(x)),replace=False)],anchors[rng.choice(len(anchors),min(512,len(anchors)),replace=False)]])
    with torch.no_grad():
        anchor_targets=source.policy.tracking_mean(anchors).detach().clone()
        d=source.policy.get_distribution(guard).distribution;original=torch.distributions.Normal(d.loc.detach().clone(),d.scale.detach().clone())
    history=[];best=None;selected=None;baseline=None
    try:
        for arm,lr in enumerate(spec['bc_learning_rates']):
            model=PPO.load(ROOT/spec['source_model'],device='cpu');model.policy.configure_learning(train_critic=False)
            params=[p for n,p in model.policy.named_parameters() if n.startswith('multi_adapter.') and p.requires_grad]
            optimizer=torch.optim.Adam(params,lr=lr);rng=np.random.default_rng(spec['seed']+10000000);accepted=0;rejected=0
            def checkpoint(update):
                nonlocal baseline,best,selected
                status('validating',arm=arm,learning_rate=lr,bc_updates=update)
                result=evaluate_ppo(model,collection['configuration'],collection['evaluation_settings'],validation,.12,amplitude=.18)
                if baseline is None:
                    baseline=result['metrics'];dump(report/'baseline_validation.json',result)
                if update==0:assert result['metrics']==baseline
                value=score(result['metrics'],baseline)
                event={'arm':arm,'learning_rate':lr,'bc_updates':update,'metrics':result['metrics'],'score':value,'accepted_updates':accepted,'rejected_updates':rejected}
                assert model.num_timesteps==counter and state_digest(model.policy.optimizer.state_dict())==optimizer_hash
                assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in initial.items() if not k.startswith('multi_adapter.'))
                model.save(models/f'arm{arm}_bc{update}.zip')
                if best is None or tuple(value)>best:
                    best=tuple(value);selected=copy.deepcopy(event);model.save(models/'selected_model.zip');torch.save(optimizer.state_dict(),models/'selected_warm_optimizer.pt')
                dump(report/'validation'/f'arm{arm}_bc{update}.json',result);history.append(event);dump(report/'history.json',history)
                status('checkpoint',**event,selected=selected)
            checkpoint(0)
            for update in range(1,spec['bc_updates']+1):
                ids=rng.integers(0,len(x),spec['batch_size']);ids_anchor=rng.integers(0,len(anchors),spec['batch_size'])
                with torch.no_grad():
                    d=model.policy.get_distribution(guard).distribution;previous=torch.distributions.Normal(d.loc.detach().clone(),d.scale.detach().clone())
                weights=[p.detach().clone() for p in params];adam=copy.deepcopy(optimizer.state_dict())
                expert_loss=(model.policy.tracking_mean(x[ids])-y[ids]).square().mean()
                anchor_loss=(model.policy.tracking_mean(anchors[ids_anchor])-anchor_targets[ids_anchor]).square().mean()
                loss=expert_loss+spec['anchor_weight']*anchor_loss
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite correction loss')
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(params,.5);optimizer.step()
                with torch.no_grad():
                    current=model.policy.get_distribution(guard).distribution
                    step_kl=float(torch.distributions.kl_divergence(previous,current).sum(-1).mean());source_kl=float(torch.distributions.kl_divergence(original,current).sum(-1).mean())
                keep=bool(np.isfinite([step_kl,source_kl]).all() and step_kl<=spec['maximum_step_kl'] and source_kl<=spec['maximum_source_kl'])
                if keep:accepted+=1
                else:
                    with torch.no_grad():
                        for p,w in zip(params,weights):p.copy_(w)
                    optimizer.load_state_dict(adam);rejected+=1
                with (report/'loss.jsonl').open('a') as f:f.write(json.dumps({'arm':arm,'update':update,'expert_loss':float(expert_loss.detach()),'anchor_loss':float(anchor_loss.detach()),'source_kl':source_kl,'step_kl':step_kl,'accepted':keep})+'\n')
                if update in spec['bc_checkpoints']:checkpoint(update)
        chosen=PPO.load(models/'selected_model.zip',device='cpu');shutil.copy2(models/'selected_model.zip',report/'selected_model.zip')
        fresh=torch.optim.Adam([p for n,p in chosen.policy.named_parameters() if n.startswith('multi_adapter.')],lr=1e-7)
        torch.save({'optimizer':fresh.state_dict(),'rng':np.random.default_rng(spec['seed']+40000000).bit_generator.state,
            'accepted_updates':0,'rejected_updates':0,'pool':data['training_pairs'],'ppo_num_timesteps':counter},models/'selected_auxiliary_state.pt')
        dump(report/'candidate_manifest.json',{'configuration':collection['configuration'],'evaluation_settings':collection['evaluation_settings'],
            'model_sha256':sha(report/'selected_model.zip'),'dynamic_only_labels':True,'selected_bc_updates':selected['bc_updates'],'selected_bc_lr':selected['learning_rate']})
        improved=tuple(selected['score'])>tuple(history[0]['score'])
        summary={'selected':selected,'selected_bc_updates':selected['bc_updates'],'validation_improved':improved,'source_validation':baseline,
            'bc_updates':spec['bc_updates']*len(spec['bc_learning_rates']),'teacher_episodes':len(data['episodes']),'teacher_transitions':len(x),
            'success_anchor_states':len(anchors),'ppo_steps':0,'static_labels':0,'source_unchanged':sha(ROOT/spec['source_model'])==source_hash,'independent_test_evaluated':False}
        for f,h in code.items():assert sha(ROOT/f)==h,'Code changed during BC'
        for f,h in inputs.items():assert sha(ROOT/f)==h,'Input changed during BC'
        dump(report/'summary.json',summary);status('complete',status='completed',summary=summary)
    except Exception as exc:status('failed',status='failed',error=repr(exc));raise


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'configs/four_dynamic_corrections.yaml');p.add_argument('--run',action='store_true');a=p.parse_args();spec=yaml.safe_load(a.config.read_text())
    print({'mode':'run' if a.run else 'check_only','learning_rates':spec['bc_learning_rates'],'steps_per_arm':spec['bc_updates']},flush=True)
    if a.run:run(spec,a.config.resolve())

if __name__=='__main__':main()
