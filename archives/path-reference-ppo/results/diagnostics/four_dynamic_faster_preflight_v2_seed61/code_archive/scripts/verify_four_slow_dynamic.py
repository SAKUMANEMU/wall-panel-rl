"""Small physical preflight for slow dynamic obstacles; no training entry point."""
import _bootstrap
import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from stable_baselines3 import PPO
from run_targeted_bc import read,dump,sha
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.episode_policy import EpisodeChoice
from wall_panel_rl.rl.episode_imitation import make_rrt,episode_record
from wall_panel_rl.rl.slow_dynamic_four import (load_profile,dynamic_configuration,fresh_dynamic_case,
    SlowDynamicFourEnv,planning_snapshot,assert_planning_paused)


def execute(model,config,settings,case,owner):
    env=SlowDynamicFourEnv(copy.deepcopy(config),[],settings);total=0.;frames=[]
    try:
        obs,info=env.reset(options={'scenario':case['scene'],'task':'dynamic','strength':case['strength']})
        initial=planning_snapshot(env);reference=env.cursor.path.copy()
        expert=make_rrt(env,int(case['scene']['seed'])+100000) if owner=='rrt' else None
        assert_planning_paused(env,initial)
        if expert is not None:assert expert.timed
        def frame():
            return {'time_s':env.env.sim_time,'joints':env.env.joint_state()[0].tolist(),
                'panel_position':list(env.env.panel_pose()[0]),'reference_index':int(env.cursor.index),
                'obstacle_centers':[c.tolist() for c,_ in env.env.obstacles],
                'obstacle_velocities':[env.env.obstacle_velocity(i).tolist() for i in range(4)]}
        frames.append(frame())
        if expert is not None and not expert.planning_success:
            info=dict(info,termination_reason='planning_failure')
        else:
            while True:
                before=planning_snapshot(env)
                action=expert.action(obs) if expert else model.predict(obs,deterministic=True)[0]
                assert_planning_paused(env,before)
                obs,reward,done,truncated,info=env.step(action);total+=reward
                frames.append(frame())
                if done or truncated:break
        assert np.array_equal(reference,env.cursor.path)
        anchors=np.array([m['anchor'] for m in case['scene']['obstacle_motion']])
        centers=np.array([f['obstacle_centers'] for f in frames]);velocities=np.array([f['obstacle_velocities'] for f in frames])
        offsets=np.linalg.norm(centers-anchors,axis=2);speeds=np.linalg.norm(velocities,axis=2)
        assert offsets.max()<=case['strength'][0]+1e-10 and speeds.max()<=case['strength'][1]+1e-10
        row=episode_record(env,info,owner,total,expert.planning_time if expert else 0.)
        row.update(max_displacement_from_anchor_m=float(offsets.max()),max_observed_speed_m_s=float(speeds.max()),
            obstacle_axis_ranges_m=np.ptp(centers,axis=0).tolist(),physics_elapsed_s=env.env.sim_time,
            planning_paused_time_and_restored_scene=True,original_reference_preserved=True,
            obstacle_count=4,known_motion=True,dynamic_planning_mode='timed',
            replans=expert.replans if expert else 0,hold_steps=expert.hold_steps if expert else 0,
            scheduled_waits=expert.scheduled_waits if expert else 0)
        return row,frames
    finally:env.close()


def run(spec,config_file):
    root=ROOT/spec['preflight']['report_root']
    if root.exists():raise FileExistsError('Refusing preflight overwrite')
    root.mkdir(parents=True);torch.set_num_threads(1)
    source=ROOT/spec['source_report'];model_file=source/'selected_model.zip';manifest=read(source/'candidate_manifest.json')
    model_hash=sha(model_file)
    if model_hash!=manifest['model_sha256']:raise ValueError('Checkpoint does not match the selected source manifest')
    model=PPO.load(model_file,device='cpu');original_counter=model.num_timesteps;original_updates=model._n_updates
    weights={k:v.clone() for k,v in model.policy.state_dict().items()}
    config,settings=dynamic_configuration(manifest['configuration'],manifest['evaluation_settings'],spec)
    files=[str(Path(config_file).relative_to(ROOT)),'scripts/verify_four_slow_dynamic.py','scripts/demo_four_slow_dynamic.py',
        'scripts/demo_autonomous_ppo.py','src/wall_panel_rl/rl/slow_dynamic_four.py','src/wall_panel_rl/rl/conditional_recovery.py',
        'src/wall_panel_rl/rl/episode_imitation.py','src/wall_panel_rl/rl/avoidance_curriculum.py',
        'src/wall_panel_rl/rl/autonomous_dynamic_ppo.py','src/wall_panel_rl/baselines/classical.py',
        'src/wall_panel_rl/planning/timed_rrt.py','src/wall_panel_rl/planning/rrt_connect.py',
        'src/wall_panel_rl/envs/wall_panel_env.py']
    hashes={f:sha(ROOT/f) for f in files}
    dump(root/'protocol.json',{'profile':spec,'source_model':str(model_file),'source_sha256':model_hash,
        'configuration':config,'evaluation_settings':settings,'training_steps':0,
        'planning_time_basis':'simulation clock paused during planning','code_hashes':hashes,
        'scope':'small functional preflight, not an independent success-rate study'})
    def status(phase,**extra):
        state={'phase':phase,'updated_at':datetime.now().astimezone().isoformat(),'training_steps':0,**extra}
        dump(root/'status.json',state);print('SLOW_DYNAMIC',json.dumps(state),flush=True)
    cases=[];rejects=[];seed=int(spec['preflight']['first_scene_seed']);count=int(spec['preflight']['cases'])
    try:
        while len(cases)<count:
            if seed>=spec['preflight']['first_scene_seed']+count*spec['placement_attempts']:
                raise RuntimeError('Too many geometry-only placement rejections')
            status('generating_scenes',accepted=len(cases),requested=count,scene_seed=seed)
            try:
                case,_=fresh_dynamic_case(config,settings,spec,seed,model=model)
                cases.append(case)
            except (RuntimeError,ValueError) as exc:
                stage='nominal_reference' if 'Cannot build a complete nominal route' in str(exc) else 'geometry'
                rejects.append({'seed':seed,'stage':stage,'reason':str(exc)})
            seed+=1
        dump(root/'cases.json',cases);dump(root/'scene_construction_rejections.json',rejects)
        choice=EpisodeChoice(spec['expert_probability'],spec['preflight']['controller_seed']);rows=[]
        for i,case in enumerate(cases):
            owner=choice.begin_episode();status('executing',case=i+1,total=count,controller=owner,scene_seed=case['scene']['seed'])
            row,frames=execute(model,config,settings,case,owner);rows.append(row);choice.end_episode()
            dump(root/'traces'/f"{case['scene']['seed']}_{owner}.json",{'result':row,'frames':frames})
            dump(root/'episodes.json',rows);print('SLOW_DYNAMIC_RESULT',json.dumps(row),flush=True)
        assert model.num_timesteps==original_counter and model._n_updates==original_updates
        assert all(torch.equal(v,model.policy.state_dict()[k]) for k,v in weights.items())
        assert sha(model_file)==model_hash
        summary={'episodes':len(rows),'expert_probability':.5,'training_steps':0,'source_unchanged':True,
            'model_weights_and_update_counter_unchanged':True,'independent_test':False,
            'by_controller':{owner:dict(Counter(r['reason'] for r in rows if r['controller']==owner)) for owner in ('rrt','ppo')},
            'total_outcomes':dict(Counter(r['reason'] for r in rows)),
            'scene_construction_rejections':len(rejects),'rejections_by_stage':dict(Counter(r['stage'] for r in rejects)),
            'planning_paused_time_and_restored_scene':True,
            'original_reference_preserved':True,'motion_bounds_verified':True,
            'max_displacement_from_anchor_m':max(r['max_displacement_from_anchor_m'] for r in rows),
            'max_observed_speed_m_s':max(r['max_observed_speed_m_s'] for r in rows),
            'all_executed_cases_moved_in_xyz':all(np.all(np.array(r['obstacle_axis_ranges_m'])>1e-7) for r in rows if r['steps']>0),
            'default_static_model_and_entries_preserved':True}
        dump(root/'summary.json',summary)
        for f,expected in hashes.items():
            assert sha(ROOT/f)==expected,'Runtime source changed during preflight'
            target=root/'code_archive'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
        text=(f"# Four moving obstacles: functional preflight\n\nUnchanged model from `{spec['source_report']}`. "
              f"Four different-size boxes retain randomized on-path anchors. Known smooth three-axis motion has independent random phases/frequencies, "
              f"a {spec['motion']['amplitude_m']:g} m displacement norm bound and {spec['motion']['max_speed_m_s']:g} m/s velocity norm bound. "
              "Planning does not advance the physics clock. RRT uses time-aware collision prediction and may traverse swept regions at safe times; "
              "PPO has no online avoidance planner. Noise is zero.\n\n"
              '''This small run checks functionality, not general success rate. Scene construction checks geometry and requires a completed obstacle-free nominal rollout to define the on-path anchors. Construction failures are logged separately; no dynamic policy/planning outcome is filtered. Both branches use the same 77-observation / 7-action interface and original reference endpoint. There is no training, checkpoint promotion, or fallback to the other branch.\n\n| Seed | Controller | Outcome | Steps | Position (m) | Planning (s) |\n|---|---|---|---:|---:|---:|\n''')
        for r in rows:text+=f"| {r['seed']} | {r['controller']} | {r['reason']} | {r['steps']} | {r['position_m']:.4f} | {r['planning_seconds']:.2f} |\n"
        text+='\n```json\n'+json.dumps(summary,indent=2)+'\n```\n\nSee `cases.json` for exact phases/anchors, `episodes.json` for results, `traces/` for moving box/arm trajectories, and `protocol.json` for configuration and hashes.\n'
        (root/'README.md').write_text(text);status('completed',episodes=len(rows),summary=summary)
        visible=Path('/home/johnsonchen/wall-panel-rl/results/diagnostics')/root.name
        if visible.resolve()!=root.resolve():
            if visible.exists() or visible.is_symlink():raise FileExistsError('Visible report alias already exists')
            visible.symlink_to(root,target_is_directory=True)
        print('Report:',visible,flush=True)
    except Exception as exc:status('failed',error=repr(exc));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/four_slow_dynamic.yaml')
    parser.add_argument('--run',action='store_true',help='Generate the configured scenes and physically verify the episode mixture')
    args=parser.parse_args();spec=load_profile(args.config)
    print({'mode':'functional_preflight' if args.run else 'check_only','training_steps':0,'expert_probability':.5,'motion':spec['motion']},flush=True)
    if args.run:run(spec,args.config.resolve())


if __name__=='__main__':main()
