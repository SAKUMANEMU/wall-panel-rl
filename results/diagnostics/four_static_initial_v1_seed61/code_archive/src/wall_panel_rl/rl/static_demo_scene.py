"""Fresh static demo scenes; placement checks never inspect policy outcomes."""
import copy
import numpy as np
from wall_panel_rl.envs.wall_panel_env import WallPanelEnv
from wall_panel_rl.rl.avoidance_curriculum import stage_scene
from wall_panel_rl.utils.metrics import capture_scenario


def fresh_static_case(config, settings, seed=None, attempts=20):
    stage = settings['active_stage']
    if stage['mode'] != 'static': raise ValueError('Fresh demo requires a static stage')
    requested = int(seed) if seed is not None else None
    rng = np.random.default_rng(); env = WallPanelEnv(copy.deepcopy(config))
    rejects = []
    try:
        for _ in range(1 if requested is not None else attempts):
            current = requested if requested is not None else int(rng.integers(2_000_000, 2_000_000_000))
            try:
                env.reset(seed=current)
                scene = stage_scene(capture_scenario(env,current),stage)
                for i,a in enumerate(scene['obstacles']):
                    for b in scene['obstacles'][i+1:]:
                        gap=np.maximum(np.abs(np.asarray(a['center'])-b['center'])-
                            (np.asarray(a['dimensions'])+b['dimensions'])/2,0)
                        if np.linalg.norm(gap)<config['obstacles']['min_obstacle_gap']:
                            raise RuntimeError('Static demo boxes violate minimum spacing')
                env.reset(seed=current,options={'scenario':scene})
                if not env.collision.is_configuration_valid(scene['initial_q']):
                    raise RuntimeError('Static demo start overlaps an obstacle')
                if not env.collision.is_configuration_valid(scene['reference_joint_path'][-1]):
                    raise RuntimeError('Static demo goal overlaps an obstacle')
                return {'suite':'stage_target','task':'static',
                    'strength':[stage['amplitude'],stage['speed']],'scene':scene}, rejects
            except RuntimeError as exc:
                rejects.append({'seed':current,'reason':str(exc)})
                if requested is not None: raise
        raise RuntimeError(f'Could not generate a valid random static scene: {rejects}')
    finally: env.close()
