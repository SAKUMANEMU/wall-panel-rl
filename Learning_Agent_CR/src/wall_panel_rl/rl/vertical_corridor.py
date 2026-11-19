"""Place a larger height stagger along a preliminary physical execution.

The preliminary result is not a success filter. All measured states are a
placement guide only; the immutable policy observation reference stays intact.
Final geometry guards and actual task outcomes are never replaced by the guide.
"""
import copy
import numpy as np
from wall_panel_rl.utils.config import ROOT
from wall_panel_rl.rl.slow_dynamic_four import load_profile, SlowDynamicFourEnv
from wall_panel_rl.rl.motion_aware_placement import fresh_motion_aware_case, fit_motion_anchors
from wall_panel_rl.rl.demo_scene_adjustments import adjust_obstacle_heights
from wall_panel_rl.rl.progress_recovery import ProgressRecoveryController


def preliminary_route(model, config, settings, case, selection):
    env = SlowDynamicFourEnv(copy.deepcopy(config), [], copy.deepcopy(settings))
    joints, positions = [], []
    try:
        obs, _ = env.reset(options={'scenario': case['scene'], 'task': 'dynamic', 'strength': case['strength']})
        original_reference = env.cursor.path.copy()
        agent = ProgressRecoveryController(model, env, selection['selected']['horizon'],
                                           selection['gains'], selection['recovery'])
        def capture():
            joints.append(env.env.joint_state()[0].tolist())
            positions.append(list(env.env.panel_pose()[0]))
        capture()
        while True:
            clock = (env.env.steps, env.env.sim_time)
            action = agent.action(obs)
            assert clock == (env.env.steps, env.env.sim_time)
            obs, _, done, truncated, info = env.step(action)
            capture()
            if done or truncated:
                break
        assert np.array_equal(original_reference, env.cursor.path)
        return {'joint_positions': joints, 'panel_positions': positions, 'steps': env.env.steps,
                'termination_reason': info['termination_reason'],
                'route_kind': 'preliminary_physical_episode_no_success_filter'}
    finally:
        env.close()


def fresh_vertical_corridor_case(config, settings, profile, seed, model, selection):
    preliminary_profile = load_profile(ROOT/profile['preliminary_motion_profile'])
    for key in ['amplitude_m', 'max_speed_m_s']:
        if profile['motion'][key] != preliminary_profile['motion'][key]:
            raise ValueError('Preliminary and final motion strengths must match')
    preliminary = fresh_motion_aware_case(config, settings, preliminary_profile, seed, model)
    preliminary = adjust_obstacle_heights(config, preliminary,
                                         preliminary_profile['demo_obstacle_height_offsets_m'])
    nominal = preliminary_route(model, config, settings, preliminary, selection)
    candidate = copy.deepcopy(preliminary)
    scene = candidate['scene']
    scene['original_nominal_motion'] = scene['nominal_motion']
    scene['nominal_motion'] = nominal
    scene['preliminary_episode'] = {'controller': 'ppo_progress', 'steps': nominal['steps'],
                                    'reason': nominal['termination_reason'], 'success_filter': False}
    # Reanchoring creates new trajectories. Do not mark their O2 translation as
    # already applied just because the preliminary layout had that translation.
    scene.pop('demo_height_offsets_m', None)
    scene.pop('unadjusted_motion_anchor_placement', None)
    candidate = fit_motion_anchors(config, settings, profile, candidate)
    candidate = adjust_obstacle_heights(config, candidate, profile['demo_obstacle_height_offsets_m'])
    if profile.get('enforce_final_stagger'):
        scene = candidate['scene']
        order = scene['unadjusted_motion_anchor_placement']['route_slot_order']
        heights = np.array([scene['obstacle_motion'][i]['anchor'][2] for i in order])
        span = float(np.ptp(heights))
        adjacent = float(np.min(np.abs(np.diff(heights))))
        if span < profile['height_stagger']['min_span_m'] or adjacent < profile['height_stagger']['min_adjacent_delta_m']:
            raise RuntimeError('Final height stagger too small after the O2 translation')
        scene['final_height_stagger'] = {'anchor_height_span_m': span,
                                         'minimum_adjacent_delta_m': adjacent,
                                         'checked_after_O2_translation': True}
    return candidate
