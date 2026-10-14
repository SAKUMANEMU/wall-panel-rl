"""Identical rollout and ground-truth metrics for every high-level method."""
import time
import numpy as np
from wall_panel_rl.baselines.classical import ClassicalController


def rollout(env, method, seed, model=None, on_step=None, on_plan=None):
    observation, info = env.reset(seed=seed, options={"noise_seed": seed + 1000000})
    scenario = {
        "seed": seed,
        "noise_seed": seed + 1000000,
        "initial_q": env.joint_state()[0].tolist(),
        "obstacles": [
            {"center": c.tolist(), "dimensions": d.tolist()} for c, d in env.obstacles
        ],
        "target_position": env.target_pose[0].tolist(),
        "target_orientation": env.target_pose[1].tolist(),
        "obstacle_motion": [{k: v.tolist() if isinstance(v, np.ndarray) else v for k,v in spec.items()} for spec in env.obstacle_motion],
        "reference_joint_path": [q.tolist() for q in env.reference_path],
        "reference_panel_positions": [p.tolist() for p in env.reference_positions],
    }
    planner = ClassicalController(env, seed) if method == "rrt" else None
    if on_plan is not None and planner is not None:
        on_plan(env, planner)
    failed = planner is not None and not planner.planning_success
    positions = [list(env.panel_pose()[0])]
    joint_positions = [env.joint_state()[0].tolist()]
    obstacle_positions = [[c.tolist() for c, _ in env.obstacles]]
    cumulative_reward, steps = 0.0, 0
    begin = time.perf_counter()
    if not failed:
        for steps in range(1, env.config["simulation"]["max_steps"] + 1):
            action = (
                planner.action(observation)
                if planner
                else model.predict(observation, deterministic=True)[0]
            )
            observation, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
            positions.append(list(env.panel_pose()[0]))
            joint_positions.append(env.joint_state()[0].tolist())
            obstacle_positions.append([c.tolist() for c, _ in env.obstacles])
            if on_step:
                on_step(env, steps, info)
            if done or truncated:
                break
    elapsed = time.perf_counter() - begin
    result = {
        "seed": seed,
        "success": bool(info["success"]) and not failed,
        "collision": bool(info["collision"]),
        "joint_limit_violation": info["joint_limit_violation"],
        "planning_success": planner.planning_success if planner else np.nan,
        "ik_success": planner.ik_success if planner else np.nan,
        "position_error_m": info["position_error"],
        "orientation_error_deg": np.rad2deg(info["orientation_error"]),
        "steps": steps,
        "joint_path_length_rad": env.joint_path_length,
        "cartesian_path_length_m": env.cartesian_path_length,
        "planning_time_s": planner.planning_time if planner else 0.0,
        "execution_time_s": elapsed,
        "simulated_time_s": steps
        * env.config["simulation"]["substeps"]
        * env.config["simulation"]["timestep"],
        "cumulative_reward": cumulative_reward,
        "replans": planner.replans if planner else 0,
        "replan_failures": planner.replan_failures if planner else 0,
        "hold_steps": planner.hold_steps if planner else 0,
        "physics_elapsed_s": env.sim_time,
    }
    return result, {
        "scenario": scenario,
        "panel_positions": positions,
        "joint_positions": joint_positions,
        "obstacle_positions": obstacle_positions,
    }
