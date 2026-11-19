"""Validate actual execution waypoints as well as the coarse planner grid.

Finite sampling is not continuous collision certification. Keep both original
and refined samples, at the existing configured planning clearance. Physics
collision thresholds and actuator behavior are deliberately unchanged.
"""
import numpy as np
from wall_panel_rl.planning.rrt_connect import interpolate


def execution_route(path, increment):
    dense=[np.asarray(path[0],dtype=float)]
    for a,b in zip(path[:-1],path[1:]):dense.extend(interpolate(a,b,increment)[1:])
    return dense


def fine_edge_valid(checker,a,b,resolution):
    if resolution<=0:raise ValueError('Validation resolution must be positive')
    if not checker.edge_valid(a,b):return False
    return all(checker.is_configuration_valid(q) for q in interpolate(a,b,resolution))


def execution_edge_valid(checker,a,b,increment,resolution):
    # Original coarse-edge samples remain required; refinement cannot discard
    # a previously checked location by shifting its interpolation grid.
    if not checker.edge_valid(a,b):return False
    dense=execution_route([a,b],increment)
    return all(fine_edge_valid(checker,x,y,resolution) for x,y in zip(dense[:-1],dense[1:]))


def dense_route_valid(checker,path,resolution):
    if path is None or len(path)<2:return False
    return all(fine_edge_valid(checker,a,b,resolution) for a,b in zip(path[:-1],path[1:]))
