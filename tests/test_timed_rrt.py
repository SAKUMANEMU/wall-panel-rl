"""Scheduling must exploit time-dependent openings and validate waiting too."""
import time
from types import SimpleNamespace
import numpy as np
from wall_panel_rl.planning.timed_rrt import TimedRRT


def test_scheduler_waits_for_opening_and_rejects_unsafe_wait():
    env = SimpleNamespace(config={'simulation': {'timestep': 1., 'substeps': 1, 'max_steps': 8}},
                          sim_time=0., steps=0)
    planner = TimedRRT(env)
    path = [np.array([0.]), np.array([1.]), np.array([2.])]
    def gate(a,b,t):
        # The passage from 0 to 1 is occupied until t=2; waiting at 0 is safe.
        return not (a[0] == 0 and b[0] == 1 and t < 2)
    planner.edge_valid = gate
    timed = planner.schedule(path,time.perf_counter()+1)
    assert [q[0] for q in timed] == [0,0,0,1,2]
    assert planner.wait_count == 2
    assert all(gate(a,b,t) for t,(a,b) in enumerate(zip(timed[:-1],timed[1:])))
    # An approaching object also blocks the waiting location; no valid schedule.
    planner.edge_valid = lambda a,b,t: gate(a,b,t) and not (a[0] == b[0] == 0)
    assert planner.schedule(path,time.perf_counter()+1) is None
