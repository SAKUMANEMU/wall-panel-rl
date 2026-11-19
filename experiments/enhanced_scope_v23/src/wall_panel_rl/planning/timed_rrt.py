"""RRT geometric candidates with time-expanded advance/wait path scheduling.

Known obstacle motion is predicted on simulation time. Each scheduled edge checks
both arm and payload at intermediate times; this is discrete, not certified CCD.
"""
import heapq
import time
import numpy as np
from wall_panel_rl.planning.rrt_connect import RRTConnect


class TimedRRT:
    def __init__(self, env, seed=0):
        self.env = env
        self.seed = seed
        self.iterations = 0
        self.dt = env.config['simulation']['timestep'] * env.config['simulation']['substeps']
        self.wait_count = 0

    def edge_valid(self, a, b, start_time):
        e = self.env
        # Resolve both moving obstacles and arm motion within each control step.
        samples = max(4, int(np.ceil(np.linalg.norm(b-a) / .015)))
        try:
            for fraction in np.linspace(0., 1., samples+1):
                e.move_obstacles(start_time + fraction*self.dt)
                if not e.collision.is_configuration_valid(a+(b-a)*fraction):
                    return False
            return True
        finally:
            e.move_obstacles(e.sim_time)

    def densify(self, path):
        out = [np.array(path[0])]
        increment = self.env.max_increment * .65
        for a,b in zip(path[:-1],path[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(b-a))/increment)))
            out.extend(np.linspace(a,b,count+1)[1:])
        return out

    def schedule(self, path, deadline):
        e = self.env
        horizon = e.config['simulation']['max_steps'] - e.steps
        n = len(path)-1
        if n > horizon:
            return None
        # State (path index, arrival control tick); wait and advance each cost 1.
        queue = [(n, 0, 0)]
        parents = {(0,0): None}
        while queue and time.perf_counter() < deadline:
            _, tick, index = heapq.heappop(queue)
            self.iterations += 1
            if index == n:
                states = []
                state = (index,tick)
                while state is not None:
                    states.append(state)
                    state = parents[state]
                states.reverse()
                self.wait_count = sum(a[0] == b[0] for a,b in zip(states[:-1],states[1:]))
                return [path[i] for i,t in states]
            if tick >= horizon:
                continue
            for next_index in [index+1,index]:
                state = (next_index,tick+1)
                if state in parents or tick+1+n-next_index > horizon:
                    continue
                if self.edge_valid(path[index],path[next_index],e.sim_time+tick*self.dt):
                    parents[state] = (index,tick)
                    heapq.heappush(queue,(tick+1+n-next_index,tick+1,next_index))
        return None

    def plan(self, start, goal):
        e = self.env
        deadline = time.perf_counter() + e.config['rrt'].get('timed_timeout',30.)
        self.iterations = 0
        # Generate routes at several future snapshots; do not inflate obstacles
        # into their full swept volumes. The scheduler handles arrival times.
        for candidate, future in enumerate([0., .5, 1., 2.]):
            remaining = deadline-time.perf_counter()
            if remaining <= 0:
                break
            config = dict(e.config['rrt'], timeout=min(4., remaining*.4))
            planner = RRTConnect(e.lower,e.upper,e.collision.is_configuration_valid,
                                 e.collision.edge_valid,config,self.seed+candidate)
            try:
                e.move_obstacles(e.sim_time+future)
                path = planner.plan(start,goal)
            finally:
                e.move_obstacles(e.sim_time)
            if path is not None:
                # Give other candidate geometries a chance if timing is blocked.
                scheduled = self.schedule(self.densify(path), min(deadline,time.perf_counter()+6.))
                if scheduled is not None:
                    return scheduled
        return None
