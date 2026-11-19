"""Bidirectional RRT-Connect implemented here, with discretely checked edges."""
import time
import numpy as np


def interpolate(a, b, resolution):
    count = max(1, int(np.ceil(np.linalg.norm(np.asarray(b) - a) / resolution)))
    return np.linspace(a, b, count + 1)


class RRTConnect:
    def __init__(self, lower, upper, valid, edge_valid, config, seed=0, *, path_valid=None):
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.valid, self.edge_valid = valid, edge_valid
        self.config, self.rng = config, np.random.default_rng(seed)
        self.iterations = 0
        self.path_valid = path_valid
        self.rejected_paths = 0

    def accept_path(self, path):
        if self.path_valid is not None and not self.path_valid(path):
            self.rejected_paths += 1
            return False
        return True

    @staticmethod
    def trace(tree, index):
        path = []
        while index != -1:
            node, parent = tree[index]
            path.append(node)
            index = parent
        return path[::-1]

    def extend(self, tree, target):
        index = int(np.argmin([np.linalg.norm(node - target) for node, _ in tree]))
        near = tree[index][0]
        delta = target - near
        distance = np.linalg.norm(delta)
        if distance < 1e-10:
            return "reached", index
        new = near + delta * min(1, self.config["step_size"] / distance)
        if not self.edge_valid(near, new):
            return "trapped", index
        tree.append((new, index))
        return ("reached" if distance <= self.config["step_size"] else "advanced"), len(
            tree
        ) - 1

    def plan(self, start, goal):
        start, goal = np.asarray(start), np.asarray(goal)
        if not self.valid(start) or not self.valid(goal):
            return None
        deadline = time.perf_counter() + self.config["timeout"]
        if self.edge_valid(start, goal) and self.accept_path([start, goal]):
            return [start, goal]
        a, b, swapped = [(start, -1)], [(goal, -1)], False
        for iteration in range(self.config["max_iterations"]):
            self.iterations = iteration + 1
            if time.perf_counter() >= deadline:
                break
            target = (
                b[0][0]
                if self.rng.random() < self.config["goal_bias"]
                else self.rng.uniform(self.lower, self.upper)
            )
            status, ia = self.extend(a, target)
            if status != "trapped":
                while time.perf_counter() < deadline:
                    other, ib = self.extend(b, a[ia][0])
                    if other == "reached":
                        pa, pb = self.trace(a, ia), self.trace(b, ib)
                        path = pa + pb[-2::-1]
                        if swapped:
                            path.reverse()
                        candidate = self.smooth(path, deadline)
                        if self.accept_path(candidate):
                            return candidate
                        # A rejected final route is not a planning success.
                        # Keep searching within the SAME iteration/time budget.
                        break
                    if other == "trapped":
                        break
            a, b, swapped = b, a, not swapped
        return None

    def smooth(self, path, deadline):
        for _ in range(self.config["shortcut_attempts"]):
            if len(path) < 3 or time.perf_counter() >= deadline:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if (j > i + 1 and self.edge_valid(path[i], path[j]) and
                (self.path_valid is None or self.path_valid([path[i], path[j]]))):
                path = path[: i + 1] + path[j:]
        return path
