from pathlib import Path
from typing import List, Tuple, Optional
import networkx as nx
import json
import random
import numpy as np
import inspyred


class NSGAGraphOptimizer:
    def __init__(
        self,
        graph_edges_path: Path,
        graph_nodes_path: Path,
        origin: int,
        destination: int,
        vehicle_allowed_in_lez: bool,
        population_size: int = 50,
        generations: int = 50,
        max_archive_size: int = 25,
        penalty_signal: float = 0.005,
        penalty_lanes_factor: float = 1.2,
        seed: Optional[int] = None
    ):
        self.origin = origin
        self.destination = destination
        self.vehicle_allowed_in_lez = vehicle_allowed_in_lez
        self.population_size = population_size
        self.generations = generations
        self.max_archive_size = max_archive_size
        self.penalty_signal = penalty_signal
        self.penalty_lanes_factor = penalty_lanes_factor
        self.random = random.Random(seed)

        self.archive = []
        self.history_hv = []
        self.history_spread = []
        self.n_evaluations = 0

        self.graph = self._load_graph(graph_edges_path, graph_nodes_path)
        self.norm_distance = max(nx.get_edge_attributes(self.graph, 'length').values())
        self.norm_time = max(nx.get_edge_attributes(self.graph, 'travel_time').values())
        self.norm_fuel = max(nx.get_edge_attributes(self.graph, 'fuel_consumption').values())

    def _load_graph(self, edges_path: Path, nodes_path: Path) -> nx.MultiDiGraph:
        with open(edges_path) as f:
            edges = json.load(f)
        with open(nodes_path) as f:
            nodes = json.load(f)
        G = nx.MultiDiGraph()
        for node in nodes:
            G.add_node(int(node["osmid"]), **node)
        for edge in edges:
            edge_data = dict(edge)
            edge_key = edge_data.pop("key")
            G.add_edge(edge["u"], edge["v"], key=edge_key, **edge_data)
        return G

    def _evaluate_route(self, route: List[Tuple[int, int, int]]) -> Tuple[float, float, float]:
        total_distance = total_time = total_fuel = 0.0
        for u, v, k in route:
            if not self.graph.has_edge(u, v, key=k):
                return float('inf'), float('inf'), float('inf')
            edge = self.graph[u][v][k]
            if edge.get("oneway", False) and (u, v) != (edge["u"], edge["v"]):
                return float('inf'), float('inf'), float('inf')
            if edge.get("lez", False) and not self.vehicle_allowed_in_lez:
                return float('inf'), float('inf'), float('inf')
            distance = edge["length"]
            time = edge["travel_time"]
            fuel = edge["fuel_consumption"]
            if self.graph.nodes[u].get("highway") in ["traffic_signals", "stop"] or \
               self.graph.nodes[v].get("highway") in ["traffic_signals", "stop"]:
                fuel += self.penalty_signal
            try:
                lanes = max([int(l) for l in edge["lanes"]]) if isinstance(edge.get("lanes"), list) else int(edge.get("lanes", 1))
            except:
                lanes = 1
            if lanes == 1:
                fuel *= self.penalty_lanes_factor
            total_distance += distance
            total_time += time
            total_fuel += fuel
        return (round(total_distance / self.norm_distance, 6),
                round(total_time / self.norm_time, 6),
                round(total_fuel / self.norm_fuel, 6))

    def _construct_solution(self) -> List[Tuple[int, int, int]]:
        current, route, visited_edges, attempts = self.origin, [], set(), 0
        while current != self.destination and attempts < 250:
            neighbors = [(u, v, k) for u, v, k in self.graph.out_edges(current, keys=True)
                         if (u, v, k) not in visited_edges]
            if not neighbors:
                break
            u, v, k = self.random.choice(neighbors)
            route.append((u, v, k))
            visited_edges.add((u, v, k))
            current = v
            attempts += 1
        return route if current == self.destination else []

    def _observer(self, population, num_generations, num_evaluations, args):
        valid = [
            (ind.candidate, tuple(ind.fitness))
            for ind in population
            if all(np.isfinite(ind.fitness))
        ]
        archive = self._get_non_dominated(valid)
        archive = archive[:self.max_archive_size]
        objectives = [fit for _, fit in archive]
        hv = self._compute_hypervolume(objectives)
        spread = self._compute_spread(objectives)
        self.history_hv.append(hv)
        self.history_spread.append(spread)

    def _crossover(self, random, parents, args):
        children = []
        for i in range(0, len(parents), 2):
            p1, p2 = parents[i], parents[i+1 if i+1 < len(parents) else i]
            common = set(p1) & set(p2)
            if not common:
                children.extend([p1, p2])
                continue
            point = self.random.choice(list(common))
            cut1 = p1[:p1.index(point)+1]
            cut2 = p2[p2.index(point)+1:]
            child = cut1 + cut2
            children.append(child if self._is_valid_route(child) else p1)
            children.append(p2)
        return children

    def _mutate(self, random, candidates, args):
        mutated = []
        for route in candidates:
            if len(route) < 2:
                mutated.append(route)
                continue
            idx = self.random.randint(0, len(route)-1)
            subpath = self._construct_solution()
            mutated.append(route[:idx] + subpath if subpath else route)
        return mutated

    def _is_valid_route(self, route: List[Tuple[int, int, int]]) -> bool:
        return bool(route and route[0][0] == self.origin and route[-1][1] == self.destination
                    and all(route[i][1] == route[i+1][0] for i in range(len(route)-1)))

    def _generator(self, random, args):
        while True:
            s = self._construct_solution()
            if s:
                return s

    def _evaluator(self, candidates, args):
        return [self._evaluate_route(c) for c in candidates]

    def run(self):
        ea = inspyred.ec.emo.NSGA2(self.random)

        ea.variator = [
            lambda random, candidates, args: self._crossover(random, candidates, args),
            lambda random, candidates, args: self._mutate(random, candidates, args)
        ]

        ea.terminator = inspyred.ec.terminators.generation_termination

        self.history_hv = []
        self.history_spread = []

        final_pop = ea.evolve(
            generator=self._generator,
            evaluator=self._evaluator,
            pop_size=self.population_size,
            maximize=False,
            bounder=None,
            max_generations=self.generations,
            observers=[self._observer]
        )

        valid = [
            (ind.candidate, tuple(ind.fitness))
            for ind in final_pop
            if all(np.isfinite(ind.fitness))
        ]

        archive = self._get_non_dominated(valid)
        self.archive = archive[:self.max_archive_size]

        self.history_hv = self._compute_history_metric(valid, metric="hv")
        self.history_spread = self._compute_history_metric(valid, metric="spread")
        self.n_evaluations = len(final_pop)

        return self.archive

    def _get_non_dominated(self, population: List[Tuple[List[Tuple[int, int, int]], Tuple[float, float, float]]]):
        return [p1 for i, p1 in enumerate(population)
                if not any(self._dominates(p2[1], p1[1]) for j, p2 in enumerate(population) if i != j)]

    def _dominates(self, f1, f2):
        return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

    def _compute_history_metric(self, population, metric="hv"):
        front = [f for _, f in population if all(np.isfinite(f))]
        front.sort()
        history = []
        for i in range(1, len(front)+1):
            sub = front[:i]
            history.append(self._compute_hypervolume(sub) if metric == "hv" else self._compute_spread(sub))
        return history

    def _compute_hypervolume(self, front: List[Tuple[float, float, float]], ref=(1.1, 1.1, 1.1)) -> float:
        front = np.array([f for f in front if all(f[i] <= ref[i] for i in range(3))])
        front = front[np.lexsort(np.rot90(front))] if len(front) > 0 else []
        return sum(np.prod(np.maximum(0, np.array(ref) - p)) for p in front)

    def _compute_spread(self, front: List[Tuple[float, float, float]]) -> float:
        front = np.array(front)
        if len(front) < 2:
            return 0.0
        dists = [np.linalg.norm(front[i] - front[i+1]) for i in range(len(front) - 1)]
        return np.std(dists)