import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import networkx as nx

class ACOGraphOptimizer:
    def __init__(
        self, 
        graph_edges_path: Path, 
        graph_nodes_path: Path, 
        origin: int, 
        destination: int, 
        vehicle_allowed_in_lez: bool,
        n_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.5,
        max_no_improve: int = 10,
        penalty_signal: float = 0.005,
        penalty_lanes_factor: float = 1.2
    ):
        self.origin = origin
        self.destination = destination
        self.vehicle_allowed_in_lez = vehicle_allowed_in_lez

        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_ants = n_ants
        self.max_no_improve = max_no_improve
        self.penalty_signal = penalty_signal
        self.penalty_lanes_factor = penalty_lanes_factor

        self.graph = self._load_graph(graph_edges_path, graph_nodes_path)
        self.norm_distance = max(nx.get_edge_attributes(self.graph, 'length').values())
        self.norm_time = max(nx.get_edge_attributes(self.graph, 'travel_time').values())
        self.norm_fuel = max(nx.get_edge_attributes(self.graph, 'fuel_consumption').values())
        
        self.pheromone = defaultdict(lambda: 1.0)

    def _load_graph(self, edges_path: Path, nodes_path: Path) -> nx.MultiDiGraph:
        with open(edges_path) as f:
            edges = json.load(f)
        with open(nodes_path) as f:
            nodes = json.load(f)

        G = nx.MultiDiGraph()
        for node in nodes:
            G.add_node(int(node["osmid"]), **node)
        for edge in edges:
            G.add_edge(edge["u"], edge["v"], key=edge["key"], **edge)
        return G

    def _evaluate_route(self, route: List[Tuple[int, int, int]]) -> float:
        total_distance = 0.0
        total_time = 0.0
        total_fuel = 0.0

        for u, v, k in route:
            edge = self.graph[u][v][k]

            if edge.get("oneway", False) and (u, v) != (edge["u"], edge["v"]):
                return float('inf')
            if edge.get("lez", False) and not self.vehicle_allowed_in_lez:
                return float('inf')

            distance = edge["length"]
            time = edge["travel_time"]
            fuel = edge["fuel_consumption"]

            u_highway = self.graph.nodes[u].get("highway", "")
            v_highway = self.graph.nodes[v].get("highway", "")
            if u_highway in ["traffic_signals", "stop"] or v_highway in ["traffic_signals", "stop"]:
                fuel += 0.005

            try:
                if isinstance(edge.get("lanes"), list):
                    lanes = max([int(l) for l in edge["lanes"]])
                else:
                    lanes = int(edge.get("lanes", 1))
            except:
                lanes = 1

            if lanes == 1:
                fuel += fuel * 0.2

            total_distance += distance
            total_time += time
            total_fuel += fuel

        f1 = total_distance / self.norm_distance
        f2 = total_time / self.norm_time
        f3 = total_fuel / self.norm_fuel

        # Scalarization
        return 0.33 * f1 + 0.33 * f2 + 0.34 * f3

    def _construct_solution(self) -> List[Tuple[int, int, int]]:
        current = self.origin
        visited_edges = set()
        route = []
        attempts = 0

        while current != self.destination and attempts < 1000:
            neighbors = list(self.graph.out_edges(current, keys=True, data=True))
            candidates = []
            probs = []
            for u, v, k, data in neighbors:
                if (u, v, k) in visited_edges:
                    continue
                if data.get("lez", False) and not self.vehicle_allowed_in_lez:
                    continue
                pheromone = self.pheromone[(u, v, k)] ** self.alpha
                heuristic = 1.0 / (
                    1.0 + data["length"] / self.norm_distance +
                    data["travel_time"] / self.norm_time +
                    data["fuel_consumption"] / self.norm_fuel
                ) ** self.beta
                candidates.append((u, v, k))
                probs.append(pheromone * heuristic)

            if not candidates:
                break

            probs = np.array(probs)
            probs = probs / probs.sum()
            idx = np.random.choice(len(candidates), p=probs)
            selected = candidates[idx]
            route.append(selected)
            visited_edges.add(selected)
            current = selected[1]
            attempts += 1

        if current != self.destination:
            return []

        return route

    def optimize(self):
        no_improve = 0
        best_fitness = float('inf')
        best_route = None

        while no_improve < self.max_no_improve:
            for _ in range(self.n_ants):
                route = self._construct_solution()
                if not route:
                    continue
                fitness = self._evaluate_route(route)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_route = route
                    no_improve = 0
                else:
                    no_improve += 1
                for edge in route:
                    self.pheromone[edge] += 1.0 / fitness

            self.pheromone = {k: v * (1 - self.rho) for k, v in self.pheromone.items()}

        return [(best_route, (best_fitness, best_fitness, best_fitness))]  # mimic archive format