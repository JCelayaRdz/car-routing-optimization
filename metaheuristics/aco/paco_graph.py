import json
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import networkx as nx
import osmnx as ox
import os
import folium
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

class PACOGraphOptimizer:
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
        self.archive = []  # Pareto archive
        self.best_fitness_history = []
        self.n_evaluations = 0

        self.history_hv = []
        self.history_spread = []


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
        total_distance = 0.0
        total_time = 0.0
        total_fuel = 0.0

        for u, v, k in route:
            edge = self.graph[u][v][k]

            if edge.get("oneway", False) and (u, v) != (edge["u"], edge["v"]):
                return float('inf'), float('inf'), float('inf')

            if edge.get("lez", False) and not self.vehicle_allowed_in_lez:
                return float('inf'), float('inf'), float('inf')

            distance = edge["length"]
            time = edge["travel_time"]
            fuel = edge["fuel_consumption"]

            u_highway = self.graph.nodes[u].get("highway", "")
            v_highway = self.graph.nodes[v].get("highway", "")
            if u_highway in ["traffic_signals", "stop"] or v_highway in ["traffic_signals", "stop"]:
                fuel += self.penalty_signal   
            
            try:
                if isinstance(edge.get("lanes"), list):
                    lanes = max([int(l) for l in edge["lanes"]])
                else:
                    lanes = int(edge.get("lanes", 1))
            except:
                    lanes = 1

            if lanes == 1:
                fuel += self.penalty_lanes_factor

            total_distance += distance
            total_time += time
            total_fuel += fuel

        f1 = total_distance / self.norm_distance
        f2 = total_time / self.norm_time
        f3 = total_fuel / self.norm_fuel

        return round(f1, 6), round(f2, 6), round(f3, 6)

    def _construct_solution(self) -> List[Tuple[int, int, int]]:
        max_total_attempts = 3
        for _ in range(max_total_attempts):
            current = self.origin
            route = []
            attempts = 0

            dest_x = self.graph.nodes[self.destination]["x"]
            dest_y = self.graph.nodes[self.destination]["y"]

            while current != self.destination and attempts < 1000:
                if current not in self.graph:
                    break

                neighbors = list(self.graph.out_edges(current, keys=True, data=True))
                candidates = []
                probs = []

                for u, v, k, data in neighbors:
                    if data.get("in_lez", False) and not self.vehicle_allowed_in_lez:
                        continue

                    v_x = self.graph.nodes[v]["x"]
                    v_y = self.graph.nodes[v]["y"]
                    euclidean_to_dest = ((dest_x - v_x) ** 2 + (dest_y - v_y) ** 2) ** 0.5

                    pheromone = self.pheromone[(u, v, k)] ** self.alpha
                    heuristic = 1.0 / (
                        1.0 +
                        data["length"] / self.norm_distance +
                        data["travel_time"] / self.norm_time +
                        data["fuel_consumption"] / self.norm_fuel +
                        (euclidean_to_dest / 1000.0)
                    ) ** self.beta

                    candidates.append((u, v, k))
                    probs.append(pheromone * heuristic)

                if not candidates:
                    break

                probs = np.array(probs, dtype=np.float64)
                probs_sum = probs.sum()

                if probs_sum == 0 or np.isnan(probs_sum):
                    idx = np.random.choice(len(candidates))
                else:
                    probs = probs / probs_sum
                    probs = 0.95 * probs + 0.05 / len(probs)
                    probs = probs / probs.sum()
                    idx = np.random.choice(len(candidates), p=probs)

                selected = candidates[idx]
                route.append(selected)
                current = selected[1]

                attempts += 1

            if current == self.destination:
                return route

        return []
    
    def _update_archive(self, route: List[Tuple[int, int, int]], fitness: Tuple[float, float, float]):
        for r, f in self.archive:
            if route == r:
                return

        non_dominated = []
        dominated_by_new = []

        for r, f in self.archive:
            if self._dominates(fitness, f):
                dominated_by_new.append((r, f))
            elif self._dominates(f, fitness):
                return
            elif f == fitness:
                if route == r:
                    return
                else:
                    continue
            else:
                non_dominated.append((r, f))

        self.archive = [item for item in non_dominated if item not in dominated_by_new]
        self.archive.append((route, fitness))


    def _dominates(self, f1, f2):
        return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

    def optimize(self):
        no_improve = 0
        best_size = 0

        if not nx.has_path(self.graph, self.origin, self.destination):
            return []

        last_archive_size = 0

        while no_improve < self.max_no_improve and len(self.archive) < 25:
            fail_count = 0
            for i in range(self.n_ants):
                route = self._construct_solution()
                if not route:
                    fail_count += 1
                    if self.history_hv:
                        self.history_hv.append(self.history_hv[-1])
                        self.history_spread.append(self.history_spread[-1])
                    else:
                        self.history_hv.append(0.0)
                        self.history_spread.append(0.0)
                    continue
                fitness = self._evaluate_route(route)
                self.n_evaluations += 1
                old_archive_len = len(self.archive)
                self._update_archive(route, fitness)
                for edge in route:
                    self.pheromone[edge] += 1.0 / sum(fitness)

                objectives = [f for _, f in self.archive]
                if len(self.archive) > old_archive_len:
                    hv = self._compute_hypervolume(objectives)
                    spread = self._compute_spread(objectives)
                    self.history_hv.append(hv)
                    self.history_spread.append(spread)
                else:
                    if self.history_hv:
                        self.history_hv.append(self.history_hv[-1])
                        self.history_spread.append(self.history_spread[-1])
                    else:
                        self.history_hv.append(0.0)
                        self.history_spread.append(0.0)

            for k in list(self.pheromone.keys()):
                self.pheromone[k] *= (1 - self.rho)
            current_size = len(self.archive)


            if current_size > best_size:
                best_size = current_size
                no_improve = 0
            else:
                no_improve += 1

        return self.archive

    
    def _compute_hypervolume(self, front, reference_point=(1.1, 1.1, 1.1)):
        if not front:
            return 0.0
        front = np.array(front)
        front = front[np.all(front <= reference_point, axis=1)]
        if len(front) == 0:
            return 0.0
        hv = 0.0
        sorted_front = front[np.argsort(front[:,0])]
        last = np.array([1.1, 1.1, 1.1])
        for f in sorted_front:
            d = np.abs(last[0] - f[0])
            t = np.abs(last[1] - f[1])
            fc = np.abs(last[2] - f[2])
            hv += d * t * fc
            last = f
        return hv
    
    def _compute_spread(self, front: List[Tuple[float, float, float]]) -> float:
        front = np.array(front)
        if len(front) < 2:
            return 0.0
        dists = [np.linalg.norm(front[i] - front[i + 1]) for i in range(len(front) - 1)]
        return np.std(dists)