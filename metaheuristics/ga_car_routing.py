import networkx as nx
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import random

class GeneticAlgorithm:
    def __init__(self, graph, source, target, population_size, mutation_rate, crossover_rate, route_length):
        self.graph = graph
        self.source = source
        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.route_length = route_length
        self.nodes = list(graph.nodes)
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        population = []
        while len(population) < self.population_size:
            route = self.random_valid_route(self.route_length)
            if route:
                try:
                    edge_route = self.build_edge_route(route)
                    population.append(edge_route)
                except ValueError:
                    continue
        return population

    def evaluate_route(self, edge_route, d_max=1, t_max=1, c_max=1,
                       penalty_hard=1000, penalty_soft_base=10) -> float:
        total_d = 0
        total_t = 0
        total_c = 0
        total_Ph = 0
        total_Ps = 0

        for u, v, k in edge_route:
            edge = self.graph[u][v][k]
            d = edge.get('length', 0)
            speed = edge.get('maxspeed', 50)
            if isinstance(speed, list):
                speed = float(speed[0])
            speed = float(speed)

            t = d / (speed * 1000 / 60)

            road_type = edge.get('highway', 'unknown')
            if isinstance(road_type, list):
                road_type = road_type[0]
            base_c = d * 0.00005
            if road_type in ['residential', 'service']:
                base_c *= 1.2
            elif road_type in ['motorway', 'primary']:
                base_c *= 0.9

            in_LEZ = edge.get('in_lez', False)
            Ph = penalty_hard if in_LEZ else 0

            Ps = 0
            if road_type in ['residential', 'service']:
                Ps += 1
            if edge.get('traffic_signals'):
                Ps += 1
            lanes = edge.get('lanes', 1)
            try:
                lanes = int(lanes)
                if lanes < 2:
                    Ps += 1
            except:
                Ps += 1

            total_d += d
            total_t += t
            total_c += base_c
            total_Ph += Ph
            total_Ps += Ps ** 2

        d_norm = total_d / d_max
        t_norm = total_t / t_max
        c_norm = total_c / c_max

        return d_norm + t_norm + c_norm + total_Ph + total_Ps

    def selection(self, k=3):
        selected = random.sample(self.population, k)
        selected.sort(key=lambda route: self.evaluate_route(route))
        return selected[0]

    def crossover(self, parent1, parent2):
        nodes1 = [u for u, _, _ in parent1]
        nodes2 = [u for u, _, _ in parent2]
        common = list(set(nodes1) & set(nodes2))
        if not common:
            return parent1
        crossover_point = random.choice(common)
        idx1 = nodes1.index(crossover_point)
        idx2 = nodes2.index(crossover_point)
        new_nodes = nodes1[:idx1] + nodes2[idx2:]
        try:
            return self.build_edge_route(new_nodes)
        except ValueError:
            return parent1

    def mutate(self, route):
        if random.random() > self.mutation_rate:
            return route
        idx = random.randint(1, len(route) - 2)
        sub_route = self.random_valid_route(len(route) - idx)
        if sub_route:
            try:
                new_nodes = [route[0][0]] + [v for _, v, _ in route[:idx]] + sub_route
                return self.build_edge_route(new_nodes)
            except ValueError:
                return route
        return route

    def evolve(self, generations=50):
        for _ in range(generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
        return min(self.population, key=lambda route: self.evaluate_route(route))

    def build_edge_route(self, node_list: list) -> list:
        edge_route = []
        for i in range(len(node_list) - 1):
            u = node_list[i]
            v = node_list[i + 1]
            if not self.graph.has_edge(u, v):
                raise ValueError(f"No edge between node {u} and {v}, path is not valid.")
            edge_data = self.graph.get_edge_data(u, v)
            if not edge_data:
                raise ValueError(f"Missing edge data between {u} and {v}")
            first_key = list(edge_data.keys())[0]
            edge_route.append((u, v, first_key))
        return edge_route

    def shortest_path(self, weight="length"):
        return nx.shortest_path(self.graph, source=self.source, target=self.target, weight=weight)

    def random_valid_route(self, length=10, max_attempts=100):
        for _ in range(max_attempts):
            path = [random.choice(self.nodes)]
            valid = True
            for _ in range(length - 1):
                neighbors = list(self.graph.successors(path[-1]))
                if not neighbors:
                    valid = False
                    break
                next_node = random.choice(neighbors)
                path.append(next_node)
            if valid and len(path) == length:
                return path
        return None

    def shortest_path_avoid_lez(self, penalty=1000):
        """
        Calcula ruta entre self.source y self.target penalizando fuertemente
        las aristas que cruzan zonas LEZ.
        """
        G_copy = self.graph.copy()
        for u, v, k, data in G_copy.edges(keys=True, data=True):
            if data.get("in_lez", False):
                if "travel_time" in data:
                    data["travel_time"] += penalty
                else:
                    data["length"] += penalty
        return nx.shortest_path(G_copy, source=self.source, target=self.target, weight="travel_time")

