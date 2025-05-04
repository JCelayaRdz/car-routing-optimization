import networkx as nx
import random
import pandas as pd

highway_priority = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "unclassified"
]

class GeneticAlgorithm:
    def __init__(self, graph, edges_gdf: pd.DataFrame, source, target, population_size, mutation_rate, crossover_rate, route_length):
        self.graph = graph
        self.edges_gdf = edges_gdf
        self.source = source
        self.target = target
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.route_length = route_length
        self.nodes = list(graph.nodes)
        self.population = self.generate_initial_population()
        print(f"Generadas {len(self.population)} rutas válidas para población inicial.")
        self.d_max, self.t_max, self.c_max = self.estimate_normalization_constants(self.population)

    def get_edge_data(self, u, v, k):
        match = self.edges_gdf[
            (self.edges_gdf['u'] == u) &
            (self.edges_gdf['v'] == v) &
            (self.edges_gdf['key'] == k)
        ]
        if not match.empty:
            return match.iloc[0].to_dict()
        else:
            return {}

    def generate_initial_population(self):
        population = []
        try:
            path_generator = nx.all_simple_paths(
                self.graph,
                self.source,
                self.target,
                cutoff=self.route_length
            )
            for path in path_generator:
                try:
                    edge_route = self.build_edge_route(path)
                    population.append(edge_route)
                except ValueError:
                    continue
                if len(population) >= self.population_size:
                    break
        except nx.NetworkXNoPath:
            print("No hay ningún camino entre source y target.")
        if len(population) < self.population_size:
            raise RuntimeError(f" Solo se generaron {len(population)} rutas válidas de {self.population_size}.")
        return population

    def evaluate_route(self, edge_route, penalty_hard=1000, penalty_soft_base=10):
        total_d = total_t = total_c = total_Ph = total_Ps = 0

        for u, v, k in edge_route:
            edge = self.get_edge_data(u, v, k)

            d = edge.get('length', 0)
            total_d += d

            t = edge.get('travel_time', 0) / 60
            total_t += t

            c = edge.get('fuel_consumption', 0)
            total_c += c

            if edge.get('in_lez', False):
                total_Ph += penalty_hard

            Ps = 0
            road_type = edge.get('highway_clean', 'unknown')
            if road_type in highway_priority:
                priority_index = highway_priority.index(road_type)
                Ps += priority_index / len(highway_priority)
            else:
                Ps += 1

            lanes = edge.get('lanes', '1')
            try:
                lanes = int(str(lanes).split(';')[0])
                if lanes < 2:
                    Ps += 1
            except:
                Ps += 1

            speed = edge.get('speed_kph', 50)
            try:
                speed = float(speed[0]) if isinstance(speed, list) else float(speed)
                if speed < 40:
                    Ps += 1
            except:
                Ps += 1

            total_Ps += Ps ** 2

        d_norm = total_d / self.d_max
        t_norm = total_t / self.t_max
        c_norm = total_c / self.c_max

        return d_norm + t_norm + c_norm + total_Ph + total_Ps

    def estimate_normalization_constants(self, population):
        max_d = max_t = max_c = 1e-6
        for route in population:
            total_d = total_t = total_c = 0
            for u, v, k in route:
                edge = self.get_edge_data(u, v, k)
                total_d += edge.get("length", 0)
                total_t += edge.get("travel_time", 0) / 60
                total_c += edge.get("fuel_consumption", 0)
            max_d = max(max_d, total_d)
            max_t = max(max_t, total_t)
            max_c = max(max_c, total_c)
        return max_d, max_t, max_c

    def selection(self, k=3, p=0.8):
        selected = random.sample(self.population, k)
        selected.sort(key=lambda route: self.evaluate_route(route))
        if random.random() < p:
            return selected[0]
        else:
            return random.choice(selected[1:])

    def crossover(self, parent1, parent2):
        current = self.source
        visited = set([current])
        child = [current]

        while current != self.target:
            next1 = self._get_next_in_parent(current, parent1, visited)
            next2 = self._get_next_in_parent(current, parent2, visited)

            candidates = []
            if next1 and self.graph.has_edge(current, next1):
                candidates.append((next1, self.get_edge_data(current, next1, list(self.graph[current][next1].keys())[0]).get('length', float('inf'))))
            if next2 and self.graph.has_edge(current, next2):
                candidates.append((next2, self.get_edge_data(current, next2, list(self.graph[current][next2].keys())[0]).get('length', float('inf'))))

            if not candidates:
                break

            next_node = min(candidates, key=lambda x: x[1])[0]
            child.append(next_node)
            visited.add(next_node)
            current = next_node

        return child if current == self.target else parent1.copy()

    def _get_next_in_parent(self, current_node, parent, visited):
        try:
            idx = parent.index(current_node)
            for next_node in parent[idx + 1:]:
                if next_node not in visited:
                    return next_node
        except ValueError:
            pass
        return None

    def mutate(self, route, max_attempts=10):
        for _ in range(max_attempts):
            mutated = route.copy()
            if len(mutated) <= 3:
                return mutated
            i, j = sorted(random.sample(range(1, len(mutated) - 1), 2))
            mutated[i], mutated[j] = mutated[j], mutated[i]
            try:
                self.build_edge_route(mutated)
                return mutated
            except:
                continue
        return route

    def evolve(self, generations=50, elitism=True):
        for _ in range(generations):
            new_population = []
            if elitism:
                elite = min(self.population, key=lambda route: self.evaluate_route(route))
                new_population.append(elite)
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                if random.random() < self.crossover_rate:
                    parent1_nodes = [u for u, v, k in parent1] + [parent1[-1][1]]
                    parent2_nodes = [u for u, v, k in parent2] + [parent2[-1][1]]
                    child_nodes = self.crossover(parent1_nodes, parent2_nodes)
                else:
                    child_nodes = [u for u, v, k in parent1] + [parent1[-1][1]]
                try:
                    child = self.build_edge_route(child_nodes)
                except ValueError:
                    continue
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                new_population.append(child)
            self.population = new_population
        return min(self.population, key=lambda route: self.evaluate_route(route))

    def build_edge_route(self, node_list: list) -> list:
        edge_route = []
        for i in range(len(node_list) - 1):
            u, v = node_list[i], node_list[i + 1]
            if not self.graph.has_edge(u, v):
                raise ValueError(f"No edge between node {u} and {v}")
            keys = list(self.graph[u][v].keys())
            k = keys[0]
            edge_route.append((u, v, k))
        return edge_route
