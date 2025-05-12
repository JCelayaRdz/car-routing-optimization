import networkx as nx
import random
import pandas as pd
from typing import List, Tuple

class NSGAII:
    def __init__(self, graph, edges_gdf: pd.DataFrame, source, target, population_size, generations, route_length):
        self.graph = graph
        self.edges_gdf = edges_gdf
        self.source = source
        self.target = target
        self.population_size = population_size
        self.generations = generations
        self.route_length = route_length
        self.population = self.generate_initial_population()

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
            raise RuntimeError(f"Solo se generaron {len(population)} rutas válidas de {self.population_size}.")
        return population

    def build_edge_route(self, node_list: List[int]) -> List[Tuple[int, int, int]]:
        edge_route = []
        for i in range(len(node_list) - 1):
            u, v = node_list[i], node_list[i + 1]
            if not self.graph.has_edge(u, v):
                raise ValueError(f"No edge between node {u} and {v}")
            keys = list(self.graph[u][v].keys())
            k = keys[0]
            edge_route.append((u, v, k))
        return edge_route

    def evaluate_objectives(self, edge_route):
        total_d = total_t = total_c = 0
        for u, v, k in edge_route:
            edge = self.get_edge_data(u, v, k)
            total_d += edge.get('length', 0)
            total_t += edge.get('travel_time', 0) / 60
            total_c += edge.get('fuel_consumption', 0)
        return total_d, total_t, total_c

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

    def non_dominated_sort(self, population):
        fronts = []
        domination_count = {}
        dominated_solutions = {}
        self.non_dominated_rank = [None] * len(population)

        for i, p in enumerate(population):
            domination_count[i] = 0
            dominated_solutions[i] = []
            for j, q in enumerate(population):
                if i == j:
                    continue
                if self.dominates(self.evaluate_objectives(p), self.evaluate_objectives(q)):
                    dominated_solutions[i].append(j)
                elif self.dominates(self.evaluate_objectives(q), self.evaluate_objectives(p)):
                    domination_count[i] += 1
            if domination_count[i] == 0:
                if len(fronts) == 0:
                    fronts.append([])
                fronts[0].append(i)

        current_front = 0
        while current_front < len(fronts):
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front += 1

        return fronts

    def dominates(self, obj1, obj2):
        return all(x <= y for x, y in zip(obj1, obj2)) and any(x < y for x, y in zip(obj1, obj2))

    def crowding_distance(self, front, population):
        distances = [0] * len(front)
        num_objectives = 3  # Distance, time, fuel consumption
        for m in range(num_objectives):
            front.sort(key=lambda i: self.evaluate_objectives(population[i])[m])
            distances[0] = distances[-1] = float('inf')
            for k in range(1, len(front) - 1):
                distances[k] += (self.evaluate_objectives(population[front[k + 1]])[m] -
                                 self.evaluate_objectives(population[front[k - 1]])[m])
        return distances

    def evolve(self):
        for generation in range(self.generations):
            offspring = self.generate_offspring()
            combined_population = self.population + offspring
            fronts = self.non_dominated_sort(combined_population)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) > self.population_size:
                    distances = self.crowding_distance(front, combined_population)
                    sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                    new_population.extend([combined_population[i] for i, _ in sorted_front[:self.population_size - len(new_population)]])
                    break
                new_population.extend([combined_population[i] for i in front])
            self.population = new_population

    def generate_offspring(self):
        offspring = []
        while len(offspring) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            if random.random() < 0.1:  # Mutation probability
                child = self.mutate(child)
            offspring.append(child)
        return offspring

    def tournament_selection(self):
        k = 2
        selected = random.sample(range(len(self.population)), k)
        selected.sort(key=lambda i: (self.non_dominated_rank[i], -self.crowding_distance[i]))
        return self.population[selected[0]]

    def crossover(self, parent1, parent2):
        """
        Realiza un cruce entre dos rutas (padres) para generar un hijo.
        Combina nodos de ambos padres asegurando que el hijo sea válido.
        """
        current = self.source
        visited = set([current])
        child = [current]

        while current != self.target:
            # Obtener el siguiente nodo de cada padre
            next1 = self._get_next_in_parent(current, parent1, visited)
            next2 = self._get_next_in_parent(current, parent2, visited)

            # Seleccionar el siguiente nodo basado en la distancia
            candidates = []
            if next1 and self.graph.has_edge(current, next1):
                candidates.append((next1, self.get_edge_data(current, next1, list(self.graph[current][next1].keys())[0]).get('length', float('inf'))))
            if next2 and self.graph.has_edge(current, next2):
                candidates.append((next2, self.get_edge_data(current, next2, list(self.graph[current][next2].keys())[0]).get('length', float('inf'))))

            if not candidates:
                break

            # Seleccionar el nodo con la menor distancia
            next_node = min(candidates, key=lambda x: x[1])[0]
            child.append(next_node)
            visited.add(next_node)
            current = next_node

        # Si no se llega al destino, devolver uno de los padres
        return child if current == self.target else parent1.copy()

    def _get_next_in_parent(self, current_node, parent, visited):
        """
        Obtiene el siguiente nodo en la ruta del padre que no haya sido visitado.
        """
        try:
            idx = parent.index(current_node)
            for next_node in parent[idx + 1:]:
                if next_node not in visited:
                    return next_node
        except ValueError:
            pass
        return None

    def mutate(self, route, max_attempts=10):
        """
        Realiza una mutación en una ruta intercambiando dos nodos aleatorios.
        """
        for _ in range(max_attempts):
            mutated = route.copy()
            if len(mutated) <= 3:  # Si la ruta es muy corta, no mutar
                return mutated
            # Seleccionar dos índices aleatorios para intercambiar
            i, j = sorted(random.sample(range(1, len(mutated) - 1), 2))
            mutated[i], mutated[j] = mutated[j], mutated[i]
            try:
                # Validar que la ruta mutada sea válida
                self.build_edge_route(mutated)
                return mutated
            except ValueError:
                continue
        return route