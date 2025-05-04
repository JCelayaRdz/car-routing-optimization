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
        self.d_max, self.t_max, self.c_max = self.estimate_normalization_constants(self.population)

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

            # Distancia en metros
            d = edge.get('length', 0)
            total_d += d

            # Tiempo de viaje en minutos (ya precalculado por osmnx)
            t = edge.get('travel_time', 0) / 60
            total_t += t

            # Consumo de combustible (en litros), ya precalculado en tu dataset
            c = edge.get('fuel_consumption', 0)
            total_c += c

            # Penalización dura (LEZ)
            if edge.get('in_lez', False):
                total_Ph += penalty_hard

            # Penalización suave
            Ps = 0

            # Penalizar tipo de vía lenta
            road_type = edge.get('highway_clean', 'unknown')
            if road_type in ['residential', 'service', 'living_street']:
                Ps += 1

            # Penalizar calles de un solo carril
            lanes = edge.get('lanes', '1')
            try:
                lanes = int(str(lanes).split(';')[0])
                if lanes < 2:
                    Ps += 1
            except:
                Ps += 1

            # Penalizar velocidad baja (ej: zonas 30)
            speed = edge.get('speed_kph', 50)
            try:
                speed = float(speed[0]) if isinstance(speed, list) else float(speed)
                if speed < 40:
                    Ps += 1
            except:
                Ps += 1

            # Acumular penalización suave (cuadrada para castigar acumulación)
            total_Ps += Ps ** 2

        # Normalización
        d_norm = total_d / d_max
        t_norm = total_t / t_max
        c_norm = total_c / c_max

        # Función de evaluación final
        fitness = d_norm + t_norm + c_norm + total_Ph + total_Ps
        return fitness

    def estimate_normalization_constants(self, population):
        """
        Estima los valores máximos de distancia, tiempo y consumo
        a partir de una población de rutas.

        Args:
            population (list): Lista de rutas, donde cada ruta es una lista de edges (u, v, k)

        Returns:
            tuple: (d_max, t_max, c_max)
        """
        max_d = 1e-6  # evitar división entre cero
        max_t = 1e-6
        max_c = 1e-6

        for route in population:
            total_d = 0
            total_t = 0
            total_c = 0

            for u, v, k in route:
                edge = self.graph[u][v][k]

                # Distancia
                d = edge.get("length", 0)
                total_d += d

                # Tiempo (ya en segundos, lo pasamos a minutos)
                t = edge.get("travel_time", 0) / 60
                total_t += t

                # Consumo
                c = edge.get("fuel_consumption", 0)
                total_c += c

            max_d = max(max_d, total_d)
            max_t = max(max_t, total_t)
            max_c = max(max_c, total_c)

        return max_d, max_t, max_c

    def selection(self, k=3, p=0.8):
        """
        Selección por torneo estocástico.
        De k individuos al azar, elige al mejor con probabilidad p, y a otro con 1-p.

        Args:
            k (int): número de individuos en el torneo.
            p (float): probabilidad de seleccionar el mejor.

        Returns:
            Individuo seleccionado (una ruta).
        """
        # Seleccionar k rutas aleatorias
        selected = random.sample(self.population, k)

        # Ordenar por fitness (menor = mejor)
        selected.sort(key=lambda route: self.evaluate_route(route, self.d_max, self.t_max, self.c_max))

        # Con probabilidad p, elegimos el mejor
        if random.random() < p:
            return selected[0]
        else:
            # Con 1-p, elegimos aleatoriamente uno de los otros
            return random.choice(selected[1:])

    def crossover(self, parent1, parent2):
        """
        Order Crossover (OX1) para rutas (listas de nodos).
        Genera un hijo válido sin duplicados.

        Args:
            parent1 (list): Lista de nodos (ruta)
            parent2 (list): Lista de nodos (ruta)

        Returns:
            list: Ruta hija (lista de nodos)
        """
        size = len(parent1)

        # Elegimos dos puntos de corte aleatorios
        a, b = sorted(random.sample(range(1, size - 1), 2))  # sin incluir source ni target

        # Paso 1: Copiar el segmento central de parent1
        child = [None] * size
        child[a:b] = parent1[a:b]

        # Paso 2: Rellenar con el orden de parent2, saltando duplicados
        fill_nodes = [node for node in parent2 if node not in child[a:b]]
        idx = 0
        for i in list(range(0, a)) + list(range(b, size)):
            child[i] = fill_nodes[idx]
            idx += 1

        return child

    def mutate(self, route, max_attempts=10):
        """
        Mutación por intercambio (Swap Mutation).
        Intercambia dos nodos intermedios y repara si es necesario.

        Args:
            route (list): Ruta como lista de nodos.

        Returns:
            list: Nueva ruta mutada (o la original si no es válida tras varios intentos).
        """
        for _ in range(max_attempts):
            mutated = route.copy()

            if len(mutated) <= 3:
                return mutated  # nada que mutar

            # Elegimos dos posiciones internas al azar (excluye source y target)
            i, j = sorted(random.sample(range(1, len(mutated) - 1), 2))
            mutated[i], mutated[j] = mutated[j], mutated[i]

            try:
                # Validar reconstruyendo aristas
                self.build_edge_route(mutated)
                return mutated  # si no lanza error, es válida
            except:
                continue  # intenta otra vez

        return route  # si ninguna mutación es válida, devuelve original

    def evolve(self, generations=50, elitism=True):
        for _ in range(generations):
            new_population = []

            # Elitismo: conservar el mejor de la generación anterior
            if elitism:
                elite = min(self.population,
                            key=lambda route: self.evaluate_route(route, self.d_max, self.t_max, self.c_max))
                new_population.append(elite)

            # Generar el resto de la nueva población
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()

                # Crossover con probabilidad
                if random.random() < self.crossover_rate:
                    child = self.crossover(parent1, parent2)
                else:
                    child = parent1.copy()

                # Mutación con probabilidad
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)

                new_population.append(child)

            self.population = new_population

        # Devolver el mejor individuo encontrado
        return min(self.population, key=lambda route: self.evaluate_route(route, self.d_max, self.t_max, self.c_max))

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

    def random_valid_route(self, max_length=20, max_attempts=100):
        for _ in range(max_attempts):
            try:
                # Usa una búsqueda simple y aleatoria desde source hasta target
                path = [self.source]
                current = self.source
                visited = set()
                while current != self.target and len(path) < max_length:
                    neighbors = list(self.graph.successors(current))
                    neighbors = [n for n in neighbors if n not in visited]  # evita ciclos
                    if not neighbors:
                        break
                    next_node = random.choice(neighbors)
                    path.append(next_node)
                    visited.add(current)
                    current = next_node
                if current == self.target:
                    return path
            except:
                continue
        return None

