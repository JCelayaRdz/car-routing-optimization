import networkx as nx
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from shapely.ops import unary_union
import random


'''
Función que toma una lista de nodos [A, B, C, D] y la
transforma en una lista de aristas: 
[(A, B, key), (B, C, key), (C, D, key)].
'''

def build_edge_route(graph: nx.MultiDiGraph, node_list: list) -> list:
    """
    Given a list of node IDs, return the corresponding list of edges (u, v, key)
    if the path is valid and connected. Raises ValueError otherwise.

    Parameters:
        graph (nx.MultiDiGraph): Road network graph.
        node_list (list): List of node IDs forming a path.

    Returns:
        List of (u, v, key) tuples representing the edges in the route.
    """
    edge_route = []

    for i in range(len(node_list) - 1):
        u = node_list[i]
        v = node_list[i + 1]

        if not graph.has_edge(u, v):
            raise ValueError(f"No edge between node {u} and {v}, path is not valid.")

        edge_data = graph.get_edge_data(u, v)
        if not edge_data:
            raise ValueError(f"Missing edge data between {u} and {v}")

        # There may be multiple edges (MultiGraph), so we take the first one (key=0)
        first_key = list(edge_data.keys())[0]
        edge_route.append((u, v, first_key))

    return edge_route

'''
Funciones de creación de rutas
'''

def shortest_path(graph, source, target, weight="length"):
    """
    Calcula la ruta más corta entre dos nodos según un atributo (e.g., 'length', 'travel_time').
    """
    return nx.shortest_path(graph, source=source, target=target, weight=weight)



def random_valid_route(graph, length=10, max_attempts=100):
    """
    Genera una ruta aleatoria válida con una secuencia de nodos conectados.

    Parameters:
        graph: grafo dirigido (nx.MultiDiGraph)
        length: longitud deseada de la ruta (en número de nodos)
        max_attempts: número máximo de intentos antes de rendirse

    Returns:
        Una lista de nodos representando la ruta, o None si no se encuentra.
    """
    nodes = list(graph.nodes)

    for _ in range(max_attempts):
        path = [random.choice(nodes)]
        valid = True

        for _ in range(length - 1):
            neighbors = list(graph.successors(path[-1]))
            if not neighbors:
                valid = False
                break
            next_node = random.choice(neighbors)
            path.append(next_node)

        if valid and len(path) == length:
            return path

    return None  # No se encontró una ruta válida tras max_attempts



def shortest_path_avoid_lez(graph, source, target, penalty=1000):
    """
    Calcula ruta penalizando fuertemente las aristas dentro de zonas LEZ.
    """
    G_copy = graph.copy()
    for u, v, k, data in G_copy.edges(keys=True, data=True):
        if data.get("in_lez", False):
            if "travel_time" in data:
                data["travel_time"] += penalty
            else:
                data["length"] += penalty
    return nx.shortest_path(G_copy, source=source, target=target, weight="travel_time")


class GeneticAlgorithm:
    def __init__(self, graph, source, target, population_size, mutation_rate, crossover_rate):
        ...

    def generate_initial_population(self):
        ...

    def evaluate_route(self, graph, edge_route,
                       d_max=1, t_max=1, c_max=1,
                       penalty_hard=1000, penalty_soft_base=10) -> float:
        """
        Evaluates a route composed of a list of edges (u, v, key), returning a fitness value according to the function:
        fitness = d(r)/d_max + t(r)/t_max + c(r)/c_max + Ph(r) + Ps(r)

        Parameters:
        - graph: a MultiDiGraph representing the road network.
        - edge_route: list of tuples (u, v, key) that form a complete route.
        - d_max, t_max, c_max: maximum values for distance, time, and consumption, used for normalization.
        - penalty_hard: penalty value applied if the route enters a low-emission zone (LEZ).
        - penalty_soft_base: base penalty for using less preferred roads (residential, signals, few lanes).

        Returns:
        - A float representing the total fitness value of the route (lower is better).
        """

        total_d = 0
        total_t = 0
        total_c = 0
        total_Ph = 0
        total_Ps = 0

        for u, v, k in edge_route:
            edge = graph[u][v][k]
            d = edge.get('length', 0)
            speed = edge.get('maxspeed', 50)
            if isinstance(speed, list):
                speed = float(speed[0])
            speed = float(speed)

            t = d / (speed * 1000 / 60)  # minutos

            # Consumo estimado
            road_type = edge.get('highway', 'unknown')
            if isinstance(road_type, list):
                road_type = road_type[0]
            base_c = d * 0.00005
            if road_type in ['residential', 'service']:
                base_c *= 1.2
            elif road_type in ['motorway', 'primary']:
                base_c *= 0.9

            # Penalización dura (LEZ)
            in_LEZ = edge.get('in_lez', False)
            Ph = penalty_hard if in_LEZ else 0

            # Penalización suave
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

        # Normalización
        d_norm = total_d / d_max
        t_norm = total_t / t_max
        c_norm = total_c / c_max

        return d_norm + t_norm + c_norm + total_Ph + total_Ps

    def selection(self):
        ...

    def crossover(self, parent1, parent2):
        ...

    def mutate(self, route):
        ...

    def evolve(self):
        ...
