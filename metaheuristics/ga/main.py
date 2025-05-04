import os
import osmnx as ox
import matplotlib.pyplot as plt
from metaheuristics.ga_car_routing import GeneticAlgorithm
import networkx as nx
import random
import random
import numpy as np

def main():

    random.seed(42)
    np.random.seed(42)

    # 1. Cargar grafo preprocesado
    grafo_path = "../../data/madrid.graphml"
    assert os.path.exists(grafo_path), f"El archivo {grafo_path} no existe."

    print("Cargando grafo limpio desde archivo...")
    G = ox.load_graphml(grafo_path)

    # 2. Seleccionar nodos origen y destino con camino existente
    print("Seleccionando nodos origen y destino...")
    nodes = list(G.nodes)
    # Elegir nodos con camino corto (máximo 40 nodos de longitud)
    while True:
        source, target = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, source, target)
            if len(path) <= 20:
                break
        except nx.NetworkXNoPath:
            continue

    print(f"Origen: {source}, Destino: {target}")

    # 3. Inicializar el algoritmo genético
    print("Inicializando algoritmo genético...")

    ga = GeneticAlgorithm(
        graph=G,
        source=source,
        target=target,
        population_size=20,
        mutation_rate=0.3,
        crossover_rate=0.8,
        route_length=20
    )

    # 4. Ejecutar evolución
    print("Ejecutando algoritmo genético (30 generaciones)...")
    best_route = ga.evolve(generations=30)

    # 5. Mostrar resultado
    print("\nMejor ruta (lista de aristas):")
    for edge in best_route:
        print(edge)

    # 6. Calcular métricas de la mejor ruta
    print("\nMétricas de la mejor ruta:")
    fitness = ga.evaluate_route(best_route, ga.d_max, ga.t_max, ga.c_max)
    total_d = sum(G[u][v][k].get('length', 0) for u, v, k in best_route)
    total_t = sum(G[u][v][k].get('travel_time', 0) for u, v, k in best_route) / 60  # minutos
    total_c = sum(G[u][v][k].get('fuel_consumption', 0) for u, v, k in best_route)

    print(f"- Fitness total: {fitness:.2f}")
    print(f"- Distancia total: {total_d:.1f} m")
    print(f"- Tiempo estimado: {total_t:.1f} min")
    print(f"- Consumo estimado: {total_c:.2f} L")

    # 7. Visualizar
    print("\nMostrando ruta en el grafo...")
    if best_route:
        route_nodes = [edge[0] for edge in best_route] + [best_route[-1][1]]
        ox.plot_graph_route(G, route_nodes, route_linewidth=4, node_size=0)
    else:
        print("⚠ No se encontró una ruta válida.")


if __name__ == "__main__":
    main()
