import osmnx as ox

import matplotlib.pyplot as plt
from metaheuristics.ga_car_routing import GeneticAlgorithm
import networkx as nx
import random

def main():
    # 1. Cargar grafo preprocesado
    print("Cargando grafo limpio desde archivo...")
    G = ox.load_graphml("../../data/madrid.graphml")

    # 2. Seleccionar nodos origen y destino

    nodes = list(G.nodes)
    while True:
        source, target = random.sample(nodes, 2)
        if nx.has_path(G, source, target):
            break

    print(f"Origen: {source}, Destino: {target}")

    # 3. Inicializar el algoritmo genético
    ga = GeneticAlgorithm(
        graph=G,
        source=source,
        target=target,
        population_size=20,
        mutation_rate=0.3,
        crossover_rate=0.8,
        route_length=10
    )

    # 4. Ejecutar evolución
    print("Ejecutando algoritmo genético...")
    best_route = ga.evolve(generations=30)

    # 5. Mostrar resultado
    print("Mejor ruta (como lista de aristas):")
    for edge in best_route:
        print(edge)

    # 6. Visualizar
    print("Mostrando ruta...")
    if best_route:
        route_nodes = [edge[0] for edge in best_route] + [best_route[-1][1]]
        ox.plot_graph_route(G, route_nodes, route_linewidth=4, node_size=0)
    else:
        print("⚠ No se encontró una ruta válida.")


if __name__ == "__main__":
    main()
