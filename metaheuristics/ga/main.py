import os
import osmnx as ox
import networkx as nx
import random
import numpy as np
import pandas as pd
from metaheuristics.ga_car_routing import GeneticAlgorithm  # Asegúrate de que el path es correcto

def get_edge_value(edges_gdf, u, v, k, column):
    """
    Devuelve el valor del atributo 'column' de la arista (u, v, k) desde el GeoDataFrame.
    Si no se encuentra, devuelve 0.
    """
    match = edges_gdf[(edges_gdf['u'] == u) & (edges_gdf['v'] == v) & (edges_gdf['key'] == k)]
    if not match.empty:
        return match.iloc[0][column]
    return 0

def main():
    random.seed(42)
    np.random.seed(42)

    # 1. Cargar grafo y GeoDataFrame
    grafo_path = "../../data/madrid.graphml"
    edges_path = "../../data/edges_clean.json"
    assert os.path.exists(grafo_path), f"El archivo {grafo_path} no existe."
    assert os.path.exists(edges_path), f"El archivo {edges_path} no existe."

    print("Cargando grafo limpio desde archivo...")
    G = ox.load_graphml(grafo_path)

    print("Cargando GeoDataFrame de aristas con atributos...")
    edges_gdf = pd.read_json(edges_path)

    # 2. Seleccionar nodos válidos
    print("Seleccionando nodos origen y destino...")
    nodes = list(G.nodes)

    while True:
        source, target = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(G, source, target)
            if len(path) <= 30:
                break
        except nx.NetworkXNoPath:
            continue

    print(f"Origen: {source}, Destino: {target}")

    # 3. Inicializar algoritmo genético
    print("Inicializando algoritmo genético...")
    ga = GeneticAlgorithm(
        graph=G,
        edges_gdf=edges_gdf,
        source=source,
        target=target,
        population_size=30,
        mutation_rate=0.3,
        crossover_rate=0.8,
        route_length=30
    )

    # 4. Evolución
    print("Ejecutando algoritmo genético (30 generaciones)...")
    best_route = ga.evolve(generations=30)

    # 5. Mostrar ruta
    print("\nMejor ruta (lista de aristas):")
    for edge in best_route:
        print(edge)

    # 6. Métricas de ruta
    print("\nMétricas de la mejor ruta:")
    fitness = ga.evaluate_route(best_route)
    total_d = sum(get_edge_value(edges_gdf, u, v, k, 'length') for u, v, k in best_route)
    total_t = sum(get_edge_value(edges_gdf, u, v, k, 'travel_time') for u, v, k in best_route) / 60
    total_c = sum(get_edge_value(edges_gdf, u, v, k, 'fuel_consumption') for u, v, k in best_route)

    print(f"- Fitness total: {fitness:.2f}")
    print(f"- Distancia total: {total_d:.1f} m")
    print(f"- Tiempo estimado: {total_t:.1f} min")
    print(f"- Consumo estimado: {total_c:.4f} L")

    # 7. Visualización
    print("\nMostrando ruta en el grafo...")
    if best_route:
        route_nodes = [u for u, _, _ in best_route] + [best_route[-1][1]]
        ox.plot_graph_route(G, route_nodes, route_linewidth=4, node_size=0)
    else:
        print("No se encontró una ruta válida.")

if __name__ == "__main__":
    main()
