import osmnx as ox
import pandas as pd
import networkx as nx
from main import NSGAII  # Asegúrate de que el path es correcto

def main():
    # 1. Cargar el grafo y los datos de las aristas
    grafo_path = "../../data/madrid.graphml"
    edges_path = "../../data/edges_clean.json"

    assert os.path.exists(grafo_path), f"El archivo {grafo_path} no existe."
    assert os.path.exists(edges_path), f"El archivo {edges_path} no existe."

    print("Cargando grafo desde archivo...")
    graph = ox.load_graphml(grafo_path)

    print("Cargando datos de las aristas...")
    edges_gdf = pd.read_json(edges_path)

    # 2. Seleccionar nodos de origen y destino
    print("Seleccionando nodos de origen y destino...")
    nodes = list(graph.nodes)
    source, target = None, None

    while True:
        source, target = random.sample(nodes, 2)
        try:
            path = nx.shortest_path(graph, source, target)
            if len(path) <= 30:  # Asegúrate de que la ruta no sea demasiado larga
                break
        except nx.NetworkXNoPath:
            continue

    print(f"Nodo origen: {source}, Nodo destino: {target}")

    # 3. Configurar y ejecutar el algoritmo NSGA-II
    print("Inicializando NSGA-II...")
    nsga2 = NSGAII(
        graph=graph,
        edges_gdf=edges_gdf,
        source=source,
        target=target,
        population_size=20,
        generations=10,
        route_length=30
    )

    print("Ejecutando NSGA-II...")
    nsga2.evolve()

    # 4. Mostrar resultados
    print("\nSoluciones finales:")
    for route in nsga2.population:
        objectives = nsga2.evaluate_objectives(route)
        print(f"Ruta: {route}")
        print(f"Objetivos (Distancia, Tiempo, Consumo): {objectives}")

if __name__ == "__main__":
    main()