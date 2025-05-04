import osmnx as ox
from shapely.ops import unary_union

# Descargar grafo de la red vial de Madrid
graph = ox.graph_from_place("Madrid, Spain", network_type="drive")

# Añadir velocidades estimadas y tiempos de viaje
graph = ox.add_edge_speeds(graph)
graph = ox.add_edge_travel_times(graph)

#Guardar grafo original como archivo GraphML (para reutilizar después)
ox.save_graphml(graph, filepath="grafo_madrid.graphml")

# Convertir a GeoDataFrames
nodes, edges = ox.graph_to_gdfs(graph)

# Obtener zonas LEZ (Low Emission Zone)
lez_gdf = ox.features_from_place("Madrid, Spain", {"boundary": "low_emission_zone"})
lez_polygon = unary_union(lez_gdf.geometry)  # usar método actual recomendado

# Añadir columna booleana a los nodos: si están dentro de una zona LEZ
nodes["lez"] = nodes.geometry.within(lez_polygon)
edges["in_lez"] = edges.geometry.intersects(lez_polygon)

# Guardar GeoPackage con capas separadas
edges.to_file("grafo_madrid.gpkg", layer="edges", driver="GPKG")
nodes.to_file("grafo_madrid.gpkg", layer="nodes", driver="GPKG")

# También guardar como GeoJSON para usos rápidos o web
edges.to_file("edges_con_LEZ.geojson", driver="GeoJSON")
nodes.to_file("nodes.geojson", driver="GeoJSON")


# Mostrar datos de ejemplo (opcional, para debugging o visualización)
print(nodes.head())
print(edges.head())
print(edges["highway"].value_counts().sort_values(ascending=False))
print(edges[edges["highway"] == "motorway_link"].head())
print(edges[edges["osmid"].apply(lambda x: isinstance(x, list))][["osmid"]])
