import osmnx as ox
import pandas as pd

# ----------------------------- Configuración -----------------------------

highway_priority = [
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "living_street",
    "unclassified"
]
valid_types = set(highway_priority)

# ----------------------------- Funciones auxiliares -----------------------------

def clean_highway(h):
    if isinstance(h, list):
        for hw in highway_priority:
            if hw in h:
                return hw
        return h[0]
    return h

def calc_avg_fuel_consumption(highway, speed):
    if highway == "motorway":
        return 6.5 / 100
    elif highway == "motorway_link":
        return 7.0 / 100
    elif highway == "trunk":
        return 6.8 / 100
    elif highway == "trunk_link":
        return 7.3 / 100
    elif highway in ["primary", "primary_link"]:
        return 7.5 / 100
    elif highway in ["secondary", "secondary_link"]:
        return 7.8 / 100
    elif highway in ["tertiary", "tertiary_link"]:
        return 8.2 / 100
    elif highway in ["residential", "living_street"]:
        return 9.0 / 100
    elif highway == "unclassified":
        return 8.0 / 100
    else:
        return 8.5 / 100
    
def normalize_name(x):
    if isinstance(x, list):
        return ", ".join(map(str, x)) if x else None
    return x if pd.notnull(x) else None



# ----------------------------- Ejecución -----------------------------

if __name__ == "__main__":
    # Descargar grafo de Madrid
    graph = ox.graph_from_place("Madrid, Spain", network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    # Guardar grafo original por si acaso
    ox.save_graphml(graph, filepath="madrid.graphml")

    # Convertir a GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(graph)
    edges = edges.reset_index()  # Poner u, v, key como columnas

    # Obtener geometría de la zona de bajas emisiones (ZBE)
    lez_geodataframe = ox.features_from_place("Madrid, Spain", {"boundary": "low_emission_zone"})
    lez_polygon = lez_geodataframe.geometry.union_all()

    # Añadir etiqueta 'lez' a los nodos
    nodes["lez"] = nodes.geometry.within(lez_polygon)
    edges["in_lez"] = edges.geometry.intersects(lez_polygon)

    # Asegurar que osmid está como columna en nodos
    if "osmid" not in nodes.columns:
        nodes = nodes.reset_index()

    # Limpiar y formatear 'osmid'
    nodes["osmid"] = nodes["osmid"].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x)).astype(str)
    edges["osmid"] = edges["osmid"].apply(lambda x: ",".join(map(str, x)) if isinstance(x, list) else str(x)).astype(str)

    # Limpiar highway y calcular consumo
    edges["highway_clean"] = edges["highway"].apply(clean_highway)
    edges = edges[edges["highway_clean"].isin(valid_types)].copy()
    edges["length_km"] = edges["length"] / 1000
    edges["fuel_consumption"] = edges.apply(
        lambda row: row["length_km"] * calc_avg_fuel_consumption(row["highway_clean"], row["speed_kph"]),
        axis=1
    )

    # Limpiar columnas no necesarias
    nodes["geometry"] = nodes["geometry"].apply(lambda g: g.wkt)
    edges["geometry"] = edges["geometry"].apply(lambda g: g.wkt)

    edges["name"] = edges["name"].apply(normalize_name)

    nodes = nodes[["osmid", "x", "y", "lez", "highway", "geometry"]].copy()
    edges = edges[[
        "u", "v", "key", "osmid", "length", "speed_kph", "travel_time",
        "highway_clean", "fuel_consumption", "geometry", "lanes", "name",
        "oneway", "in_lez"
    ]].copy()

    # Guardar como JSON (newline-delimited)
    nodes.to_json("nodes_clean.json", orient="records", lines=False)
    edges.to_json("edges_clean.json", orient="records", lines=False)
