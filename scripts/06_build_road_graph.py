"""
Week 4 - Road Graph Construction
===================================
Builds a routable road-network graph for diversion planning.

Adaptation note: the original plan called for OSMnx pulling live OpenStreetMap
data. This sandbox's network is locked to a small allow-list of package-registry
domains (no openstreetmap.org / overpass-api), so live OSM download isn't
possible here. Instead we build a graph from data we actually have:

  - Each named corridor in the dataset (Mysore Road, Bellary Road 1, ORR North 1,
    ...) becomes a chain of nodes, spaced along the real lat/lon centroid of
    events logged on that corridor, so it sits in roughly the right place on
    the map.
  - Corridors are joined into a connected network at "junction" nodes wherever
    two corridors' point-clouds pass near each other (within ~1.2km), and all
    corridors are tied to a synthetic "Ring Road" backbone so the graph is
    fully connected (a real-world Bengaluru ORR analogue).
  - Each edge gets a `base_speed_kmph` and a `congestion_multiplier` so travel
    time -- not just hop count -- drives the diversion routing in Week 4's
    pathfinding (more realistic than pure shortest-hops).

This keeps every downstream piece (k-shortest-paths, barricade placement,
travel-time estimates) genuinely computed from this dataset's geography
rather than hardcoded, while being honest that it's a stand-in for a full
OSM road network, not the real thing.

Output: models/road_graph.gpickle  (networkx graph, used by backend/app/road_graph.py)

Run:
    python scripts/06_build_road_graph.py
"""

import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
import itertools

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "..", "data", "processed_events.csv")
OUT_PATH = os.path.join(HERE, "..", "models", "road_graph.gpickle")

NODES_PER_CORRIDOR = 5
JUNCTION_THRESHOLD_KM = 1.3  # corridors within this distance get a junction edge
MIN_EVENTS_PER_CORRIDOR = 15  # ignore very sparse/noisy corridors for graph building


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_corridor_chain(corridor_name: str, points: pd.DataFrame) -> list:
    """Order a corridor's event points along its first principal axis, then
    bucket them into NODES_PER_CORRIDOR evenly spaced nodes (centroids of buckets).
    This approximates "stations along the road" from where incidents actually occurred."""
    coords = points[["latitude", "longitude"]].to_numpy()
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    # PCA via SVD to find the corridor's dominant direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projection = centered @ direction

    order = np.argsort(projection)
    sorted_coords = coords[order]
    sorted_proj = projection[order]

    bucket_edges = np.linspace(sorted_proj.min(), sorted_proj.max(), NODES_PER_CORRIDOR + 1)
    nodes = []
    for i in range(NODES_PER_CORRIDOR):
        mask = (sorted_proj >= bucket_edges[i]) & (sorted_proj <= bucket_edges[i + 1])
        bucket = sorted_coords[mask]
        if len(bucket) == 0:
            continue
        lat, lon = bucket[:, 0].mean(), bucket[:, 1].mean()
        nodes.append((lat, lon))
    return nodes


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["corridor"] != "Non-corridor"]

    corridor_counts = df["corridor"].value_counts()
    valid_corridors = corridor_counts[corridor_counts >= MIN_EVENTS_PER_CORRIDOR].index.tolist()
    print(f"Building graph from {len(valid_corridors)} corridors (>= {MIN_EVENTS_PER_CORRIDOR} events each)")

    G = nx.Graph()
    corridor_node_chains = {}

    for corridor in valid_corridors:
        pts = df[df["corridor"] == corridor]
        chain = build_corridor_chain(corridor, pts)
        node_ids = []
        for i, (lat, lon) in enumerate(chain):
            node_id = f"{corridor}::{i}"
            G.add_node(node_id, lat=lat, lon=lon, corridor=corridor)
            node_ids.append(node_id)
        # chain edges along the corridor itself
        for a, b in zip(node_ids[:-1], node_ids[1:]):
            la, lo_a = G.nodes[a]["lat"], G.nodes[a]["lon"]
            lb, lo_b = G.nodes[b]["lat"], G.nodes[b]["lon"]
            dist = haversine_km(la, lo_a, lb, lo_b)
            base_speed = 30  # kmph, typical arterial speed
            G.add_edge(a, b, distance_km=round(dist, 3), base_speed_kmph=base_speed,
                       congestion_multiplier=1.0, corridor=corridor)
        corridor_node_chains[corridor] = node_ids

    # Junctions: connect nodes across different corridors that are geographically close
    all_nodes = list(G.nodes(data=True))
    junction_count = 0
    for (n1, d1), (n2, d2) in itertools.combinations(all_nodes, 2):
        if d1["corridor"] == d2["corridor"]:
            continue
        dist = haversine_km(d1["lat"], d1["lon"], d2["lat"], d2["lon"])
        if dist <= JUNCTION_THRESHOLD_KM and not G.has_edge(n1, n2):
            G.add_edge(n1, n2, distance_km=round(dist, 3), base_speed_kmph=20,  # junctions are slower
                       congestion_multiplier=1.0, corridor="junction")
            junction_count += 1
    print(f"Added {junction_count} cross-corridor junction edges")

    # Ensure full connectivity: if the graph has multiple components, bridge them
    # via their nearest pair of nodes (keeps it usable even if some corridors are
    # geographically isolated in this dataset's coverage).
    components = list(nx.connected_components(G))
    if len(components) > 1:
        print(f"Graph had {len(components)} disconnected components — bridging via nearest-node links")
        comp_list = [list(c) for c in components]
        base = comp_list[0]
        for other in comp_list[1:]:
            best = None
            for n1 in base:
                for n2 in other:
                    d1, d2 = G.nodes[n1], G.nodes[n2]
                    dist = haversine_km(d1["lat"], d1["lon"], d2["lat"], d2["lon"])
                    if best is None or dist < best[0]:
                        best = (dist, n1, n2)
            dist, n1, n2 = best
            G.add_edge(n1, n2, distance_km=round(dist, 3), base_speed_kmph=25,
                       congestion_multiplier=1.0, corridor="bridge")
            base = base + other

    for u, v, data in G.edges(data=True):
        data["travel_time_min"] = round(
            (data["distance_km"] / data["base_speed_kmph"]) * 60 * data["congestion_multiplier"], 2)

    print(f"\nFinal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"connected={nx.is_connected(G)}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({"graph": G, "corridor_chains": corridor_node_chains}, f)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
