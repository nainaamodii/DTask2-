"""
Week 4 - Diversion Routing & Barricade Placement Engine
===========================================================
Loads the road graph built by scripts/06_build_road_graph.py and, given an
incident location + severity, computes:

  1. The nearest graph node(s) to the incident (i.e. which corridor/segment is hit).
  2. K shortest alternate paths between the two corridor nodes adjacent to the
     incident, AFTER removing/penalizing the blocked edge(s) — this is the
     "diversion" recommendation.
  3. A barricade perimeter: points offset around the incident location at the
     recommended radius (driven by the impact score from rule_engine/ml_engine),
     simulating containment points a control room would physically staff.

This mirrors the plan's "Use NetworkX + OSMnx for road graph and diversion
path computation" step — using our synthetic-but-data-derived graph in place
of a live OSM pull (see scripts/06_build_road_graph.py for why).
"""

import os
import pickle
import math
import itertools
import networkx as nx
import numpy as np

HERE = os.path.dirname(__file__)
GRAPH_PATH = os.path.join(HERE, "..", "..", "models", "road_graph.gpickle")

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))


def offset_point(lat, lon, distance_km, bearing_deg):
    """Compute a lat/lon offset by distance_km along bearing_deg (0=N, 90=E)."""
    R = EARTH_R_KM
    brng = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                      math.cos(lat1) * math.sin(distance_km / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(distance_km / R) * math.cos(lat1),
                              math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


class RoadGraphEngine:
    def __init__(self):
        with open(GRAPH_PATH, "rb") as f:
            payload = pickle.load(f)
        self.G: nx.Graph = payload["graph"]
        self.corridor_chains: dict = payload["corridor_chains"]
        self._node_coords = {n: (d["lat"], d["lon"]) for n, d in self.G.nodes(data=True)}

    def nearest_node(self, lat: float, lon: float, corridor: str = None) -> str:
        """Find the nearest graph node to a lat/lon, optionally restricted to a named corridor."""
        candidates = self._node_coords.items()
        if corridor and corridor in self.corridor_chains:
            wanted = set(self.corridor_chains[corridor])
            candidates = [(n, c) for n, c in candidates if n in wanted]
        best_node, best_dist = None, float("inf")
        for node, (nlat, nlon) in candidates:
            d = haversine_km(lat, lon, nlat, nlon)
            if d < best_dist:
                best_node, best_dist = node, d
        return best_node, best_dist

    def diversions(self, lat: float, lon: float, corridor: str, k: int = 3, top_n: int = 2):
        """
        Compute up to `top_n` alternate routes between the two nodes nearest the
        incident on its corridor, after penalizing the segment(s) closest to the
        incident itself (treated as blocked).

        Returns a list of dicts: {path: [node ids], distance_km, est_minutes, via_corridors}
        or [] if the corridor isn't represented in the graph / has < 2 nodes.
        """
        if corridor not in self.corridor_chains or len(self.corridor_chains[corridor]) < 2:
            return []

        chain = self.corridor_chains[corridor]
        # find the two chain nodes that bracket the incident point
        dists = [(n, haversine_km(lat, lon, *self._node_coords[n])) for n in chain]
        dists.sort(key=lambda x: x[1])
        nearest = dists[0][0]
        idx = chain.index(nearest)
        start = chain[max(0, idx - 1)]
        end = chain[min(len(chain) - 1, idx + 1)]
        if start == end:
            return []

        # Penalize (rather than hard-delete) edges directly on the blocked segment so
        # the algorithm still finds a route even in a sparse graph, but strongly
        # prefers to route around it.
        G2 = self.G.copy()
        for u, v in zip(chain, chain[1:]):
            if G2.has_edge(u, v):
                G2[u][v]["travel_time_min"] *= 25  # heavy penalty == "treat as closed"

        try:
            paths_gen = nx.shortest_simple_paths(G2, start, end, weight="travel_time_min")
            results = []
            for path in itertools.islice(paths_gen, top_n):
                results.append(self._summarize_path(G2, path))
            return results
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def _summarize_path(self, G, path):
        total_dist = 0.0
        total_time = 0.0
        corridors_used = []
        for u, v in zip(path[:-1], path[1:]):
            edge = G[u][v]
            total_dist += edge["distance_km"]
            total_time += edge["travel_time_min"] if edge["travel_time_min"] < 100 else edge["travel_time_min"] / 25
            c = edge.get("corridor")
            if c and c not in corridors_used:
                corridors_used.append(c)
        return {
            "path_nodes": path,
            "path_coords": [self._node_coords[n] for n in path],
            "distance_km": round(total_dist, 2),
            "est_minutes": round(total_time, 1),
            "via_corridors": corridors_used,
        }

    def barricade_points(self, lat: float, lon: float, n_barricades: int, radius_km: float = 0.25):
        """Evenly space `n_barricades` containment points around the incident location."""
        if n_barricades <= 0:
            return []
        points = []
        for i in range(n_barricades):
            bearing = (360.0 / n_barricades) * i
            plat, plon = offset_point(lat, lon, radius_km, bearing)
            points.append({"latitude": round(plat, 6), "longitude": round(plon, 6),
                            "bearing_deg": round(bearing, 1)})
        return points


_engine_singleton = None


def get_road_graph_engine() -> RoadGraphEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = RoadGraphEngine()
    return _engine_singleton
