"""
PathSnap - Road-Network Graph Module (Member 3)
================================================
Provides graph construction, validation, topological analysis, and shortest-path
routing over directed road networks using NetworkX and pyproj (WGS84 Geodesic).

Preserves:
- Zero-based node IDs (matching .nodes row indices)
- Zero-based arc IDs (matching .arcs row indices as edge keys)
- Parallel directed arcs between identical junction node pairs via MultiDiGraph
- Physical geodesic lengths in metres along the WGS84 ellipsoid

Collaborative Integration:
- Member 1: Use `get_arc_geometry()` and `get_arc_endpoints()` for candidate generation.
- Member 4: Use `get_arc_geometry()` for Folium map visualization.
- Member 3 (Next Phase): Use `get_successor_arcs()` and `shortest_path_between_arcs()`
  for HMM transition probability matrices $A_{ij} = P(arc_j | arc_i)$.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import networkx as nx
import pyproj


# ==============================================================================
# 1. Geodesic Distance Calculations
# ==============================================================================

def calculate_geodesic_distance(
    lon1: float, lat1: float,
    lon2: float, lat2: float,
    ellipsoid: str = "WGS84"
) -> float:
    """
    Calculates exact geodesic distance in metres between two coordinates
    along the WGS84 ellipsoid using pyproj.Geod.
    """
    geod = pyproj.Geod(ellps=ellipsoid)
    _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
    return float(dist)


def calculate_arc_lengths_geodesic(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    ellipsoid: str = "WGS84"
) -> np.ndarray:
    """
    Vectorized calculation of geodesic physical arc lengths in metres
    for all road segments in arcs_df using pyproj.Geod(WGS84).

    Returns:
        np.ndarray of length N_arcs containing physical length in metres.
    """
    geod = pyproj.Geod(ellps=ellipsoid)
    u = arcs_df["from_node_id"].values if "from_node_id" in arcs_df.columns else arcs_df.iloc[:, 0].values
    v = arcs_df["to_node_id"].values if "to_node_id" in arcs_df.columns else arcs_df.iloc[:, 1].values

    lon_col = "longitude" if "longitude" in nodes_df.columns else nodes_df.columns[0]
    lat_col = "latitude" if "latitude" in nodes_df.columns else nodes_df.columns[1]

    # Node coordinates lookup
    lons = nodes_df[lon_col].values
    lats = nodes_df[lat_col].values

    from_lon = lons[u]
    from_lat = lats[u]
    to_lon = lons[v]
    to_lat = lats[v]

    _, _, distances = geod.inv(from_lon, from_lat, to_lon, to_lat)
    return np.asarray(distances, dtype=np.float64)


# ==============================================================================
# 2. Graph Integrity & Validation
# ==============================================================================

def validate_road_network_integrity(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    lengths_m: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Validates structural and spatial integrity of road network tables:
    - Invalid node references (< 0 or >= N_nodes)
    - Duplicate arc IDs (row indexing consistency)
    - Self-loops (from_node_id == to_node_id)
    - Zero-length or near-zero arcs (< 0.01 metres)
    - Parallel directed arcs between the same node pair
    """
    n_nodes = len(nodes_df)
    n_arcs = len(arcs_df)

    u = arcs_df["from_node_id"].values if "from_node_id" in arcs_df.columns else arcs_df.iloc[:, 0].values
    v = arcs_df["to_node_id"].values if "to_node_id" in arcs_df.columns else arcs_df.iloc[:, 1].values

    # 1. Invalid node references
    invalid_u = (u < 0) | (u >= n_nodes)
    invalid_v = (v < 0) | (v >= n_nodes)
    invalid_node_refs = int(np.sum(invalid_u | invalid_v))

    # 2. Duplicate arc IDs
    duplicate_arc_ids = int(arcs_df.index.duplicated().sum())

    # 3. Self-loops
    self_loops = int(np.sum(u == v))

    # 4. Parallel directed edges (same u and v)
    edge_pairs = pd.DataFrame({"u": u, "v": v})
    parallel_edges_count = int(edge_pairs.duplicated(keep=False).sum())
    unique_parallel_pairs = int(edge_pairs.duplicated(keep="first").sum())

    # 5. Zero-length arcs
    if lengths_m is None and invalid_node_refs == 0:
        lengths_m = calculate_arc_lengths_geodesic(nodes_df, arcs_df)

    zero_length_count = 0
    if lengths_m is not None:
        zero_length_count = int(np.sum(lengths_m < 0.01))

    return {
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "valid_edges": n_arcs - invalid_node_refs,
        "invalid_node_references": invalid_node_refs,
        "duplicate_arc_ids": duplicate_arc_ids,
        "self_loops": self_loops,
        "parallel_edges_count": parallel_edges_count,
        "unique_parallel_pairs": unique_parallel_pairs,
        "zero_length_arcs": zero_length_count,
        "has_integrity_issues": (invalid_node_refs > 0 or duplicate_arc_ids > 0)
    }


# ==============================================================================
# 3. RoadNetwork Object & Constructor
# ==============================================================================

class RoadNetwork:
    """
    Encapsulates a directed road-network graph using NetworkX MultiDiGraph,
    preserving zero-based node IDs, zero-based arc IDs (as edge keys),
    parallel arcs, and physical geodesic lengths.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        trip_id: Optional[str] = None,
        arc_lookup: Optional[Dict[int, Tuple[int, int, int]]] = None,
        validation_report: Optional[Dict[str, Any]] = None
    ):
        self.graph = graph
        self.trip_id = trip_id
        self._arc_lookup = arc_lookup or {}
        self.validation_report = validation_report or {}

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_arcs(self) -> int:
        return self.graph.number_of_edges()

    def get_node_coordinates(self, node_id: int) -> Tuple[float, float]:
        """Returns (longitude, latitude) for a given node_id."""
        node_data = self.graph.nodes[node_id]
        return (node_data["longitude"], node_data["latitude"])

    def get_arc_endpoints(self, arc_id: int) -> Tuple[int, int]:
        """
        Retrieves (from_node_id, to_node_id) for a given arc_id in O(1) time.
        """
        if arc_id in self._arc_lookup:
            u, v, _ = self._arc_lookup[arc_id]
            return (u, v)
        raise KeyError(f"Arc ID {arc_id} not found in road network.")

    def get_arc_geometry(self, arc_id: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Retrieves line-segment geometry for an arc:
        ((from_lon, from_lat), (to_lon, to_lat))
        """
        u, v = self.get_arc_endpoints(arc_id)
        edge_data = self.graph.get_edge_data(u, v, key=arc_id)
        if edge_data and "geometry" in edge_data:
            return edge_data["geometry"]
        # Fallback to node coordinates
        u_coords = self.get_node_coordinates(u)
        v_coords = self.get_node_coordinates(v)
        return (u_coords, v_coords)

    def get_arc_length(self, arc_id: int) -> float:
        """Returns physical length in metres for an arc."""
        u, v = self.get_arc_endpoints(arc_id)
        edge_data = self.graph.get_edge_data(u, v, key=arc_id)
        if edge_data and "length_m" in edge_data:
            return float(edge_data["length_m"])
        u_lon, u_lat = self.get_node_coordinates(u)
        v_lon, v_lat = self.get_node_coordinates(v)
        return calculate_geodesic_distance(u_lon, u_lat, v_lon, v_lat)

    def get_successor_arcs(self, arc_id: int) -> List[int]:
        """
        Retrieves all downstream arc IDs that originate from the destination
        node (to_node_id) of the specified arc.
        
        Essential for HMM candidate transition modeling.
        """
        _, to_node = self.get_arc_endpoints(arc_id)
        successor_edges = self.graph.out_edges(to_node, keys=True)
        return [key for (_, _, key) in successor_edges]

    def get_predecessor_arcs(self, arc_id: int) -> List[int]:
        """
        Retrieves all upstream arc IDs that terminate at the origin
        node (from_node_id) of the specified arc.
        """
        from_node, _ = self.get_arc_endpoints(arc_id)
        predecessor_edges = self.graph.in_edges(from_node, keys=True)
        return [key for (_, _, key) in predecessor_edges]

    def shortest_path_distance(
        self,
        source_node: int,
        target_node: int,
        weight: str = "length_m"
    ) -> Optional[float]:
        """
        Computes shortest-path network distance in metres between two nodes
        using Dijkstra's algorithm.

        Returns None if target is unreachable (no path) or nodes do not exist.
        """
        if not self.graph.has_node(source_node) or not self.graph.has_node(target_node):
            return None
        if source_node == target_node:
            return 0.0

        try:
            dist = nx.shortest_path_length(
                self.graph,
                source=source_node,
                target=target_node,
                weight=weight
            )
            return float(dist)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def shortest_path(
        self,
        source_node: int,
        target_node: int,
        weight: str = "length_m"
    ) -> Optional[List[int]]:
        """
        Finds sequence of nodes forming the shortest path from source_node to target_node.
        Returns None if unreachable.
        """
        if not self.graph.has_node(source_node) or not self.graph.has_node(target_node):
            return None
        if source_node == target_node:
            return [source_node]

        try:
            path = nx.shortest_path(
                self.graph,
                source=source_node,
                target=target_node,
                weight=weight
            )
            return list(path)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def shortest_path_between_arcs(
        self,
        from_arc_id: int,
        to_arc_id: int,
        weight: str = "length_m"
    ) -> Optional[float]:
        """
        Computes transition network distance from the end of from_arc to the start of to_arc.
        If from_arc connects directly to to_arc (i.e. from_arc.to == to_arc.from), returns 0.0.
        If unreachable, returns None.
        """
        if from_arc_id == to_arc_id:
            return 0.0
        _, end_node = self.get_arc_endpoints(from_arc_id)
        start_node, _ = self.get_arc_endpoints(to_arc_id)
        return self.shortest_path_distance(end_node, start_node, weight=weight)

    def compute_degree_statistics(self) -> Dict[str, Any]:
        """Computes in-degree, out-degree, and total degree metrics across all nodes."""
        in_degrees = [d for _, d in self.graph.in_degree()]
        out_degrees = [d for _, d in self.graph.out_degree()]
        total_degrees = [d for _, d in self.graph.degree()]

        return {
            "in_degree_min": int(np.min(in_degrees)) if in_degrees else 0,
            "in_degree_max": int(np.max(in_degrees)) if in_degrees else 0,
            "in_degree_mean": float(np.mean(in_degrees)) if in_degrees else 0.0,
            "in_degree_median": float(np.median(in_degrees)) if in_degrees else 0.0,
            "out_degree_min": int(np.min(out_degrees)) if out_degrees else 0,
            "out_degree_max": int(np.max(out_degrees)) if out_degrees else 0,
            "out_degree_mean": float(np.mean(out_degrees)) if out_degrees else 0.0,
            "out_degree_median": float(np.median(out_degrees)) if out_degrees else 0.0,
            "total_degree_mean": float(np.mean(total_degrees)) if total_degrees else 0.0,
            "dead_ends_count": int(sum(1 for d in out_degrees if d == 0)),
            "source_nodes_count": int(sum(1 for d in in_degrees if d == 0))
        }

    def compute_connectivity_statistics(self) -> Dict[str, Any]:
        """
        Analyzes graph connectivity:
        - Isolated nodes
        - Weakly connected components (WCC)
        - Strongly connected components (SCC)
        """
        isolated_nodes = list(nx.isolates(self.graph))
        wcc = list(nx.weakly_connected_components(self.graph))
        scc = list(nx.strongly_connected_components(self.graph))

        wcc_sizes = [len(c) for c in wcc]
        scc_sizes = [len(c) for c in scc]

        return {
            "isolated_nodes_count": len(isolated_nodes),
            "weakly_connected_components": len(wcc),
            "largest_wcc_size": max(wcc_sizes) if wcc_sizes else 0,
            "largest_wcc_node_pct": (max(wcc_sizes) / self.n_nodes * 100.0) if (wcc_sizes and self.n_nodes > 0) else 0.0,
            "strongly_connected_components": len(scc),
            "largest_scc_size": max(scc_sizes) if scc_sizes else 0,
            "largest_scc_node_pct": (max(scc_sizes) / self.n_nodes * 100.0) if (scc_sizes and self.n_nodes > 0) else 0.0
        }

    def compute_arc_length_statistics(self) -> Dict[str, Any]:
        """Computes summary statistics of arc lengths in metres."""
        lengths = [
            data.get("length_m", 0.0)
            for _, _, _, data in self.graph.edges(keys=True, data=True)
        ]
        if not lengths:
            return {"count": 0}

        return {
            "arc_length_min_m": float(np.min(lengths)),
            "arc_length_max_m": float(np.max(lengths)),
            "arc_length_mean_m": float(np.mean(lengths)),
            "arc_length_median_m": float(np.median(lengths)),
            "arc_length_std_m": float(np.std(lengths)),
            "total_network_length_km": float(np.sum(lengths) / 1000.0)
        }

    def summary(self) -> Dict[str, Any]:
        """Consolidates complete road network statistics into a dictionary."""
        deg_stats = self.compute_degree_statistics()
        conn_stats = self.compute_connectivity_statistics()
        len_stats = self.compute_arc_length_statistics()

        res = {
            "trip_id": self.trip_id,
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            **self.validation_report,
            **deg_stats,
            **conn_stats,
            **len_stats
        }
        return res

    def __repr__(self) -> str:
        return (f"<RoadNetwork trip_id='{self.trip_id}' nodes={self.n_nodes} "
                f"arcs={self.n_arcs} parallel_pairs={self.validation_report.get('unique_parallel_pairs', 0)}>")


# ==============================================================================
# 4. Builder Functions
# ==============================================================================

def build_road_network(
    nodes_df: pd.DataFrame,
    arcs_df: pd.DataFrame,
    trip_id: Optional[str] = None,
    ellipsoid: str = "WGS84"
) -> RoadNetwork:
    """
    Constructs a directed NetworkX MultiDiGraph from nodes and arcs DataFrames.

    - Nodes are added with 'longitude' and 'latitude' attributes.
    - Arcs are added as directed edges (from_node_id -> to_node_id) with key=arc_id.
    - Preserves parallel arcs and zero-based indexing.
    - Computes physical arc length_m via pyproj.Geod(WGS84).
    """
    # 1. Validation & Distance Precomputation
    lengths_m = calculate_arc_lengths_geodesic(nodes_df, arcs_df, ellipsoid=ellipsoid)
    val_report = validate_road_network_integrity(nodes_df, arcs_df, lengths_m=lengths_m)

    lon_col = "longitude" if "longitude" in nodes_df.columns else nodes_df.columns[0]
    lat_col = "latitude" if "latitude" in nodes_df.columns else nodes_df.columns[1]

    u_col = "from_node_id" if "from_node_id" in arcs_df.columns else arcs_df.columns[0]
    v_col = "to_node_id" if "to_node_id" in arcs_df.columns else arcs_df.columns[1]

    lons = nodes_df[lon_col].values
    lats = nodes_df[lat_col].values
    u_vals = arcs_df[u_col].values
    v_vals = arcs_df[v_col].values

    # 2. Initialize MultiDiGraph
    G = nx.MultiDiGraph()

    # Batch add nodes
    node_tuples = (
        (i, {"longitude": float(lons[i]), "latitude": float(lats[i])})
        for i in range(len(nodes_df))
    )
    G.add_nodes_from(node_tuples)

    # Batch add edges with key=arc_id
    arc_lookup = {}
    edge_tuples = []
    for arc_id in range(len(arcs_df)):
        u = int(u_vals[arc_id])
        v = int(v_vals[arc_id])
        dist = float(lengths_m[arc_id])
        geom = ((float(lons[u]), float(lats[u])), (float(lons[v]), float(lats[v])))

        arc_lookup[arc_id] = (u, v, arc_id)
        edge_tuples.append((
            u, v, arc_id,
            {
                "arc_id": arc_id,
                "from_node_id": u,
                "to_node_id": v,
                "length_m": dist,
                "geometry": geom
            }
        ))

    G.add_edges_from(edge_tuples)

    return RoadNetwork(
        graph=G,
        trip_id=trip_id,
        arc_lookup=arc_lookup,
        validation_report=val_report
    )


def load_and_build_road_network(
    nodes_source: Union[str, pd.DataFrame],
    arcs_source: Optional[Union[str, pd.DataFrame]] = None,
    trip_id: Optional[str] = None,
    partition: str = "train",
    base_dir: str = "."
) -> RoadNetwork:
    """
    Convenience function that loads .nodes and .arcs files and builds a RoadNetwork.

    Supports:
    1. Member 1's data loader if available (`load_trip` or `load_nodes`/`load_arcs`).
    2. Direct filepaths to .nodes and .arcs files.
    3. Trip ID string lookup across base_dir.
    """
    # Case 1: Trip ID passed as string (e.g. "00000001")
    if isinstance(nodes_source, str) and (arcs_source is None or not os.path.isfile(arcs_source)):
        tid = nodes_source
        trip_id = tid
        # Attempt using Member 1 loader
        try:
            from .data_loader import load_trip
            trip_data = load_trip(tid, partition=partition, base_dir=base_dir)
            return build_road_network(trip_data.nodes, trip_data.arcs, trip_id=tid)
        except Exception:
            # Fallback direct file lookup
            p_dir = os.path.join(base_dir, "data", partition, tid)
            if not os.path.isdir(p_dir):
                p_dir = os.path.join(base_dir, partition, tid)
            nodes_path = os.path.join(p_dir, f"{tid}.nodes")
            arcs_path = os.path.join(p_dir, f"{tid}.arcs")
            nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["longitude", "latitude"])
            arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
            return build_road_network(nodes_df, arcs_df, trip_id=tid)

    # Case 2: File paths passed
    if isinstance(nodes_source, str) and isinstance(arcs_source, str):
        nodes_df = pd.read_csv(nodes_source, sep=r"\s+", header=None, names=["longitude", "latitude"])
        arcs_df = pd.read_csv(arcs_source, sep=r"\s+", header=None, names=["from_node_id", "to_node_id"])
        return build_road_network(nodes_df, arcs_df, trip_id=trip_id)

    # Case 3: DataFrames passed directly
    if isinstance(nodes_source, pd.DataFrame) and isinstance(arcs_source, pd.DataFrame):
        return build_road_network(nodes_source, arcs_source, trip_id=trip_id)

    raise ValueError(f"Invalid arguments for load_and_build_road_network: {nodes_source}, {arcs_source}")
