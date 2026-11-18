"""
PathSnap - Dataset Loader Module
================================
Provides reusable, robust functions to load and verify trajectory and road-network data.

Dataset structure per trip:
- .track: whitespace-separated [longitude, latitude, timestamp (seconds)]
- .nodes: whitespace-separated [longitude, latitude] (0-based line index is node_id)
- .arcs: whitespace-separated [from_node_id, to_node_id] (0-based line index is arc_id)
- .route: single integer column [arc_id] (reference route sequence, when available)

All raw data is preserved unchanged without modifications.
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class TripData:
    """Represents a complete trip dataset."""
    trip_id: str
    partition: str
    track: pd.DataFrame       # Columns: ['longitude', 'latitude', 'timestamp']
    nodes: pd.DataFrame       # Columns: ['longitude', 'latitude'] (index=node_id)
    arcs: pd.DataFrame        # Columns: ['from_node_id', 'to_node_id'] (index=arc_id)
    route: Optional[pd.DataFrame]  # Columns: ['arc_id'] or None if unavailable

    @property
    def n_points(self) -> int:
        return len(self.track)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_arcs(self) -> int:
        return len(self.arcs)

    @property
    def n_route_arcs(self) -> int:
        return len(self.route) if self.route is not None else 0

    def __repr__(self) -> str:
        return (f"<TripData trip_id='{self.trip_id}' partition='{self.partition}' "
                f"points={self.n_points} nodes={self.n_nodes} arcs={self.n_arcs} route_arcs={self.n_route_arcs}>")


def resolve_partition_dir(partition: str, base_dir: str = ".") -> str:
    """
    Locates the partition directory, checking both:
    1. {base_dir}/data/{partition}
    2. {base_dir}/{partition}
    """
    candidates = [
        os.path.join(base_dir, "data", partition),
        os.path.join(base_dir, partition)
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"Partition directory not found for '{partition}'. Checked: {candidates}")


def list_trips(partition: str = "train", base_dir: str = ".") -> List[str]:
    """
    Returns a sorted list of trip identifiers for a given partition.
    """
    p_dir = resolve_partition_dir(partition, base_dir)
    trips = [
        d for d in os.listdir(p_dir)
        if not d.startswith(".") and os.path.isdir(os.path.join(p_dir, d))
    ]
    return sorted(trips)


def load_track(filepath: str) -> pd.DataFrame:
    """
    Loads raw GPS observations from a .track file.
    Columns: longitude (float), latitude (float), timestamp (float).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Track file not found: {filepath}")
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
        dtype={"longitude": np.float64, "latitude": np.float64, "timestamp": np.float64}
    )
    return df


def load_nodes(filepath: str) -> pd.DataFrame:
    """
    Loads road-network nodes from a .nodes file.
    0-based row index represents node_id.
    Columns: longitude (float), latitude (float).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Nodes file not found: {filepath}")
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
        dtype={"longitude": np.float64, "latitude": np.float64}
    )
    df.index.name = "node_id"
    return df


def load_arcs(filepath: str) -> pd.DataFrame:
    """
    Loads road-network arcs (directed road segments) from a .arcs file.
    0-based row index represents arc_id.
    Columns: from_node_id (int), to_node_id (int).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Arcs file not found: {filepath}")
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype={"from_node_id": np.int64, "to_node_id": np.int64}
    )
    df.index.name = "arc_id"
    return df


def load_route(filepath: str) -> Optional[pd.DataFrame]:
    """
    Loads reference route (sequence of traversed arc_ids) from a .route file.
    Returns None if file does not exist.
    """
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(
        filepath,
        header=None,
        names=["arc_id"],
        dtype={"arc_id": np.int64}
    )
    return df


def load_trip(trip_id: str, partition: str = "train", base_dir: str = ".") -> TripData:
    """
    Loads all associated data for a specific trip into a TripData object.
    Preserves raw data exactly as stored.
    """
    p_dir = resolve_partition_dir(partition, base_dir)
    trip_dir = os.path.join(p_dir, trip_id)
    if not os.path.isdir(trip_dir):
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")

    track_df = load_track(track_file)
    nodes_df = load_nodes(nodes_file)
    arcs_df = load_arcs(arcs_file)
    route_df = load_route(route_file)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        track=track_df,
        nodes=nodes_df,
        arcs=arcs_df,
        route=route_df
    )


def verify_trip_integrity(trip: TripData) -> Dict[str, Any]:
    """
    Performs structural integrity and indexing verification on a trip:
    - 0-based node reference bounds in arcs
    - 0-based arc reference bounds in route
    - Latitude and longitude validity bounds
    - Monotonicity of timestamps
    - Route continuity check (if route exists)
    """
    issues = []
    n_nodes = trip.n_nodes
    n_arcs = trip.n_arcs

    # 1. Coordinate ranges in track
    lat_valid = trip.track["latitude"].between(-90.0, 90.0).all()
    lon_valid = trip.track["longitude"].between(-180.0, 180.0).all()
    if not lat_valid:
        issues.append("Invalid latitude found in .track outside [-90, 90]")
    if not lon_valid:
        issues.append("Invalid longitude found in .track outside [-180, 180]")

    # 2. Coordinate ranges in nodes
    n_lat_valid = trip.nodes["latitude"].between(-90.0, 90.0).all()
    n_lon_valid = trip.nodes["longitude"].between(-180.0, 180.0).all()
    if not n_lat_valid:
        issues.append("Invalid latitude found in .nodes outside [-90, 90]")
    if not n_lon_valid:
        issues.append("Invalid longitude found in .nodes outside [-180, 180]")

    # 3. Arcs referencing node bounds
    u_min, u_max = trip.arcs["from_node_id"].min(), trip.arcs["from_node_id"].max()
    v_min, v_max = trip.arcs["to_node_id"].min(), trip.arcs["to_node_id"].max()
    arcs_valid = (u_min >= 0) and (u_max < n_nodes) and (v_min >= 0) and (v_max < n_nodes)
    if not arcs_valid:
        issues.append(f"Arc node index out of bounds [0, {n_nodes - 1}]: u=[{u_min}, {u_max}], v=[{v_min}, {v_max}]")

    # 4. Route referencing arc bounds
    route_valid = True
    continuity_pct = 100.0
    if trip.route is not None and len(trip.route) > 0:
        r_min, r_max = trip.route["arc_id"].min(), trip.route["arc_id"].max()
        route_valid = (r_min >= 0) and (r_max < n_arcs)
        if not route_valid:
            issues.append(f"Route arc index out of bounds [0, {n_arcs - 1}]: [{r_min}, {r_max}]")

        # Route continuity: does arc[i].to_node_id == arc[i+1].from_node_id?
        if len(trip.route) > 1:
            route_arcs = trip.arcs.iloc[trip.route["arc_id"].values].reset_index(drop=True)
            prev_to = route_arcs["to_node_id"].values[:-1]
            next_from = route_arcs["from_node_id"].values[1:]
            continuity_matches = np.sum(prev_to == next_from)
            continuity_pct = float((continuity_matches / len(prev_to)) * 100.0)

    # 5. Timestamp monotonicity
    diffs = np.diff(trip.track["timestamp"].values)
    is_monotonic = bool(np.all(diffs >= 0))
    if not is_monotonic:
        issues.append("Timestamps are not strictly non-decreasing in .track")

    return {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_points": trip.n_points,
        "n_nodes": trip.n_nodes,
        "n_arcs": trip.n_arcs,
        "n_route_arcs": trip.n_route_arcs,
        "lat_range": (float(trip.track["latitude"].min()), float(trip.track["latitude"].max())),
        "lon_range": (float(trip.track["longitude"].min()), float(trip.track["longitude"].max())),
        "nodes_lat_range": (float(trip.nodes["latitude"].min()), float(trip.nodes["latitude"].max())),
        "nodes_lon_range": (float(trip.nodes["longitude"].min()), float(trip.nodes["longitude"].max())),
        "arcs_valid": arcs_valid,
        "route_valid": route_valid,
        "route_continuity_pct": continuity_pct,
        "is_timestamp_monotonic": is_monotonic,
        "issues": issues,
        "has_issues": len(issues) > 0
    }
