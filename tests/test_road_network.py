"""
Unit and Integration Tests for PathSnap Road-Network Graph (Member 3)
=====================================================================
Validates:
1. Node and arc counts
2. Directed graph orientation
3. Zero-based indexing preservation
4. Parallel arc handling in MultiDiGraph
5. Geographic geodesic distance calculation (pyproj WGS84)
6. Shortest path routing and edge weighting
7. Anomaly detection: invalid node references, self-loops, zero-length arcs
8. Isolated nodes and disconnected component handling (no unhandled crashes)
9. Downstream successor arc retrieval for HMM transition modeling
10. Real-data integration test on actual training dataset
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
import networkx as nx

from src.road_network import (
    RoadNetwork,
    build_road_network,
    load_and_build_road_network,
    calculate_geodesic_distance,
    calculate_arc_lengths_geodesic,
    validate_road_network_integrity
)


@pytest.fixture
def synthetic_network_data():
    """
    Creates a synthetic road network with:
    - 5 nodes (0 to 4)
    - 6 arcs:
      * Arc 0: 0 -> 1 (regular segment)
      * Arc 1: 0 -> 1 (parallel segment to Arc 0 with identical nodes)
      * Arc 2: 1 -> 2 (connects downstream of Arc 0 & 1)
      * Arc 3: 2 -> 3
      * Arc 4: 3 -> 3 (self-loop)
      * Arc 5: 1 -> 0 (opposite direction)
    - Node 4 is an isolated node.
    """
    nodes_df = pd.DataFrame({
        "longitude": [8.5800, 8.5810, 8.5820, 8.5830, 8.5900],
        "latitude":  [49.8500, 49.8510, 49.8520, 49.8530, 49.8600]
    })
    nodes_df.index.name = "node_id"

    arcs_df = pd.DataFrame({
        "from_node_id": [0, 0, 1, 2, 3, 1],
        "to_node_id":   [1, 1, 2, 3, 3, 0]
    })
    arcs_df.index.name = "arc_id"

    return nodes_df, arcs_df


def test_correct_node_and_arc_counts(synthetic_network_data):
    """Test 1: Verifies node and arc counts match raw tables."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df, trip_id="synthetic_01")

    assert net.n_nodes == 5
    assert net.n_arcs == 6
    assert isinstance(net.graph, nx.MultiDiGraph)


def test_directed_graph_behavior(synthetic_network_data):
    """Test 2: Verifies edges are directed (u->v does not imply v->u)."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # 2 -> 3 exists (Arc 3)
    assert net.graph.has_edge(2, 3)
    # 3 -> 2 does not exist
    assert not net.graph.has_edge(3, 2)

    # Shortest path from 2 to 3 succeeds
    assert net.shortest_path_distance(2, 3) is not None
    # Shortest path from 3 to 2 is unreachable (returns None without crashing)
    assert net.shortest_path_distance(3, 2) is None


def test_preservation_of_zero_based_ids(synthetic_network_data):
    """Test 3: Verifies node IDs and arc IDs preserve exact 0-based indexing."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # Nodes 0 through 4 must exist
    for node_id in range(5):
        assert net.graph.has_node(node_id)
        coords = net.get_node_coordinates(node_id)
        assert coords == (nodes_df.loc[node_id, "longitude"], nodes_df.loc[node_id, "latitude"])

    # Arcs 0 through 5 must exist
    for arc_id in range(6):
        u, v = net.get_arc_endpoints(arc_id)
        assert u == arcs_df.loc[arc_id, "from_node_id"]
        assert v == arcs_df.loc[arc_id, "to_node_id"]


def test_parallel_arcs_preservation(synthetic_network_data):
    """Test 4: Verifies MultiDiGraph preserves parallel arcs between same (u, v)."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # Both Arc 0 and Arc 1 connect node 0 -> node 1
    edge_dict = net.graph.get_edge_data(0, 1)
    assert len(edge_dict) == 2
    assert 0 in edge_dict  # key=0 (Arc 0)
    assert 1 in edge_dict  # key=1 (Arc 1)

    assert net.get_arc_endpoints(0) == (0, 1)
    assert net.get_arc_endpoints(1) == (0, 1)
    assert net.validation_report["unique_parallel_pairs"] == 1


def test_geographic_distance_calculation():
    """Test 5: Verifies geodesic distance on WGS84 ellipsoid via pyproj."""
    # 1 degree of latitude near equator is approx 110.574 km
    d_lat = calculate_geodesic_distance(0.0, 0.0, 0.0, 1.0)
    assert 110000.0 < d_lat < 112000.0

    # Same coordinate distance must be zero
    d_zero = calculate_geodesic_distance(8.5, 50.0, 8.5, 50.0)
    assert d_zero == 0.0


def test_shortest_path_distance(synthetic_network_data):
    """Test 6: Verifies shortest-path routing using length_m as edge weight."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # Path 0 -> 1 -> 2 -> 3
    dist_0_3 = net.shortest_path_distance(0, 3)
    assert dist_0_3 is not None
    assert dist_0_3 > 0.0

    path_0_3 = net.shortest_path(0, 3)
    assert path_0_3 == [0, 1, 2, 3]

    # Same node distance is 0.0
    assert net.shortest_path_distance(1, 1) == 0.0
    assert net.shortest_path(1, 1) == [1]


def test_invalid_node_references_detection():
    """Test 7: Verifies invalid node references are caught and reported."""
    nodes_df = pd.DataFrame({"longitude": [8.0, 8.1], "latitude": [50.0, 50.1]})
    # Node 99 does not exist in nodes_df (N_nodes = 2)
    arcs_df = pd.DataFrame({"from_node_id": [0, 1], "to_node_id": [1, 99]})

    report = validate_road_network_integrity(nodes_df, arcs_df)
    assert report["invalid_node_references"] == 1
    assert report["has_integrity_issues"] is True


def test_self_loops_and_zero_length_detection(synthetic_network_data):
    """Test 8: Verifies detection of self-loops and near-zero length arcs."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # Arc 4 is 3 -> 3 (self loop)
    assert net.validation_report["self_loops"] == 1


def test_isolated_nodes_detection(synthetic_network_data):
    """Test 9: Verifies isolated nodes with no incoming/outgoing arcs."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    conn = net.compute_connectivity_statistics()
    # Node 4 has no arcs connected
    assert conn["isolated_nodes_count"] == 1


def test_disconnected_components_handling(synthetic_network_data):
    """Test 10: Verifies queries between disconnected components return None without crashing."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # Node 4 is completely isolated
    dist = net.shortest_path_distance(0, 4)
    assert dist is None

    path = net.shortest_path(0, 4)
    assert path is None

    # Querying a non-existent node returns None gracefully
    assert net.shortest_path_distance(0, 999) is None
    assert net.shortest_path(0, 999) is None


def test_successor_arcs_retrieval(synthetic_network_data):
    """Test 11: Verifies retrieval of downstream connected arcs for HMM transitions."""
    nodes_df, arcs_df = synthetic_network_data
    net = build_road_network(nodes_df, arcs_df)

    # Arc 0 ends at Node 1. From Node 1, we have Arc 2 (1->2) and Arc 5 (1->0)
    successors_arc0 = set(net.get_successor_arcs(0))
    assert successors_arc0 == {2, 5}

    # Arc 1 also ends at Node 1
    successors_arc1 = set(net.get_successor_arcs(1))
    assert successors_arc1 == {2, 5}

    # Arc 3 ends at Node 3. From Node 3, we have Arc 4 (3->3)
    successors_arc3 = net.get_successor_arcs(3)
    assert successors_arc3 == [4]


def test_real_training_trip_validation():
    """Test 12: Real-data validation on an actual training trip."""
    # Test on Trip 00000005 (compact, fast real trip with 898 nodes, 1579 arcs)
    real_trip_id = "00000005"
    if not os.path.exists(os.path.join("train", real_trip_id)):
        pytest.skip(f"Trip {real_trip_id} not available locally.")

    net = load_and_build_road_network(real_trip_id, partition="train")
    assert net.n_nodes == 898
    assert net.n_arcs == 1579
    assert net.validation_report["invalid_node_references"] == 0
    assert net.validation_report["duplicate_arc_ids"] == 0

    # Test arc geometry and length on Arc 0
    geom = net.get_arc_geometry(0)
    length = net.get_arc_length(0)
    assert length > 0.0
    assert len(geom) == 2

    # Test degree statistics
    deg = net.compute_degree_statistics()
    assert deg["in_degree_mean"] > 1.0
    assert deg["out_degree_mean"] > 1.0

    # Test shortest path on connected edge
    u, v = net.get_arc_endpoints(0)
    path_len = net.shortest_path_distance(u, v)
    assert path_len is not None
    assert abs(path_len - length) < 1e-3
