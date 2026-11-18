"""
Real-Data Road-Network Validation Script (Member 3)
===================================================
Executes graph construction, validation, topological profiling, and shortest-path
routing on real training data, saving results/road_network_validation.csv.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.road_network import load_and_build_road_network


def run_real_data_validation(trip_id: str = "00000001", partition: str = "train"):
    print(f"=== Running Real-Data Road-Network Validation on Trip {trip_id} ({partition}) ===")
    t0 = time.time()
    net = load_and_build_road_network(trip_id, partition=partition)
    build_time = time.time() - t0
    print(f"Graph loaded and constructed in {build_time:.3f} seconds.")

    # 1. Summary metrics
    summary = net.summary()

    # 2. Shortest-path query examples on actual ground truth route
    route_file = os.path.join("train", trip_id, f"{trip_id}.route")
    route_arcs = []
    if os.path.exists(route_file):
        with open(route_file) as fp:
            route_arcs = [int(line.strip()) for line in fp if line.strip()]

    print(f"Ground-truth route has {len(route_arcs)} arcs.")

    queries = []
    if len(route_arcs) >= 2:
        # Query 1: Consecutive route arcs (direct transition)
        arc_a, arc_b = route_arcs[0], route_arcs[1]
        u_a, v_a = net.get_arc_endpoints(arc_a)
        u_b, v_b = net.get_arc_endpoints(arc_b)
        dist_direct = net.shortest_path_between_arcs(arc_a, arc_b)
        queries.append({
            "query_type": "consecutive_route_transition",
            "source_arc": arc_a,
            "target_arc": arc_b,
            "source_node": v_a,
            "target_node": u_b,
            "path_distance_m": dist_direct,
            "is_reachable": (dist_direct is not None)
        })

        # Query 2: Multi-hop route traversal (Arc 0 to Arc 10)
        idx_end = min(10, len(route_arcs) - 1)
        arc_start = route_arcs[0]
        arc_end = route_arcs[idx_end]
        u_s, v_s = net.get_arc_endpoints(arc_start)
        u_e, v_e = net.get_arc_endpoints(arc_end)
        dist_multihop = net.shortest_path_distance(u_s, v_e)
        queries.append({
            "query_type": "multihop_route_traversal",
            "source_arc": arc_start,
            "target_arc": arc_end,
            "source_node": u_s,
            "target_node": v_e,
            "path_distance_m": dist_multihop,
            "is_reachable": (dist_multihop is not None)
        })

    # Query 3: Random far apart node query
    nodes_list = list(net.graph.nodes())
    if len(nodes_list) > 100:
        n_far_1 = nodes_list[10]
        n_far_2 = nodes_list[-10]
        dist_far = net.shortest_path_distance(n_far_1, n_far_2)
        queries.append({
            "query_type": "distant_node_pair",
            "source_arc": None,
            "target_arc": None,
            "source_node": n_far_1,
            "target_node": n_far_2,
            "path_distance_m": dist_far,
            "is_reachable": (dist_far is not None)
        })

    # Query 4: Self loop / identical node query
    n_self = nodes_list[0]
    queries.append({
        "query_type": "identical_node",
        "source_arc": None,
        "target_arc": None,
        "source_node": n_self,
        "target_node": n_self,
        "path_distance_m": net.shortest_path_distance(n_self, n_self),
        "is_reachable": True
    })

    # Save to results/road_network_validation.csv
    os.makedirs("results", exist_ok=True)
    out_csv = os.path.join("results", "road_network_validation.csv")

    # Combine summary and queries into a structured DataFrame
    summary_rows = [{"metric": k, "value": v} for k, v in summary.items()]
    summary_rows.append({"metric": "graph_build_time_sec", "value": round(build_time, 4)})

    for i, q in enumerate(queries, 1):
        summary_rows.append({
            "metric": f"shortest_path_query_{i}_{q['query_type']}",
            "value": f"dist={q['path_distance_m']}m (reachable={q['is_reachable']})"
        })

    df_out = pd.DataFrame(summary_rows)
    df_out.to_csv(out_csv, index=False)
    print(f"\nValidation metrics saved to: {out_csv} ({len(df_out)} rows)")

    # Print summary to console
    print("\n=== ROAD NETWORK STATISTICAL REPORT ===")
    for k, v in summary.items():
        print(f"  {k:32s}: {v}")
    print(f"  {'build_time_sec':32s}: {build_time:.4f}s")

    print("\n=== SHORTEST PATH QUERY EXAMPLES ===")
    for q in queries:
        print(f"  [{q['query_type']}]: Source {q['source_node']} -> Target {q['target_node']} | Distance: {q['path_distance_m']} m | Reachable: {q['is_reachable']}")

    return summary, queries


if __name__ == "__main__":
    run_real_data_validation("00000001", "train")
