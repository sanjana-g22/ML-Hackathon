"""
PathSnap - Continuous Route Reconstruction (Member 3)
=====================================================
Converts discrete observation-level candidate arcs selected by Viterbi decoding
into a continuous, topologically valid directed route along the road network.

Key Capabilities:
- Connects consecutive non-adjacent arcs using directed shortest paths (Dijkstra).
- Reconstructs intermediate connecting road arcs in chronological order.
- Handles parallel directed arcs by selecting optimal physical segment lengths.
- Preserves legitimate loops and repeated road visits without global deduplication.
- Preserves trip-local zero-based arc IDs matching .arcs and .route files.
- Detects and documents disconnected route gaps where no path exists.
- Generates two explicit outputs:
  1. Observation-level selected road arcs (aligned with GPS points).
  2. Complete continuous reconstructed road arc sequence (including intermediate arcs).
- Computes comprehensive route metrics: continuity, length, and ground-truth similarity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np

from src.road_network import RoadNetwork


@dataclass
class RouteGap:
    """Represents a topological discontinuity between consecutive observation arcs."""
    from_obs_idx: int
    to_obs_idx: int
    from_arc_id: int
    to_arc_id: int
    from_node: int
    to_node: int


@dataclass
class ReconstructedRoute:
    """Encapsulates the complete reconstructed trajectory and evaluation diagnostics."""
    trip_id: str
    observation_arcs: List[int]                  # Arc ID per GPS point (len == N_obs)
    continuous_route_arcs: List[int]             # Full chronological arc sequence including intermediate arcs
    has_gaps: bool                               # True if any topological break occurred
    gaps: List[RouteGap]                         # Detailed list of route gaps
    total_length_m: float                        # Cumulative physical geodesic length in metres
    intermediate_arcs_added: int                 # Count of intermediate road arcs inserted
    diagnostics: Dict[str, Any]                  # Continuity, gap, and quality metrics


def get_best_edge_arc_id(
    road_network: RoadNetwork,
    u: int,
    v: int
) -> Optional[int]:
    """
    Finds the optimal arc_id connecting node u to node v.
    In case of parallel edges between u and v, selects the one with minimum length_m.
    """
    if not road_network.graph.has_edge(u, v):
        return None

    edge_dict = road_network.graph.get_edge_data(u, v)
    if not edge_dict:
        return None

    # MultiDiGraph stores dict of {key: data_dict}
    best_key = None
    min_len = float("inf")
    for key, data in edge_dict.items():
        length = data.get("length_m", float("inf"))
        if length < min_len:
            min_len = length
            best_key = key

    return best_key


def reconstruct_continuous_route(
    trip_id: str,
    selected_observation_arcs: List[int],
    road_network: RoadNetwork
) -> ReconstructedRoute:
    """
    Reconstructs a continuous directed road route from Viterbi-selected candidate arcs.

    Args:
        trip_id: Trip identifier string.
        selected_observation_arcs: List of arc_id integers corresponding to each GPS observation.
                                   (-1 represents missing or invalid observation).
        road_network: RoadNetwork object for the trip.

    Returns:
        ReconstructedRoute instance containing observation arcs, continuous route arcs,
        gap documentation, and physical metrics.
    """
    n_obs = len(selected_observation_arcs)
    if n_obs == 0:
        return ReconstructedRoute(
            trip_id=trip_id,
            observation_arcs=[],
            continuous_route_arcs=[],
            has_gaps=False,
            gaps=[],
            total_length_m=0.0,
            intermediate_arcs_added=0,
            diagnostics={
                "n_observations": 0,
                "n_continuous_arcs": 0,
                "route_continuity": 1.0,
                "gap_count": 0
            }
        )

    # Filter out leading invalid arcs (-1) to find initial starting point
    valid_indices = [i for i, arc in enumerate(selected_observation_arcs) if arc >= 0]

    if not valid_indices:
        # All arcs are -1
        return ReconstructedRoute(
            trip_id=trip_id,
            observation_arcs=list(selected_observation_arcs),
            continuous_route_arcs=[],
            has_gaps=True,
            gaps=[],
            total_length_m=0.0,
            intermediate_arcs_added=0,
            diagnostics={
                "n_observations": n_obs,
                "n_continuous_arcs": 0,
                "route_continuity": 0.0,
                "gap_count": 0
            }
        )

    continuous_arcs: List[int] = []
    gaps: List[RouteGap] = []
    intermediate_count = 0

    first_idx = valid_indices[0]
    first_arc = selected_observation_arcs[first_idx]
    continuous_arcs.append(first_arc)

    prev_arc = first_arc
    prev_obs_idx = first_idx

    for i in range(1, len(valid_indices)):
        curr_obs_idx = valid_indices[i]
        curr_arc = selected_observation_arcs[curr_obs_idx]

        # Case 1: Consecutive observations on the exact same road arc
        if curr_arc == prev_arc:
            prev_obs_idx = curr_obs_idx
            continue

        # Case 2: Consecutive observations on different arcs
        _, u_prev_end = road_network.get_arc_endpoints(prev_arc)
        v_curr_start, _ = road_network.get_arc_endpoints(curr_arc)

        if u_prev_end == v_curr_start:
            # Direct connection at junction node
            continuous_arcs.append(curr_arc)
        else:
            # Find directed shortest path connecting u_prev_end to v_curr_start
            node_path = road_network.shortest_path(u_prev_end, v_curr_start, weight="length_m")

            if node_path is not None and len(node_path) >= 2:
                # Add intermediate connecting arcs
                for step_idx in range(len(node_path) - 1):
                    u_step = node_path[step_idx]
                    v_step = node_path[step_idx + 1]
                    intermediate_arc = get_best_edge_arc_id(road_network, u_step, v_step)
                    if intermediate_arc is not None:
                        continuous_arcs.append(intermediate_arc)
                        intermediate_count += 1
                continuous_arcs.append(curr_arc)
            else:
                # Disconnected transition: no path exists between arcs
                gaps.append(RouteGap(
                    from_obs_idx=prev_obs_idx,
                    to_obs_idx=curr_obs_idx,
                    from_arc_id=prev_arc,
                    to_arc_id=curr_arc,
                    from_node=u_prev_end,
                    to_node=v_curr_start
                ))
                # Append curr_arc to allow downstream route reconstruction
                continuous_arcs.append(curr_arc)

        prev_arc = curr_arc
        prev_obs_idx = curr_obs_idx

    # Calculate total length in metres
    total_length = 0.0
    for arc_id in continuous_arcs:
        try:
            total_length += road_network.get_arc_length(arc_id)
        except Exception:
            pass

    # Compute observation-level route continuity
    # Share of consecutive observation arcs that are identical or directly connected
    valid_obs_arcs = [a for a in selected_observation_arcs if a >= 0]
    if len(valid_obs_arcs) < 2:
        obs_continuity = 1.0
    else:
        connected_steps = 0
        total_steps = len(valid_obs_arcs) - 1
        for a1, a2 in zip(valid_obs_arcs[:-1], valid_obs_arcs[1:]):
            if a1 == a2:
                connected_steps += 1
            else:
                try:
                    _, u_end = road_network.get_arc_endpoints(a1)
                    v_start, _ = road_network.get_arc_endpoints(a2)
                    if u_end == v_start:
                        connected_steps += 1
                except Exception:
                    pass
        obs_continuity = connected_steps / max(1, total_steps)

    # Continuous route continuity: all consecutive pairs in continuous_arcs should be directly connected
    if len(continuous_arcs) < 2:
        route_continuity = 1.0
    else:
        valid_trans = 0
        for a1, a2 in zip(continuous_arcs[:-1], continuous_arcs[1:]):
            if a1 == a2:
                valid_trans += 1
            else:
                try:
                    _, u_end = road_network.get_arc_endpoints(a1)
                    v_start, _ = road_network.get_arc_endpoints(a2)
                    if u_end == v_start:
                        valid_trans += 1
                except Exception:
                    pass
        route_continuity = valid_trans / max(1, len(continuous_arcs) - 1)

    diagnostics = {
        "n_observations": n_obs,
        "n_valid_observations": len(valid_obs_arcs),
        "n_continuous_arcs": len(continuous_arcs),
        "intermediate_arcs_added": intermediate_count,
        "gap_count": len(gaps),
        "has_gaps": len(gaps) > 0,
        "observation_continuity": float(obs_continuity),
        "route_continuity": float(route_continuity),
        "total_length_km": float(total_length / 1000.0)
    }

    return ReconstructedRoute(
        trip_id=trip_id,
        observation_arcs=list(selected_observation_arcs),
        continuous_route_arcs=continuous_arcs,
        has_gaps=len(gaps) > 0,
        gaps=gaps,
        total_length_m=float(total_length),
        intermediate_arcs_added=intermediate_count,
        diagnostics=diagnostics
    )


def compute_route_similarity_lcs(
    predicted_route: Union[List[int], np.ndarray],
    ground_truth_route: Union[List[int], np.ndarray]
) -> Dict[str, float]:
    """
    Computes Longest Common Subsequence (LCS) precision, recall, and Sequence F1
    between the reconstructed continuous route and the ground-truth .route sequence.

    Matching Team Specification:
    - Repeats are collapsed (several consecutive visits on the same arc count as one visit,
      while legitimate loops/revisits later in the trip are preserved).
    - Precision = LCS(pred, true) / len(pred)
    - Recall = LCS(pred, true) / len(true)
    - Sequence F1 = 2 * P * R / (P + R)
    """
    pred_seq = np.asarray(predicted_route, dtype=np.int64)
    true_seq = np.asarray(ground_truth_route, dtype=np.int64)

    # Collapse consecutive repeats
    def _collapse_repeats(seq: np.ndarray) -> np.ndarray:
        if len(seq) < 2:
            return seq
        keep = np.r_[True, seq[1:] != seq[:-1]]
        return seq[keep]

    p_collapsed = _collapse_repeats(pred_seq)
    t_collapsed = _collapse_repeats(true_seq)

    len_p = len(p_collapsed)
    len_t = len(t_collapsed)

    if len_p == 0 or len_t == 0:
        return {
            "lcs_length": 0,
            "pred_length": len_p,
            "true_length": len_t,
            "route_precision": 0.0,
            "route_recall": 0.0,
            "sequence_f1": 0.0
        }

    # Standard space-efficient LCS DP
    prev = np.zeros(len_t + 1, dtype=np.int32)
    for x in p_collapsed:
        cur = prev.copy()
        cur[1:] = np.maximum(prev[1:], prev[:-1] + (t_collapsed == x))
        prev = np.maximum.accumulate(cur)

    lcs_val = int(prev[-1])
    precision = float(lcs_val / max(len_p, 1))
    recall = float(lcs_val / max(len_t, 1))
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "lcs_length": lcs_val,
        "pred_length": len_p,
        "true_length": len_t,
        "route_precision": precision,
        "route_recall": recall,
        "sequence_f1": f1
    }
