"""
PathSnap - Hidden Markov Model (HMM) for Map Matching (Member 3)
================================================================
Defines the spatial Hidden Markov Model for aligning GPS trajectories
to directed road network graphs across all ML candidate scoring models:
1. Logistic Regression
2. K-Nearest Neighbours (KNN)
3. Random Forest (RF)
4. LightGBM (LGBM)
5. Multi-Layer Perceptron (MLP)

HMM Formulation:
----------------
- Hidden States S_t: The candidate directed road arcs A_t = {a_{t,1}, ..., a_{t,K_t}}
  available for GPS observation z_t.
- Observations z_t: GPS measurements (longitude, latitude, timestamp).
- Emission Log-Scores:
    log P(z_t | a_{t,i}) derived from ML model candidate probabilities P(a_{t,i} | features).
- Transition Log-Scores:
    log P(a_{t,j} | a_{t-1,i}) = - |d_net(a_{t-1,i}, a_{t,j}) - d_gps(z_{t-1}, z_t)| / beta
  where d_net is the directed shortest path distance along the road network in metres,
  d_gps is the Great-Circle / WGS84 geodesic distance between consecutive GPS points,
  and beta is a calibrated distance scaling parameter.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import math
import numpy as np
import pandas as pd

from src.road_network import (
    RoadNetwork,
    calculate_geodesic_distance,
)


@dataclass
class HMMConfig:
    """Configuration hyperparameters for HMM map matching."""
    beta: float = 30.0                      # Distance deviation scale parameter in metres
    eps: float = 1e-7                       # Numerical stability clamp for zero probabilities
    unreachable_penalty: float = -10000.0   # Log-score penalty for disconnected graph transitions
    normalize_emissions: bool = True        # Whether to normalize candidate probabilities per observation
    max_cache_size: int = 50000             # Maximum entries for shortest path distance cache
    speed_penalty_threshold_mps: float = 50.0  # Optional speed feasibility threshold (50 m/s ~ 180 km/h)
    speed_penalty_weight: float = 0.0       # Weight for speed violation penalty (0.0 to disable)


def compute_emission_log_scores(
    candidate_probs: np.ndarray,
    normalize: bool = True,
    eps: float = 1e-7
) -> np.ndarray:
    """
    Converts ML model candidate classification probabilities into numerically
    stable log emission scores.

    Args:
        candidate_probs: 1D array of predicted candidate probabilities in [0.0, 1.0].
        normalize: If True, normalize probabilities to sum to 1.0 across candidates
                   before taking log, creating a valid discrete probability distribution.
        eps: Minimum probability floor to prevent log(0) and -inf underflow.

    Returns:
        1D np.ndarray of log emission scores (all <= 0.0).
    """
    probs = np.asarray(candidate_probs, dtype=np.float64)
    if len(probs) == 0:
        return np.empty(0, dtype=np.float64)

    # Sanitize NaN / Inf / negative values
    probs = np.nan_to_num(probs, nan=eps, posinf=1.0, neginf=eps)
    probs = np.clip(probs, eps, 1.0)

    if normalize:
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.full_like(probs, 1.0 / len(probs))
        probs = np.clip(probs, eps, 1.0)

    return np.log(probs)


class TransitionScorer:
    """
    Computes directed road network transition scores between candidate arcs
    for consecutive GPS observations.

    Uses Dijkstra's shortest path on the directed NetworkX MultiDiGraph encapsulated
    by RoadNetwork, with LRU-style distance caching for high-throughput decoding.
    """

    def __init__(self, road_network: RoadNetwork, config: Optional[HMMConfig] = None):
        self.road_network = road_network
        self.config = config or HMMConfig()
        self._distance_cache: Dict[Tuple[int, int], Optional[float]] = {}
        self._ss_cache: Dict[Tuple[int, float], Dict[int, float]] = {}
        self._arc_length_cache: Dict[int, float] = {}
        self._arc_endpoints_cache: Dict[int, Tuple[int, int]] = {}

    def get_arc_endpoints(self, arc_id: int) -> Tuple[int, int]:
        """Retrieves cached (from_node, to_node) for an arc_id."""
        if arc_id not in self._arc_endpoints_cache:
            self._arc_endpoints_cache[arc_id] = self.road_network.get_arc_endpoints(arc_id)
        return self._arc_endpoints_cache[arc_id]

    def get_arc_length(self, arc_id: int) -> float:
        """Retrieves cached physical length in metres for an arc_id."""
        if arc_id not in self._arc_length_cache:
            self._arc_length_cache[arc_id] = self.road_network.get_arc_length(arc_id)
        return self._arc_length_cache[arc_id]

    def shortest_path_distance(self, source_node: int, target_node: int) -> Optional[float]:
        """
        Computes directed shortest path distance in metres between nodes,
        with internal memoization.
        """
        if source_node == target_node:
            return 0.0

        key = (source_node, target_node)
        if key in self._distance_cache:
            return self._distance_cache[key]

        if len(self._distance_cache) >= self.config.max_cache_size:
            self._distance_cache.clear()

        dist = self.road_network.shortest_path_distance(source_node, target_node, weight="length_m")
        self._distance_cache[key] = dist
        return dist

    def get_node_distances_map(self, source_node: int, cutoff: float) -> Dict[int, float]:
        """Runs single-source Dijkstra once and memoizes results."""
        key = (source_node, round(cutoff, -1))
        if key not in self._ss_cache:
            if len(self._ss_cache) >= 10000:
                self._ss_cache.clear()
            self._ss_cache[key] = self.road_network.single_source_shortest_path_distances(
                source_node, cutoff=cutoff, weight="length_m"
            )
        return self._ss_cache[key]

    def compute_network_distance(
        self,
        from_arc_id: int,
        to_arc_id: int,
        from_proj_ratio: Optional[float] = None,
        to_proj_ratio: Optional[float] = None,
        cutoff: Optional[float] = None
    ) -> Optional[float]:
        """
        Computes physical travel distance along directed road arcs in metres.

        Handles:
        1. Same Arc (from_arc_id == to_arc_id):
           - Forward traversal along arc: (to_ratio - from_ratio) * length
           - Backward traversal along one-way arc: Requires traversing the cycle back
        2. Direct Transition (to_node of from_arc == from_node of to_arc):
           - Connecting distance is 0.0, plus partial arc traversals
        3. Multi-hop Directed Path:
           - Shortest path distance between destination of from_arc and origin of to_arc,
             plus partial arc traversals
        """
        len_from = self.get_arc_length(from_arc_id)
        len_to = self.get_arc_length(to_arc_id)

        # Default projection ratios to midpoints if not specified
        p1 = 0.5 if from_proj_ratio is None or np.isnan(from_proj_ratio) else np.clip(from_proj_ratio, 0.0, 1.0)
        p2 = 0.5 if to_proj_ratio is None or np.isnan(to_proj_ratio) else np.clip(to_proj_ratio, 0.0, 1.0)

        # Case 1: Same Arc
        if from_arc_id == to_arc_id:
            if p2 >= p1:
                # Forward progress along the same directed segment
                return float((p2 - p1) * len_from)
            else:
                # Vehicle apparently traveled backwards on a directed one-way road.
                from_u, from_v = self.get_arc_endpoints(from_arc_id)
                if cutoff is not None:
                    d_map = self.get_node_distances_map(from_v, cutoff=cutoff)
                    loop_dist = d_map.get(from_u, None)
                else:
                    loop_dist = self.shortest_path_distance(from_v, from_u)
                if loop_dist is None:
                    return None
                return float((1.0 - p1) * len_from + loop_dist + p2 * len_to)

        # Case 2: Different Arcs
        _, u_end = self.get_arc_endpoints(from_arc_id)
        v_start, _ = self.get_arc_endpoints(to_arc_id)

        # Directed shortest path from exit of from_arc to entry of to_arc
        if u_end == v_start:
            junction_dist = 0.0
        else:
            if cutoff is not None:
                d_map = self.get_node_distances_map(u_end, cutoff=cutoff)
                junction_dist = d_map.get(v_start, None)
            else:
                junction_dist = self.shortest_path_distance(u_end, v_start)

            if junction_dist is None:
                return None

        # Total distance = remainder of from_arc + junction path + traversed portion of to_arc
        net_dist = (1.0 - p1) * len_from + junction_dist + p2 * len_to
        return float(net_dist)

    def transition_score(
        self,
        from_arc_id: int,
        to_arc_id: int,
        d_gps: float,
        dt_s: Optional[float] = None,
        from_proj_ratio: Optional[float] = None,
        to_proj_ratio: Optional[float] = None,
        cutoff: Optional[float] = None
    ) -> float:
        """
        Calculates the HMM transition log-score between two candidate arcs.
        """
        d_net = self.compute_network_distance(
            from_arc_id, to_arc_id,
            from_proj_ratio=from_proj_ratio,
            to_proj_ratio=to_proj_ratio,
            cutoff=cutoff
        )

        if d_net is None:
            return self.config.unreachable_penalty

        delta_d = abs(d_net - d_gps)
        score = - (delta_d / self.config.beta)

        if (
            self.config.speed_penalty_weight > 0.0
            and dt_s is not None
            and dt_s > 0.0
        ):
            speed_mps = d_net / dt_s
            if speed_mps > self.config.speed_penalty_threshold_mps:
                excess = speed_mps - self.config.speed_penalty_threshold_mps
                score -= self.config.speed_penalty_weight * (excess / 10.0)

        return float(score)

    def compute_transition_matrix(
        self,
        prev_candidates: List[Dict[str, Any]],
        curr_candidates: List[Dict[str, Any]],
        prev_coords: Tuple[float, float],
        curr_coords: Tuple[float, float],
        dt_s: Optional[float] = None
    ) -> np.ndarray:
        """
        Computes transition log-score matrix of shape (len(prev_candidates), len(curr_candidates)).
        """
        k_prev = len(prev_candidates)
        k_curr = len(curr_candidates)
        if k_prev == 0 or k_curr == 0:
            return np.empty((k_prev, k_curr), dtype=np.float64)

        d_gps = calculate_geodesic_distance(
            prev_coords[0], prev_coords[1],
            curr_coords[0], curr_coords[1]
        )
        cutoff = max(1500.0, 10.0 * d_gps)

        matrix = np.full((k_prev, k_curr), self.config.unreachable_penalty, dtype=np.float64)

        for i, c_prev in enumerate(prev_candidates):
            arc_prev = int(c_prev.get("arc_id", c_prev.get("candidate_arc_id", -1)))
            p_prev = c_prev.get("projection_ratio", None)

            for j, c_curr in enumerate(curr_candidates):
                arc_curr = int(c_curr.get("arc_id", c_curr.get("candidate_arc_id", -1)))
                p_curr = c_curr.get("projection_ratio", None)

                score = self.transition_score(
                    arc_prev, arc_curr,
                    d_gps=d_gps,
                    dt_s=dt_s,
                    from_proj_ratio=p_prev,
                    to_proj_ratio=p_curr,
                    cutoff=cutoff
                )
                matrix[i, j] = score

        return matrix
