"""
PathSnap - Map Matching & Route Reconstruction Pipeline (Member 3)
==================================================================
Provides high-level, end-to-end map matching integrating:
1. Candidate predictions from any of the 5 ML models (LogReg, KNN, RF, LGBM, MLP)
2. Directed Road Network representation (RoadNetwork MultiDiGraph)
3. Spatial Hidden Markov Model (HMM emission & transition scoring)
4. Dynamic programming Viterbi decoding
5. Continuous route reconstruction with intermediate road segments
6. Evaluation and comparison against ground-truth routes and baselines
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import time
import os
import json
import numpy as np
import pandas as pd

from src.road_network import (
    RoadNetwork,
    build_road_network,
    load_and_build_road_network,
)
from src.hmm import HMMConfig, TransitionScorer, compute_emission_log_scores
from src.viterbi import ViterbiResult, viterbi_decode
from src.route_reconstruction import (
    ReconstructedRoute,
    reconstruct_continuous_route,
    compute_route_similarity_lcs
)
from src.data_loader import TripData, load_trip, load_route


def match_trip_candidates(
    trip_id: str,
    candidates_df: pd.DataFrame,
    road_network: RoadNetwork,
    observations_df: Optional[pd.DataFrame] = None,
    config: Optional[HMMConfig] = None,
    ground_truth_route: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Performs complete HMM map matching and route reconstruction for a single trip
    given candidate feature/prediction rows and a RoadNetwork.

    Args:
        trip_id: Trip identifier string (e.g. '00000000').
        candidates_df: DataFrame containing candidate rows for this trip.
                       Expected columns: ['obs_idx', 'arc_id', 'prob']
                       Optional columns: ['projection_ratio', 'model_name']
        road_network: RoadNetwork instance for this trip.
        observations_df: Optional DataFrame of GPS observations.
                         Columns: ['obs_idx', 'longitude', 'latitude', 'timestamp']
                         If None, observations are derived from candidates.
        config: HMMConfig hyperparameters (beta, penalties, etc.).
        ground_truth_route: Optional list of ground truth arc IDs for evaluation.

    Returns:
        Dictionary containing:
        - 'trip_id': str
        - 'viterbi_result': ViterbiResult
        - 'reconstructed_route': ReconstructedRoute
        - 'baseline_observation_arcs': List[int] (independent Top-1 argmax per obs)
        - 'baseline_reconstructed_route': ReconstructedRoute
        - 'comparison_metrics': Dict[str, Any]
    """
    t_start = time.perf_counter()
    cfg = config or HMMConfig()

    # Filter candidates to this trip_id
    if "trip_id" in candidates_df.columns:
        c_df = candidates_df[candidates_df["trip_id"].astype(str) == str(trip_id)].copy()
    else:
        c_df = candidates_df.copy()

    # Normalize column names
    col_map = {}
    if "candidate_arc_id" in c_df.columns and "arc_id" not in c_df.columns:
        col_map["candidate_arc_id"] = "arc_id"
    if col_map:
        c_df = c_df.rename(columns=col_map)

    if len(c_df) == 0:
        raise ValueError(f"No candidates found for trip_id {trip_id}")

    # Determine unique observations
    unique_obs_indices = sorted(c_df["obs_idx"].unique())
    max_obs_idx = max(unique_obs_indices)
    total_obs = max_obs_idx + 1

    # Build observations list
    observations: List[Dict[str, Any]] = []
    if observations_df is not None:
        obs_sorted = observations_df.sort_values("obs_idx").reset_index(drop=True)
        for _, row in obs_sorted.iterrows():
            observations.append({
                "obs_idx": int(row["obs_idx"]),
                "longitude": float(row["longitude"]),
                "latitude": float(row["latitude"]),
                "timestamp": float(row["timestamp"]) if "timestamp" in row and pd.notna(row["timestamp"]) else None
            })
    else:
        # Fallback: approximate GPS coordinates from candidate midpoint/node geometries
        # or use first candidate coordinates if present
        for obs_idx in range(total_obs):
            sub = c_df[c_df["obs_idx"] == obs_idx]
            if len(sub) > 0 and "longitude" in sub.columns and "latitude" in sub.columns:
                lon = float(sub["longitude"].iloc[0])
                lat = float(sub["latitude"].iloc[0])
            elif len(sub) > 0:
                # Use first candidate arc start node coords
                arc_0 = int(sub["arc_id"].iloc[0])
                u, _ = road_network.get_arc_endpoints(arc_0)
                lon, lat = road_network.get_node_coordinates(u)
            else:
                lon, lat = 0.0, 0.0

            observations.append({
                "obs_idx": obs_idx,
                "longitude": lon,
                "latitude": lat,
                "timestamp": None
            })

    # Group candidates by observation index
    candidates_by_obs: List[List[Dict[str, Any]]] = [[] for _ in range(total_obs)]
    # Group candidates
    for obs_idx, group in c_df.groupby("obs_idx"):
        obs_i = int(obs_idx)
        if obs_i < total_obs:
            cands = []
            for _, r in group.iterrows():
                c_dict = {
                    "arc_id": int(r["arc_id"]),
                    "prob": float(r["prob"]) if "prob" in r and pd.notna(r["prob"]) else 0.5,
                    "projection_ratio": float(r["projection_ratio"]) if "projection_ratio" in r and pd.notna(r["projection_ratio"]) else None
                }
                cands.append(c_dict)
            candidates_by_obs[obs_i] = cands

    # 1. Baseline: Independent Top-1 Candidate Selection per Observation
    baseline_obs_arcs: List[int] = []
    for obs_i in range(total_obs):
        cands = candidates_by_obs[obs_i]
        if not cands:
            baseline_obs_arcs.append(-1)
        else:
            # Pick candidate with highest raw probability
            best_cand = max(cands, key=lambda c: c["prob"])
            baseline_obs_arcs.append(best_cand["arc_id"])

    # Reconstruct baseline route
    baseline_reconstructed = reconstruct_continuous_route(
        trip_id=trip_id,
        selected_observation_arcs=baseline_obs_arcs,
        road_network=road_network
    )

    # 2. HMM & Viterbi Decoding
    scorer = TransitionScorer(road_network=road_network, config=cfg)
    viterbi_res = viterbi_decode(
        trip_id=trip_id,
        observations=observations,
        candidates_by_obs=candidates_by_obs,
        transition_scorer=scorer,
        config=cfg
    )

    # 3. Continuous Route Reconstruction
    hmm_reconstructed = reconstruct_continuous_route(
        trip_id=trip_id,
        selected_observation_arcs=viterbi_res.selected_arcs,
        road_network=road_network
    )

    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    # 4. Comparative Metrics
    comparison: Dict[str, Any] = {
        "trip_id": trip_id,
        "n_observations": total_obs,
        "total_time_ms": total_time_ms,
        "baseline_observation_continuity": baseline_reconstructed.diagnostics["observation_continuity"],
        "hmm_observation_continuity": hmm_reconstructed.diagnostics["observation_continuity"],
        "baseline_route_continuity": baseline_reconstructed.diagnostics["route_continuity"],
        "hmm_route_continuity": hmm_reconstructed.diagnostics["route_continuity"],
        "baseline_gaps": len(baseline_reconstructed.gaps),
        "hmm_gaps": len(hmm_reconstructed.gaps),
        "baseline_length_km": baseline_reconstructed.diagnostics["total_length_km"],
        "hmm_length_km": hmm_reconstructed.diagnostics["total_length_km"],
        "viterbi_log_likelihood": viterbi_res.log_likelihood,
        "viterbi_disconnected_transitions": viterbi_res.diagnostics["disconnected_transitions"]
    }

    # Evaluate ground truth similarity if route is provided
    if ground_truth_route is not None and len(ground_truth_route) > 0:
        base_sim = compute_route_similarity_lcs(
            baseline_reconstructed.continuous_route_arcs,
            ground_truth_route
        )
        hmm_sim = compute_route_similarity_lcs(
            hmm_reconstructed.continuous_route_arcs,
            ground_truth_route
        )

        comparison.update({
            "ground_truth_arcs": len(ground_truth_route),
            "baseline_route_precision": base_sim["route_precision"],
            "baseline_route_recall": base_sim["route_recall"],
            "baseline_sequence_f1": base_sim["sequence_f1"],
            "hmm_route_precision": hmm_sim["route_precision"],
            "hmm_route_recall": hmm_sim["route_recall"],
            "hmm_sequence_f1": hmm_sim["sequence_f1"]
        })

    return {
        "trip_id": trip_id,
        "viterbi_result": viterbi_res,
        "reconstructed_route": hmm_reconstructed,
        "baseline_observation_arcs": baseline_obs_arcs,
        "baseline_reconstructed_route": baseline_reconstructed,
        "comparison_metrics": comparison
    }
