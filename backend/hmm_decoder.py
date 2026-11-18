"""
PathSnap - Backend HMM & Route Reconstruction Integration (Member 3)
====================================================================
Provides the service interface for Member 4's FastAPI backend and Streamlit frontend.

Functions:
- `viterbi_map_match_trip(trip, candidates_df, model_name, beta)`:
    Runs emission scoring, Dijkstra transition scoring, Viterbi decoding,
    and continuous route reconstruction.
- `predict_arcs_viterbi(trip, model_name)`:
    Drop-in replacement for backend.predictor.predict_arcs returning an array
    of observation-aligned arc IDs, decoded via global HMM consistency.
- `get_continuous_route(trip, candidates_df, model_name)`:
    Returns complete continuous road arc sequence and Polylines for Folium / Streamlit.
"""

from typing import Dict, List, Optional, Any, Union
import os
import joblib
import numpy as np
import pandas as pd

from src.data_loader import TripData
from src.road_network import build_road_network, RoadNetwork
from src.hmm import HMMConfig, TransitionScorer
from src.viterbi import viterbi_decode
from src.route_reconstruction import reconstruct_continuous_route
from maps.make_map import arcs_to_polylines


# Default artifacts location
ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


def load_model_artifact(model_name: str = "logistic"):
    """Loads trained model pipeline or classifier from artifacts/."""
    m_name = model_name.lower()
    if m_name in ["logistic", "logreg"]:
        path = os.path.join(ARTIFACTS_DIR, "logistic_pipeline.joblib")
    elif m_name in ["rf", "random_forest"]:
        path = os.path.join(ARTIFACTS_DIR, "rf_model.joblib")
    elif m_name in ["lgbm", "lightgbm"]:
        path = os.path.join(ARTIFACTS_DIR, "lgbm_untuned.joblib")
    else:
        path = os.path.join(ARTIFACTS_DIR, f"{m_name}.joblib")

    if os.path.exists(path):
        return joblib.load(path)
    return None


def viterbi_map_match_trip(
    trip: TripData,
    candidates_df: pd.DataFrame,
    beta: float = 30.0,
    model_name: str = "logistic"
) -> Dict[str, Any]:
    """
    Core backend entry point for Member 4.

    Args:
        trip: TripData instance (trip.route may be None).
        candidates_df: DataFrame with candidate features or predictions.
                       Must have ['obs_idx', 'arc_id'] and optional ['prob', 'projection_ratio'].
        beta: HMM transition distance parameter in metres.
        model_name: Name of model for reporting.

    Returns:
        Dictionary with:
        - 'observation_arcs': np.ndarray of arc IDs (len == n_points)
        - 'continuous_route_arcs': List of complete arc IDs including intermediate segments
        - 'polylines': List of polylines formatted for Folium / Leaflet / Streamlit
        - 'diagnostics': Detailed execution and continuity metrics
    """
    assert trip is not None, "trip must be provided"

    # 1. Build road network
    road_network = build_road_network(trip.nodes, trip.arcs, trip_id=trip.trip_id)

    # 2. Extract observations
    n_points = trip.n_points
    observations = []
    track_lon = trip.track["longitude"].to_numpy()
    track_lat = trip.track["latitude"].to_numpy()
    track_t = trip.track["timestamp"].to_numpy() if "timestamp" in trip.track.columns else None

    for i in range(n_points):
        observations.append({
            "obs_idx": i,
            "longitude": float(track_lon[i]),
            "latitude": float(track_lat[i]),
            "timestamp": float(track_t[i]) if track_t is not None else None
        })

    # 3. Format candidate groups
    candidates_by_obs = [[] for _ in range(n_points)]
    c_df = candidates_df.copy()
    if "candidate_arc_id" in c_df.columns and "arc_id" not in c_df.columns:
        c_df = c_df.rename(columns={"candidate_arc_id": "arc_id"})

    for obs_idx, group in c_df.groupby("obs_idx"):
        idx = int(obs_idx)
        if idx < n_points:
            cands = []
            for _, r in group.iterrows():
                cands.append({
                    "arc_id": int(r["arc_id"]),
                    "prob": float(r["prob"]) if "prob" in r and pd.notna(r["prob"]) else 0.5,
                    "projection_ratio": float(r["projection_ratio"]) if "projection_ratio" in r and pd.notna(r["projection_ratio"]) else None
                })
            candidates_by_obs[idx] = cands

    # 4. HMM & Viterbi
    cfg = HMMConfig(beta=beta)
    scorer = TransitionScorer(road_network=road_network, config=cfg)
    viterbi_res = viterbi_decode(
        trip_id=trip.trip_id,
        observations=observations,
        candidates_by_obs=candidates_by_obs,
        transition_scorer=scorer,
        config=cfg
    )

    # 5. Continuous Route Reconstruction
    reconstructed = reconstruct_continuous_route(
        trip_id=trip.trip_id,
        selected_observation_arcs=viterbi_res.selected_arcs,
        road_network=road_network
    )

    # 6. Generate Polylines for Folium / Streamlit
    polylines = arcs_to_polylines(trip, reconstructed.continuous_route_arcs)

    return {
        "trip_id": trip.trip_id,
        "model_name": model_name,
        "observation_arcs": np.array(viterbi_res.selected_arcs, dtype=int),
        "continuous_route_arcs": reconstructed.continuous_route_arcs,
        "polylines": polylines,
        "diagnostics": {
            **viterbi_res.diagnostics,
            **reconstructed.diagnostics,
            "log_likelihood": viterbi_res.log_likelihood
        }
    }
