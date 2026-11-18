"""
Prediction pipeline used by the API.

IMPORTANT: this module must never read the .route file. It receives a trip
whose `route` has been removed (see backend.main) and returns one predicted arc_id
per GPS observation (-1 where it has no prediction).

Pipeline Architecture (Member 1 -> Member 2 -> Member 3):
    1. Spatial candidate generation & feature extraction (src.candidates, src.features)
    2. ML classifier probability scoring (Random Forest / Logistic Regression)
    3. Global HMM Viterbi decoding & continuous route reconstruction (backend.hmm_decoder)
"""
import logging
import os
import joblib
import numpy as np
from scipy.spatial import cKDTree

from src.data_loader import TripData
from backend.hmm_decoder import viterbi_map_match_trip
from src.network import build_network
from src.features import make_features

log = logging.getLogger("pathsnap.predictor")

EARTH_RADIUS_M = 6_371_000.0
PIPELINE_NAME = "HMM + Viterbi (Member 3) with Random Forest"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

# Preload best trained classifier (Random Forest preferred, fallback to Logistic)
_MODEL_PKG = None
_MODEL_TYPE = "None"

try:
    rf_path = os.path.join(ARTIFACTS_DIR, "rf_model.joblib")
    lr_path = os.path.join(ARTIFACTS_DIR, "logistic_pipeline.joblib")
    if os.path.exists(rf_path):
        _MODEL_PKG = joblib.load(rf_path)
        _MODEL_TYPE = "Random Forest"
        PIPELINE_NAME = "HMM + Viterbi (Member 3) with Random Forest"
    elif os.path.exists(lr_path):
        _MODEL_PKG = joblib.load(lr_path)
        _MODEL_TYPE = "Logistic Regression"
        PIPELINE_NAME = "HMM + Viterbi (Member 3) with Logistic Regression"
except Exception as e:
    log.warning(f"Failed to preload model artifact: {e}")


def _xy(lon, lat, lon0, lat0):
    x = np.radians(lon - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def _nearest_arc_fallback(trip: TripData) -> np.ndarray:
    """Fast fallback finding the nearest road arc midpoint to each GPS point."""
    lon0, lat0 = trip.track["longitude"].mean(), trip.track["latitude"].mean()
    nodes = _xy(trip.nodes["longitude"].to_numpy(), trip.nodes["latitude"].to_numpy(), lon0, lat0)
    arcs = trip.arcs[["from_node_id", "to_node_id"]].to_numpy()
    gps = _xy(trip.track["longitude"].to_numpy(), trip.track["latitude"].to_numpy(), lon0, lat0)

    mid = (nodes[arcs[:, 0]] + nodes[arcs[:, 1]]) / 2
    _, idx = cKDTree(mid).query(gps)
    return idx.astype(int)


def predict_arcs(trip: TripData) -> np.ndarray:
    """
    Predicts one arc_id per observation using HMM Viterbi decoding over ML emission scores.
    Guarantees strict compliance with the API contract and runtime safety.
    """
    assert trip.route is None, "predictor must not see the reference route"

    fallback = None

    try:
        # 1. Build spatial network & extract candidate features
        spatial_net = build_network(trip)
        feat_df, _, _ = make_features(trip, spatial_net, radius=250.0, k=30)

        if feat_df is None or len(feat_df) == 0:
            return _nearest_arc_fallback(trip)

        # 2. Score candidate emission probabilities with trained model
        if _MODEL_PKG is not None and "model" in _MODEL_PKG and "features" in _MODEL_PKG:
            features = _MODEL_PKG["features"]
            X = feat_df[features].fillna(0.0)
            model = _MODEL_PKG["model"]
            probs = model.predict_proba(X)[:, 1]
            feat_df["prob"] = probs
        elif "distance_m" in feat_df.columns:
            # Distance-based heuristic if model artifact is unavailable
            feat_df["prob"] = np.exp(-feat_df["distance_m"].to_numpy() / 50.0)
        else:
            feat_df["prob"] = 0.5

        # 3. Global HMM Viterbi map matching & route reconstruction
        result = viterbi_map_match_trip(
            trip=trip,
            candidates_df=feat_df,
            beta=30.0,
            model_name=_MODEL_TYPE
        )

        pred = result["observation_arcs"]

        # Ensure all observations have valid arc indices [0, n_arcs)
        if len(pred) == trip.n_points:
            invalid_mask = (pred < 0) | (pred >= trip.n_arcs)
            if np.any(invalid_mask):
                if fallback is None:
                    fallback = _nearest_arc_fallback(trip)
                pred[invalid_mask] = fallback[invalid_mask]
            return pred.astype(int)

    except Exception as exc:
        log.warning(f"HMM Viterbi prediction encountered exception on {trip.trip_id}: {exc}; falling back to nearest arc.")

    # Graceful fallback to guarantee zero crash and API contract compliance
    return _nearest_arc_fallback(trip)
