"""
Prediction pipeline used by the API.

IMPORTANT: this module must never read the .route file. It receives a trip whose
`route` has been removed (see backend.main) and returns one predicted arc_id per GPS
observation (-1 where it has no prediction).

Pipeline
    1. Candidates + features   src.member1_pipeline.inference_features (M1), the same
                               code and settings used to build the training data
    2. Candidate classifier    artifacts/<model>.joblib saved by src.metrics.run_experiment
                               (default: Random Forest; override with PATHSNAP_MODEL)
    3. Decoder                 hmm.decoder.decode_trip (M3) if that module exists,
                               otherwise classifier-only: highest-probability arc per point

Decoder contract for M3 (hmm/decoder.py):
    decode_trip(trip, network, candidates, xy, valid) -> np.ndarray of arc_id, one per
    GPS observation (-1 if undecided). `candidates` has columns obs_idx, arc_id, prob and
    every feature column; `network` is src.network.SpatialNetwork; xy are projected
    GPS points in metres and `valid` marks usable ones.
"""
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import TripData
from src.member1_pipeline import inference_features

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", REPO_ROOT / "artifacts" / "rf_model.joblib"))
CANDIDATE_RADIUS_M = 250.0     # must match the training data (src.member1_pipeline defaults)
MAX_CANDIDATES = 30

try:                            # M3's Viterbi decoder, picked up automatically once it exists
    from hmm.decoder import decode_trip as _hmm_decode
except ImportError:
    _hmm_decode = None

DECODER_NAME = "HMM/Viterbi" if _hmm_decode else "classifier-only (argmax per point)"
_MODEL_NAMES = {"rf_model": "Random Forest", "best_model_tuned": "Tuned model", "lgbm_untuned": "LightGBM",
                "logistic_pipeline": "Logistic Regression", "knn_model": "KNN"}
PIPELINE_NAME = f"{_MODEL_NAMES.get(MODEL_PATH.stem, MODEL_PATH.stem)} + {DECODER_NAME}"

_model = None


def load_model():
    """Load the classifier once. The saved file is {'model': estimator, 'features': [...]}."""
    global _model
    if _model is None:
        with warnings.catch_warnings():
            # artifacts may come from a slightly different scikit-learn version; predictions were
            # checked to match the recorded student-test metrics exactly
            warnings.simplefilter("ignore")
            artifact = joblib.load(MODEL_PATH)
        _model = (artifact["model"], list(artifact["features"]))
    return _model


@dataclass
class Prediction:
    arcs: np.ndarray                 # one arc_id per GPS observation, -1 = none
    candidates: pd.DataFrame         # candidates with features and classifier probability
    network: object                  # src.network.SpatialNetwork (reference-free)
    xy: np.ndarray                   # projected GPS points (metres)
    valid: np.ndarray                # usable GPS points
    timings_ms: dict = field(default_factory=dict)


def argmax_decode(candidates: pd.DataFrame, n_obs: int) -> np.ndarray:
    """Classifier-only route: the most probable candidate for each GPS point.
    Candidates are sorted by distance within a point, so ties go to the nearest road."""
    arcs = np.full(n_obs, -1, dtype=np.int64)
    if len(candidates):
        best = candidates.loc[candidates.groupby("obs_idx", sort=False)["prob"].idxmax()]
        arcs[best["obs_idx"].to_numpy()] = best["arc_id"].to_numpy()
    return arcs


def predict(trip: TripData) -> Prediction:
    assert trip.route is None, "predictor must not see the reference route"
    model, features = load_model()
    timings = {}

    t0 = time.perf_counter()
    candidates, network, xy, valid = inference_features(trip, CANDIDATE_RADIUS_M, MAX_CANDIDATES)
    timings["features_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    candidates["prob"] = model.predict_proba(candidates[features])[:, 1] if len(candidates) else []
    timings["classifier_ms"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if _hmm_decode is not None:
        arcs = np.asarray(_hmm_decode(trip, network, candidates, xy, valid), dtype=np.int64)
    else:
        arcs = argmax_decode(candidates, trip.n_points)
    timings["decoder_ms"] = (time.perf_counter() - t0) * 1000

    return Prediction(arcs, candidates, network, xy, valid, {k: round(v, 1) for k, v in timings.items()})


def predict_arcs(trip: TripData) -> np.ndarray:
    return predict(trip).arcs
